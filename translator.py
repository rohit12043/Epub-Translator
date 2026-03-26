import hashlib
import logging
import re
import time
import json
import os
import random
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from filelock import FileLock
from google.api_core import exceptions
from prompts import create_ner_prompt, create_translation_prompt
from exceptions import ProhibitedContentError
from glossary import GlossaryManager
from rate_limiter import ModelLimits, ModelType
from scheduler import AdaptiveTranslationScheduler
from text_utils import extract_whitespace_info, reconstruct_whitespace
from validator import TranslationValidator
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
from google.genai import errors


class GeminiTranslator:
    HISTORY_FILENAME = "translation_history.json"
    LOCK_FILENAME    = "translation_history.json.lock"
    GLOSSARY_FILENAME = "glossary.json"

    # Save history every N successful requests
    _HISTORY_SAVE_INTERVAL = 5

    def __init__(self, api_key: str, max_translations_per_session: int = 2000,
                 checkpoint_dir: str = "checkpoints", stop_event=None):
        self.logger = logging.getLogger(__name__)
        if not api_key:
            raise ValueError("API key must be provided.")
        try:
            self.client = genai.Client(api_key=api_key)
            self.logger.info("Google Generative AI Client configured successfully.")
        except Exception as e:
            self.logger.error(f"Failed to configure Generative AI Client: {e}")
            raise

        self.chapter_ner_cache: Dict[str, List[dict]] = {}
        self.max_translations_per_session = max_translations_per_session
        self.translation_count = 0
        self.scheduler = AdaptiveTranslationScheduler(self)
        self.stop_event = stop_event
        self.RETRY_ATTEMPTS = 5
        self.MIN_RESPONSE_LENGTH = 5
        self.current_chapter_id = None
        self.chunk_count_in_chapter = 0
        self._history_save_counter = 0

        self.checkpoint_dir = Path(checkpoint_dir)
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.HISTORY_FILE  = str(self.checkpoint_dir / self.HISTORY_FILENAME)
            self.LOCK_FILE     = str(self.checkpoint_dir / self.LOCK_FILENAME)
            self.GLOSSARY_FILE = str(self.checkpoint_dir / self.GLOSSARY_FILENAME)
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint dir: {e}. Using current dir.")
            self.HISTORY_FILE  = self.HISTORY_FILENAME
            self.LOCK_FILE     = self.LOCK_FILENAME
            self.GLOSSARY_FILE = self.GLOSSARY_FILENAME

        self.glossary_manager = GlossaryManager(self.GLOSSARY_FILE)
        self.validator = TranslationValidator()
        self.active_placeholders: Dict[str, Tuple[str, str]] = {}

        self.models: Dict[ModelType, Dict] = {
            ModelType.GEMINI_2_5_FLASH: {
                'model_name': 'gemini-2.5-flash',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=20, rpm=5,
                                      max_chapters_per_day=20, tokens_per_chapter=3200,
                                      max_input_tokens=1000000),
                'priority': 1, 'available': True,
            },
            ModelType.GEMINI_2_5_FLASH_LITE: {
                'model_name': 'gemini-2.5-flash-lite',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=20, rpm=10,
                                      max_chapters_per_day=20, tokens_per_chapter=3200,
                                      max_input_tokens=1000000),
                'priority': 3, 'available': True,
            },
            ModelType.GEMINI_2_5_PRO: {
                'model_name': 'gemini-2.5-pro',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=0, rpm=0,
                                      max_chapters_per_day=0, tokens_per_chapter=3200,
                                      max_input_tokens=2000000),
                'priority': 5, 'available': False,
            },
            ModelType.GEMINI_3_FLASH: {
                'model_name': 'gemini-3-flash-preview',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=20, rpm=5,
                                      max_chapters_per_day=20, tokens_per_chapter=3200,
                                      max_input_tokens=1000000),
                'priority': 2, 'available': True,
            },
            ModelType.GEMINI_3_1_PRO: {
                'model_name': 'gemini-3.1-pro',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=0, rpm=0,
                                      max_chapters_per_day=0, tokens_per_chapter=3200,
                                      max_input_tokens=2000000),
                'priority': 6, 'available': False,
            },
            ModelType.GEMINI_3_1_FLASH_LITE: {
                'model_name': 'gemini-3.1-flash-lite-preview',
                'model_instance': None,
                'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=500, rpm=15,
                                      max_chapters_per_day=500, tokens_per_chapter=3200,
                                      max_input_tokens=1000000),
                'priority': 4, 'available': True,
            },
        }
        self._initialize_models()
        self._load_history()
        self.model_priority_order = sorted(
            [mt for mt, info in self.models.items() if info.get('model_instance') is not None],
            key=lambda x: self.models[x]['priority'],
        )
        if not self.model_priority_order:
            self.logger.error("No model instances initialised. Translation will not be possible.")

    def _interruptible_sleep(self, seconds: float):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event and self.stop_event.is_set():
                return
            time.sleep(0.2)

    @staticmethod
    def _min_request_interval(limits: ModelLimits) -> float:
        """
        Minimum seconds between requests to stay within RPM.
        Uses 95 % of the theoretical interval as a small safety buffer.
        """
        if limits.rpm <= 0:
            return 1.0
        return max(1.0, (60.0 / limits.rpm) * 0.95)

    def _maybe_save_history(self, force: bool = False):
        """Save history periodically."""
        self._history_save_counter += 1
        if force or self._history_save_counter >= self._HISTORY_SAVE_INTERVAL:
            self._save_history()
            self._history_save_counter = 0

    def _initialize_models(self):
        for model_type, model_info in self.models.items():
            model_name = model_info.get('model_name')
            if not model_name:
                model_info['available'] = False
                continue
            try:
                model_info['model_instance'] = model_name
                model_info['available'] = True
                self.logger.info(f"Configured model reference: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to configure {model_name}: {e}")
                model_info['available'] = False

    def _load_history(self):
        try:
            if os.path.exists(self.HISTORY_FILE):
                with FileLock(self.LOCK_FILE, timeout=15):
                    with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                for model_type_str, model_data in history.items():
                    try:
                        model_type = ModelType(model_type_str)
                        if model_type in self.models:
                            if 'limits' in model_data:
                                self.models[model_type]['limits'] = ModelLimits.from_dict(
                                    model_data['limits']
                                )
                            if 'available' in model_data:
                                self.models[model_type]['available'] = model_data['available']
                            if 'priority' in model_data:
                                self.models[model_type]['priority'] = model_data['priority']
                    except (ValueError, Exception) as e:
                        self.logger.warning(f"Error loading history for {model_type_str}: {e}")
                self.logger.info("History loaded.")
            else:
                self.logger.info("No history file — starting fresh.")
                self._save_history()
        except FileLock.Timeout:
            self.logger.error("Timeout acquiring lock for history file.")
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to load history: {e}. Using defaults.")

    def _save_history(self):
        try:
            history = {}
            for model_type, model_info in self.models.items():
                history[model_type.value] = {
                    'limits': model_info['limits'].to_dict(),
                    'available': model_info.get('available', True),
                    'priority': model_info.get('priority', model_type.value),
                }
            history_dir = os.path.dirname(self.HISTORY_FILE)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)
            temp_file = self.HISTORY_FILE + ".tmp"
            with FileLock(self.LOCK_FILE, timeout=15):
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, self.HISTORY_FILE)
            self.logger.debug("History saved.")
        except FileLock.Timeout:
            self.logger.error("Timeout acquiring lock when saving history.")
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")

    def _calculate_optimal_model_for_chapters(self, estimated_chapters: int) -> List[ModelType]:
        """
        Rank models by remaining daily capacity × TPM / priority.
        """
        current_time = time.time()
        # Perform daily resets before calculating capacity
        for model_info in self.models.values():
            limits = model_info['limits']
            if current_time - limits.last_daily_reset >= 86400:
                self.logger.info(f"Daily reset for {model_info.get('model_name', '?')}")
                limits.daily_requests = 0
                limits.last_daily_reset = current_time
        self._save_history()

        efficiency_scores: List[Tuple[ModelType, float]] = []
        for model_type, model_info in self.models.items():
            if not model_info.get('available', False) or model_info.get('model_instance') is None:
                continue
            limits = model_info['limits']
            # Remaining capacity by RPD
            remaining_by_rpd    = max(0, limits.rpd - limits.daily_requests)
            # Remaining capacity by configured max_chapters_per_day
            remaining_by_config = max(0, limits.max_chapters_per_day - limits.daily_requests)
            remaining_daily     = min(remaining_by_rpd, remaining_by_config)

            if remaining_daily <= 0:
                continue

            priority = model_info.get('priority', 1)
            efficiency = (remaining_daily * limits.tpm) / max(priority, 0.001)
            efficiency_scores.append((model_type, efficiency))

        efficiency_scores.sort(key=lambda x: x[1], reverse=True)

        if not efficiency_scores:
            self.logger.warning("No models with remaining daily capacity.")
            self.model_priority_order = []
            return []

        ranked = [mt for mt, _ in efficiency_scores]
        self.logger.info("Model efficiency ranking:")
        for model_type, efficiency in efficiency_scores:
            limits = self.models[model_type]['limits']
            rem_rpd    = max(0, limits.rpd - limits.daily_requests)
            rem_config = max(0, limits.max_chapters_per_day - limits.daily_requests)
            rem_daily  = min(rem_rpd, rem_config)
            self.logger.info(
                f"  {model_type.value}: eff={efficiency:.1f}, "
                f"rem_rpd={rem_rpd}, rem_config_chaps={rem_config}, rem_daily={rem_daily}"
            )
        self.model_priority_order = ranked
        return ranked

    def _get_best_available_model(self, estimated_tokens: int
                                   ) -> Optional[Tuple[ModelType, str, ModelLimits]]:
        current_time = time.time()
        for model_type in self.model_priority_order:
            model_info = self.models.get(model_type)
            if not model_info:
                continue
            limits = model_info['limits']
            model_instance = model_info.get('model_instance')

            # Reset minute window
            if current_time - limits.last_reset_time >= 60:
                limits.requests_made = 0
                limits.tokens_used   = 0
                limits.last_reset_time = current_time

            # Reset daily window
            if current_time - limits.last_daily_reset >= 86400:
                limits.daily_requests  = 0
                limits.last_daily_reset = current_time

            if (model_info.get('available', False)
                    and model_instance is not None
                    and limits.requests_made < limits.rpm
                    and limits.tokens_used + estimated_tokens <= limits.tpm
                    and limits.daily_requests < limits.rpd
                    and estimated_tokens <= limits.max_input_tokens):
                return model_type, model_instance, limits
        return None

    def _wait_for_available_model(self, required_tokens: int
                                   ) -> Optional[Tuple[ModelType, str, ModelLimits]]:
        max_wait   = 600
        start_time = time.time()
        wait_seq   = [5, 10, 20, 40, 60, 90, 120]
        attempt_i  = 0

        while time.time() - start_time < max_wait:
            if self.stop_event and self.stop_event.is_set():
                return None

            best = self._get_best_available_model(required_tokens)
            if best:
                mt, _, lim = best
                self.logger.info(
                    f"Selected {mt.value}: "
                    f"{lim.rpm - lim.requests_made} RPM / "
                    f"{lim.rpd - lim.daily_requests} RPD remaining"
                )
                return best

            # Calculate time until the earliest minute-window reset
            current_time  = time.time()
            time_to_reset = 60.0
            for model_info in self.models.values():
                if model_info.get('model_instance'):
                    elapsed_in_window = (current_time - model_info['limits'].last_reset_time) % 60
                    remaining = 60.0 - elapsed_in_window
                    time_to_reset = min(time_to_reset, remaining)

            base  = wait_seq[min(attempt_i, len(wait_seq) - 1)]
            sleep = min(base, time_to_reset) * (0.9 + 0.2 * random.random())
            self.logger.warning(
                f"No model available (attempt {attempt_i + 1}), "
                f"sleeping {sleep:.1f}s (reset in {time_to_reset:.1f}s)."
            )
            time.sleep(sleep)
            attempt_i += 1
            if attempt_i % 3 == 0:
                self._save_history()

        self.logger.error(f"Could not acquire a model after {max_wait}s.")
        return None

    def estimate_tokens(self, text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        return max(1, int(len(text) / 3.5) + 120)

    def count_tokens(self, text: str, model: str = None) -> int:
        if model:
            try:
                resp = self.client.models.count_tokens(model=model, contents=text)
                return resp.total_tokens
            except Exception as e:
                self.logger.warning(f"Token count API failed: {e}")
        return self.estimate_tokens(text)

    def _split_large_chunk(self, text: str, max_tokens: int) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        if self.estimate_tokens(text) <= max_tokens:
            return [text]

        safety = 50
        text = re.sub(r'\.{3,}', '___ELLIPSIS___', text)
        text = re.sub(r'…',       '___ELLIPSIS___', text)

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        paragraphs = re.split(r'(\n\s*\n+)', text)
        para_pairs = [
            (paragraphs[i], paragraphs[i + 1] if i + 1 < len(paragraphs) else '')
            for i in range(0, len(paragraphs), 2)
        ]

        for para, sep in para_pairs:
            p_tokens = self.estimate_tokens(para + sep)
            if current_tokens > 0 and current_tokens + p_tokens + safety > max_tokens:
                chunk = ''.join(current_parts).strip().replace('___ELLIPSIS___', '...')
                if chunk:
                    chunks.append(chunk)
                current_parts = [para + sep]
                current_tokens = p_tokens
            else:
                current_parts.append(para + sep)
                current_tokens += p_tokens

        if current_parts:
            chunk = ''.join(current_parts).strip().replace('___ELLIPSIS___', '...')
            if chunk:
                chunks.append(chunk)

        # Further split by sentences if any chunk is still too large
        final: List[str] = []
        for chunk in chunks:
            if self.estimate_tokens(chunk) <= max_tokens:
                final.append(chunk)
                continue
            chunk = re.sub(r'\.{3,}', '___ELLIPSIS___', chunk)
            chunk = re.sub(r'…',       '___ELLIPSIS___', chunk)
            parts = re.split(r'([.!?。？！]+)', chunk)
            sents: List[str] = []
            for i in range(0, len(parts) - 1, 2):
                s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else '')
                if s.strip():
                    sents.append(s)
            if len(parts) % 2 == 1 and parts[-1].strip():
                sents.append(parts[-1])

            cur_sub: List[str] = []
            cur_sub_t = 0
            for sent in sents:
                st = self.estimate_tokens(sent)
                if cur_sub_t > 0 and cur_sub_t + st + safety > max_tokens:
                    sub = ''.join(cur_sub).strip().replace('___ELLIPSIS___', '...')
                    if sub:
                        final.append(sub)
                    cur_sub   = [sent]
                    cur_sub_t = st
                else:
                    cur_sub.append(sent)
                    cur_sub_t += st
            if cur_sub:
                sub = ''.join(cur_sub).strip().replace('___ELLIPSIS___', '...')
                if sub:
                    final.append(sub)

        return [c for c in final if c.strip()]

    def _reset_chunk_count(self):
        self.current_chapter_id   = None
        self.chunk_count_in_chapter = 0

    def _identify_and_add_entities(self, text: str, source_lang: str, target_lang: str,
                                    is_japanese_webnovel: bool = False) -> None:
        if self.stop_event and self.stop_event.is_set():
            return
        if not isinstance(text, str) or len(text.strip()) < 50:
            return

        chapter_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if chapter_hash in getattr(self, "chapter_ner_cache", {}):
            entities = self.chapter_ner_cache[chapter_hash]
        else:
            ner_prompt      = create_ner_prompt(text, source_lang, target_lang, is_japanese_webnovel)
            est_tokens      = self.estimate_tokens(ner_prompt)
            scheduler       = getattr(self, "scheduler", None)
            if scheduler:
                model_info = scheduler.schedule(ner_prompt, est_tokens)
                if not model_info:
                    model_info = self._wait_for_available_model(est_tokens)
            else:
                model_info = self._wait_for_available_model(est_tokens)

            if not model_info:
                self.logger.warning("No model available for NER. Skipping.")
                return

            _, model, limits = model_info
            entities: List[dict] = []

            for attempt in range(self.RETRY_ATTEMPTS):
                if self.stop_event and self.stop_event.is_set():
                    return
                try:
                    current_time = time.time()
                    if current_time - limits.last_reset_time >= 60:
                        limits.requests_made = 0
                        limits.tokens_used   = 0
                        limits.last_reset_time = current_time
                    if (limits.requests_made >= limits.rpm
                            or limits.tokens_used + est_tokens > limits.tpm
                            or limits.daily_requests >= limits.rpd):
                        wait_result = self._wait_for_available_model(est_tokens)
                        if wait_result:
                            _, model, limits = wait_result
                        continue

                    response = self.client.models.generate_content(
                        model=model,
                        contents=ner_prompt,
                        config={
                            "temperature": 0.1,
                            "max_output_tokens": 8192,
                            "response_mime_type": "application/json",
                            "response_schema": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "entity":      {"type": "STRING"},
                                        "type":        {"type": "STRING"},
                                        "translation": {"type": "STRING"},
                                    },
                                    "required": ["entity", "type", "translation"],
                                },
                            },
                            "safety_settings": [
                                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                            ],
                        },
                    )

                    if not response or not response.text:
                        raise ProhibitedContentError("NER response blocked or empty.")

                    actual_out = self.estimate_tokens(response.text or "")
                    limits.tokens_used   += est_tokens + actual_out
                    limits.requests_made += 1
                    limits.daily_requests += 1
                    self._maybe_save_history()

                    resp_text = re.sub(r"```json|```", "", response.text).strip()
                    try:
                        entities = json.loads(resp_text) if resp_text else []
                        if not isinstance(entities, list):
                            entities = []
                    except json.JSONDecodeError:
                        entities = []

                    # Adaptive delay based on model RPM
                    delay = self._min_request_interval(limits)
                    time.sleep(delay)
                    break

                except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable,
                        errors.ClientError) as e:
                    retry_delay = self._extract_retry_delay(e, attempt)
                    self.logger.warning(f"Rate limit (NER): waiting {retry_delay:.1f}s")
                    self._interruptible_sleep(retry_delay + random.uniform(0, 1))
                except ProhibitedContentError:
                    self.logger.warning("NER blocked by safety filters.")
                    break
                except Exception as e:
                    self.logger.warning(f"NER attempt {attempt + 1} failed: {e}")
                    if attempt == self.RETRY_ATTEMPTS - 1:
                        break
                    self._interruptible_sleep((2 ** attempt) + random.uniform(0, 1))

            self.chapter_ner_cache[chapter_hash] = entities

        for entity_data in entities:
            if not isinstance(entity_data, dict):
                continue
            entity_text       = entity_data.get("entity", "").strip()
            translated_entity = entity_data.get("translation", "").strip(' \n\r\t\'""\u201c\u201d')
            entity_type       = entity_data.get("type", "")
            if not entity_text or not translated_entity:
                continue
            if entity_type not in ("PERSON", "ORGANIZATION"):
                continue
            if self.glossary_manager.get_target_term(entity_text):
                continue
            if translated_entity.lower() != entity_text.lower():
                added = self.glossary_manager.add_entry(entity_text, translated_entity, force_update=False)
                if added:
                    self.logger.info(f"NER glossary: {entity_text} → {translated_entity}")

    def _post_process_translation(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        # Fix stutter artefacts (e.g. "a - a")
        text = re.sub(r'\b([a-zA-Z])- ?\1\b', r'\1', text)
        # Collapse excess internal whitespace (not at newlines)
        text = re.sub(r'(?<!\n) {2,}(?!\n)', ' ', text)
        return text

    @staticmethod
    def _extract_retry_delay(exc: Exception, attempt: int) -> float:
        """Extract retry delay from API error, falling back to exponential backoff."""
        if hasattr(exc, 'retry_delay') and exc.retry_delay is not None:
            return exc.retry_delay.total_seconds()
        try:
            m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return float(2 ** attempt)

    def translate_text(self, text: str, source_lang: str, target_lang: str,
                       is_japanese_webnovel: bool = False,
                       chapter_id: str = None) -> str:
        if self.stop_event and self.stop_event.is_set():
            return text

        original_text = text
        try:
            if not isinstance(original_text, str) or not original_text.strip():
                return original_text

            cleaned_text, whitespace_info = extract_whitespace_info(original_text)
            if not cleaned_text or len(cleaned_text) < self.MIN_RESPONSE_LENGTH:
                return reconstruct_whitespace("", whitespace_info)

            # strip [Tn] tokens before NER so they don't confuse the model
            text_for_ner = re.sub(r'\[T\d+\]', '', cleaned_text).strip()
            if self.glossary_manager is not None:
                self._identify_and_add_entities(
                    text_for_ner, source_lang, target_lang, is_japanese_webnovel
                )

            if self.glossary_manager is None:
                processed_text = cleaned_text
                self.active_placeholders = {}
            else:
                processed_text, self.active_placeholders = \
                    self.glossary_manager.create_placeholders(cleaned_text)

            est_total = self.estimate_tokens(processed_text)
            model_info = self._wait_for_available_model(est_total)
            if not model_info:
                self.logger.error("No models available — skipping translation.")
                return reconstruct_whitespace(cleaned_text, whitespace_info)

            model_type, model, limits = model_info
            self.logger.info(f"Using {model_type.value} for translation.")

            chunks = self._split_large_chunk(processed_text, limits.max_input_tokens)
            translated_chunks: List[str] = []

            if chapter_id != self.current_chapter_id:
                self._reset_chunk_count()
                self.current_chapter_id = chapter_id

            for i, chunk in enumerate(chunks):
                if self.stop_event and self.stop_event.is_set():
                    translated_chunks.extend(chunks[i:])
                    break

                if (not isinstance(chunk, str) or not chunk.strip()
                        or len(chunk.strip()) < self.MIN_RESPONSE_LENGTH):
                    translated_chunks.append(chunk)
                    continue

                use_continuation = (i > 0 and not (
                    len(chunks) > 5 and i % 3 == 2
                ))
                translated_chunk = chunk  # fallback

                for attempt in range(self.RETRY_ATTEMPTS):
                    if self.stop_event and self.stop_event.is_set():
                        break
                    try:
                        current_time = time.time()
                        if current_time - limits.last_reset_time >= 60:
                            limits.requests_made = 0
                            limits.tokens_used   = 0
                            limits.last_reset_time = current_time

                        est_chunk = self.estimate_tokens(chunk)
                        if (limits.requests_made >= limits.rpm
                                or limits.tokens_used + est_chunk > limits.tpm
                                or limits.daily_requests >= limits.rpd):
                            wait_result = self._wait_for_available_model(est_chunk)
                            if wait_result:
                                model_type, model, limits = wait_result
                            continue

                        prompt      = create_translation_prompt(
                            chunk, source_lang, target_lang,
                            self.active_placeholders,
                            is_continuation=use_continuation,
                        )
                        est_prompt  = self.estimate_tokens(prompt)

                        if self.stop_event and self.stop_event.is_set():
                            return None

                        response = self.client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config={
                                "temperature": 0.3,
                                "max_output_tokens": 4096,
                                "safety_settings": [
                                    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                     "threshold": HarmBlockThreshold.BLOCK_NONE},
                                    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                     "threshold": HarmBlockThreshold.BLOCK_NONE},
                                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                                     "threshold": HarmBlockThreshold.BLOCK_NONE},
                                    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                     "threshold": HarmBlockThreshold.BLOCK_NONE},
                                ],
                            },
                        )

                        if (response.prompt_feedback
                                and response.prompt_feedback.block_reason):
                            break  # Use fallback (original chunk)

                        response_text   = response.text if response.text is not None else ""
                        current_output  = response_text.strip()

                        # Validate glossary placeholder preservation
                        placeholder_missing = any(
                            ph in chunk and ph not in current_output
                            for ph in self.active_placeholders
                        )
                        if placeholder_missing:
                            if attempt == self.RETRY_ATTEMPTS - 1:
                                break
                            continue

                        if not current_output or len(current_output) < self.MIN_RESPONSE_LENGTH:
                            if attempt < self.RETRY_ATTEMPTS - 1:
                                raise ValueError("Empty/short response.")
                            break

                        if self.validator.validate_chunk(chunk, current_output):
                            translated_chunk = current_output
                            actual_out       = self.estimate_tokens(response_text)
                            total_tokens     = est_prompt + actual_out
                            limits.tokens_used    += total_tokens
                            limits.requests_made  += 1
                            limits.daily_requests += 1
                            limits.last_requests_made_per_request = 1
                            limits.last_tokens_used_per_request   = total_tokens
                            self._maybe_save_history()
                            # Adaptive delay based on model RPM
                            time.sleep(self._min_request_interval(limits))
                            break

                    except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable,
                            errors.ClientError) as e:
                        self.logger.error(f"API error (translation): {e}")
                        retry_delay = self._extract_retry_delay(e, attempt)
                        self.logger.warning(f"Rate limit: waiting {retry_delay:.1f}s")
                        time.sleep(retry_delay + random.uniform(0, 1))
                        if attempt == self.RETRY_ATTEMPTS - 1:
                            break
                    except Exception as e:
                        self.logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                        self._interruptible_sleep((2 ** attempt) + random.uniform(0, 1))
                        if attempt == self.RETRY_ATTEMPTS - 1:
                            break

                translated_chunks.append(translated_chunk)
                self.chunk_count_in_chapter += 1

            # Restore glossary placeholders 
            combined = "\n\n".join(translated_chunks)
            if self.glossary_manager:
                self.glossary_manager._validate_glossary_restoration(
                    processed_text, combined, self.active_placeholders
                )
                final_restored = self.glossary_manager.restore_placeholders(
                    combined, self.active_placeholders
                )
            else:
                final_restored = combined
            self.active_placeholders = {}

            post_processed = self._post_process_translation(final_restored)
            final_text     = reconstruct_whitespace(post_processed, whitespace_info)
            self.translation_count += 1
            return final_text

        except Exception as e:
            self.logger.error(f"Unhandled error in translate_text: {e}", exc_info=True)
            self._maybe_save_history(force=True)
            self.active_placeholders = {}
            _, wsi = extract_whitespace_info(original_text)
            return reconstruct_whitespace(original_text.strip(), wsi)

    def get_model_status(self) -> Dict:
        status = {}
        current_time = time.time()
        for model_type, model_info in self.models.items():
            lim = model_info['limits']
            since_reset = current_time - lim.last_reset_time
            since_daily = current_time - lim.last_daily_reset
            remaining_rpm = max(0, lim.rpm - lim.requests_made) if since_reset < 60 else lim.rpm
            remaining_tpm = max(0, lim.tpm - lim.tokens_used)   if since_reset < 60 else lim.tpm
            remaining_rpd = max(0, lim.rpd - lim.daily_requests) if since_daily < 86400 else lim.rpd
            est_chaps_remaining = min(
                max(0, lim.rpd - lim.daily_requests),
                max(0, lim.max_chapters_per_day - lim.daily_requests),
            )
            status[model_type.value] = {
                'model_name':  model_info.get('model_name', 'N/A'),
                'priority':    model_info.get('priority', 'N/A'),
                'available':   (model_info.get('available', False)
                                and model_info.get('model_instance') is not None),
                'limits': {
                    'tpm': lim.tpm, 'rpd': lim.rpd, 'rpm': lim.rpm,
                    'max_input_tokens': lim.max_input_tokens,
                    'max_chapters_per_day': lim.max_chapters_per_day,
                },
                'current_usage': {
                    'requests_made_current_minute': lim.requests_made,
                    'tokens_used_current_minute':   lim.tokens_used,
                    'daily_requests':               lim.daily_requests,
                },
                'remaining': {
                    'remaining_rpm': remaining_rpm,
                    'remaining_tpm': remaining_tpm,
                    'remaining_rpd': remaining_rpd,
                    'estimated_chapters_remaining_today': est_chaps_remaining,
                },
                'last_reset_time':  time.strftime('%Y-%m-%d %H:%M:%S',
                                                  time.localtime(lim.last_reset_time)),
                'last_daily_reset': time.strftime('%Y-%m-%d %H:%M:%S',
                                                  time.localtime(lim.last_daily_reset)),
                'last_requests_made_per_request': lim.last_requests_made_per_request,
                'last_tokens_used_per_request':   lim.last_tokens_used_per_request,
            }
        return status

    def get_daily_chapter_capacity(self) -> int:
        total = 0
        for model_info in self.models.values():
            if model_info.get('available') and model_info.get('model_instance') is not None:
                total += model_info['limits'].max_chapters_per_day
        return total

    def optimize_for_chapters(self, estimated_chapters: int):
        self.model_priority_order = self._calculate_optimal_model_for_chapters(estimated_chapters)
        status = self.get_model_status()
        total_remaining = sum(
            s['remaining']['estimated_chapters_remaining_today']
            for s in status.values() if s['available']
        )
        self.logger.info(
            f"Optimised for {estimated_chapters} chapters. "
            f"Total remaining capacity today: {total_remaining}. "
            f"Priority order: {[m.value for m in self.model_priority_order]}"
        )
        if estimated_chapters > total_remaining > 0:
            earliest_reset = min(
                (info['limits'].last_daily_reset + 86400
                 for info in self.models.values() if info.get('model_instance')),
                default=time.time() + 86400,
            )
            wait_secs = max(0, earliest_reset - time.time())
            self.logger.warning(
                f"Estimated {estimated_chapters} chapters exceeds remaining capacity "
                f"({total_remaining}). Earliest reset in {wait_secs:.0f}s."
            )
        elif estimated_chapters > 0 and total_remaining == 0:
            self.logger.warning(
                "No remaining daily capacity. Wait for daily limits to reset."
            )

    def save_state(self):
        try:
            if self.glossary_manager:
                self.glossary_manager.save_glossary()
            self._save_history()
            self.logger.info("Translator state saved.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def reset(self):
        self.translation_count = 0
        current_time = time.time()
        for model_info in self.models.values():
            lim = model_info['limits']
            lim.requests_made  = 0
            lim.tokens_used    = 0
            lim.daily_requests = 0
            lim.last_reset_time  = current_time
            lim.last_daily_reset = current_time
        self.logger.info("Translator session and usage history reset.")
        self._save_history()

    def get_translation_count(self) -> int:
        return self.translation_count

    def get_remaining_translations(self) -> int:
        return max(0, self.max_translations_per_session - self.translation_count)

    def stop(self):
        """Signal the translator to stop (for GUI stop button)."""
        if self.stop_event:
            self.stop_event.set()