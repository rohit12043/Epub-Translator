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
from prompts import create_ner_prompt, create_equivalent_translation_prompt, create_translation_prompt
from exceptions import ProhibitedContentError
from glossary import GlossaryManager
from rate_limiter import ModelLimits, ModelType
from text_utils import extract_whitespace_info, reconstruct_whitespace
from validator import TranslationValidator
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
from google.genai import errors

class GeminiTranslator:
    HISTORY_FILENAME = "translation_history.json"
    LOCK_FILENAME = "translation_history.json.lock"
    GLOSSARY_FILENAME = "glossary.json"

    def __init__(self, api_key: str, max_translations_per_session: int = 2000, checkpoint_dir: str = "checkpoints", stop_event=None):
        self.logger = logging.getLogger(__name__)
        if not api_key:
            raise ValueError("API key must be provided.")
        try:
            self.client = genai.Client(api_key=api_key)
            self.logger.info("Google Generative AI Client configured successfully.")
        except Exception as e:
            self.logger.error(f"Failed to configure Google Generative AI Client with provided API key: {e}")
            raise

        self.max_translations_per_session = max_translations_per_session
        self.translation_count = 0
        self.stop_event = stop_event
        self.RETRY_ATTEMPTS = 5
        self.MIN_RESPONSE_LENGTH = 5
        self.BURST_DELAY = 4.0
        self.current_chapter_id = None
        self.chunk_count_in_chapter = 0
        self.checkpoint_dir = Path(checkpoint_dir)
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.HISTORY_FILE = str(self.checkpoint_dir / self.HISTORY_FILENAME)
            self.LOCK_FILE = str(self.checkpoint_dir / self.LOCK_FILENAME)
            self.GLOSSARY_FILE = str(self.checkpoint_dir / self.GLOSSARY_FILENAME)
            self.logger.info(f"Checkpoint directory set to: {self.checkpoint_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint directory {checkpoint_dir}: {e}. Using current directory for files.")
            self.HISTORY_FILE = self.HISTORY_FILENAME
            self.LOCK_FILE = self.LOCK_FILENAME
            self.GLOSSARY_FILE = self.GLOSSARY_FILENAME

        self.glossary_manager = GlossaryManager(self.GLOSSARY_FILE)
        self.validator = TranslationValidator()
        self.models = {
                    ModelType.GEMINI_2_5_FLASH: {
                        'model_name': 'gemini-2.5-flash',
                        'model_instance': None,
                        'limits': ModelLimits(tpm=1000000, tpd=1000000, rpd=1500, rpm=15, max_chapters_per_day=312, tokens_per_chapter=3200, max_input_tokens=1000000),
                        'priority': 1,
                        'available': True
                    },
                    ModelType.GEMINI_2_5_PRO: {
                        'model_name': 'gemini-2.5-pro',
                        'model_instance': None,
                        'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=25, rpm=5, max_chapters_per_day=78, tokens_per_chapter=3200, max_input_tokens=2000000),
                        'priority': 2,
                        'available': True
                    },
                    ModelType.GEMINI_3_FLASH: {
                        'model_name': 'gemini-3-flash',
                        'model_instance': None,
                        'limits': ModelLimits(tpm=1000000, tpd=1000000, rpd=1500, rpm=15, max_chapters_per_day=312, tokens_per_chapter=3200, max_input_tokens=1000000),
                        'priority': 3,
                        'available': True
                    },
                    ModelType.GEMINI_3_1_PRO: {
                        'model_name': 'gemini-3.1-pro',
                        'model_instance': None,
                        'limits': ModelLimits(tpm=250000, tpd=1000000, rpd=25, rpm=5, max_chapters_per_day=78, tokens_per_chapter=3200, max_input_tokens=2000000),
                        'priority': 4,
                        'available': True
                    }
                }
        
        self._initialize_models()
        self._load_history()
        self.model_priority_order = sorted([mt for mt, info in self.models.items() if info.get('model_instance') is not None],
                                        key=lambda x: self.models[x]['priority'])
        if not self.model_priority_order:
            self.logger.error("No model instances were successfully initialized. Translation will not be possible.")
        self.active_placeholders: Dict[str, Tuple[str, str]] = {}

    def _interruptible_sleep(self, seconds: float):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event and self.stop_event.is_set():
                return
            time.sleep(0.2)
            
    def _initialize_models(self):
        """Initializes model configurations for configured models."""
        for model_type, model_info in self.models.items():
            model_name = model_info.get('model_name')
            if not model_name:
                self.logger.error(f"Model configuration for {model_type.value} is missing 'model_name'. Skipping initialization.")
                model_info['available'] = False
                continue
            try:
                model_info['model_instance'] = model_name
                self.logger.info(f"Successfully configured model reference for {model_name}")
                model_info['available'] = True
            except Exception as e:
                self.logger.error(f"Failed to configure model reference for {model_name}: {e}. Marking as unavailable.")
                model_info['available'] = False

    def _load_history(self):
        """Loads translation history including model usage statistics."""
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
                                # Load limits from history, potentially overriding defaults
                                self.models[model_type]['limits'] = ModelLimits.from_dict(model_data['limits'])
                                limits = self.models[model_type]['limits']
                                self.logger.info(
                                    f"Loaded history for {model_type.value}: "
                                    f"requests_made={limits.requests_made}, "
                                    f"tokens_used={limits.tokens_used}, "
                                    f"daily_requests={limits.daily_requests}, "
                                    f"last_reset_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(limits.last_reset_time))}, "
                                    f"last_daily_reset={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(limits.last_daily_reset))}"
                                )
                            else:
                                self.logger.warning(f"History for {model_type.value} is missing 'limits' data. Using default limits for this model.")

                            # Also load availability and priority from history if present
                            if 'available' in model_data:
                                self.models[model_type]['available'] = model_data['available']
                            if 'priority' in model_data:
                                self.models[model_type]['priority'] = model_data['priority']

                        else:
                            self.logger.warning(f"Unknown model type '{model_type_str}' found in history. Skipping.")
                    except ValueError:
                        self.logger.warning(f"Invalid model type string '{model_type_str}' in history. Skipping.")
                    except Exception as e:
                         self.logger.error(f"Error processing history data for model {model_type_str}: {e}. Skipping this model's history.")


                self.logger.info("History loading complete.")
            else:
                self.logger.info(f"No history file found at {self.HISTORY_FILE}, starting with fresh metrics.")
                self._save_history() # Save initial state with defaults

        except FileLock.Timeout:
            self.logger.error(f"Timeout acquiring lock for history file {self.LOCK_FILE}. History may not be loaded correctly. Proceeding with potentially stale or default data.")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode history JSON from {self.HISTORY_FILE}: {e}. Initializing with defaults.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading history from {self.HISTORY_FILE}: {e}. Initializing with defaults.")

    def _save_history(self):
        """Saves translation history including detailed model states."""
        try:
            history = {}
            for model_type, model_info in self.models.items():
                history[model_type.value] = {
                    'limits': model_info['limits'].to_dict(),
                    'available': model_info.get('available', True),
                    'priority': model_info.get('priority', model_type.value) # Save priority
                }

            history_dir = os.path.dirname(self.HISTORY_FILE)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)

            temp_file = self.HISTORY_FILE + ".tmp"
            with FileLock(self.LOCK_FILE, timeout=15):
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, self.HISTORY_FILE)

            self.logger.debug(f"Saved detailed translation history to {self.HISTORY_FILE}.")
        except FileLock.Timeout:
            self.logger.error(f"Timeout acquiring lock for history file {self.LOCK_FILE}. History may not be saved.")
        except Exception as e:
            self.logger.error(f"Failed to save history to {self.HISTORY_FILE}: {e}.")

    def _calculate_optimal_model_for_chapters(self, estimated_chapters: int) -> List[ModelType]:
        """
        Calculates optimal model order based on remaining daily capacity (RPD and max_chapters_per_day limits)
        and prioritizing models with higher TPM. Resets daily counters if a day has passed.
        """
        efficiency_scores = []
        current_time = time.time()

        # Perform daily reset for all models before calculating capacity
        for model_info in self.models.values():
            limits = model_info['limits']
            if current_time - limits.last_daily_reset >= 86400:
                self.logger.info(f"Resetting daily limits for {model_info.get('model_name', 'N/A')}")
                limits.daily_requests = 0
                limits.last_daily_reset = current_time
        self._save_history() # Save state after potential daily resets

        for model_type, model_info in self.models.items():
            if not model_info.get('available', True) or model_info.get('model_instance') is None:
                self.logger.debug(f"Model {model_type.value} is unavailable or not initialized.")
                continue

            limits = model_info['limits']

            # Estimate remaining chapters based on RPD and configured max_chapters_per_day
            remaining_rpd = max(0, limits.rpd - limits.daily_requests)
            remaining_chapters_by_rpd = max(0, limits.max_chapters_per_day - limits.daily_requests) # Assuming 1 request per chapter
            remaining_daily_capacity = min(remaining_rpd, remaining_chapters_by_chapers_per_day)

            if remaining_daily_capacity <= 0:
                self.logger.debug(f"Model {model_type.value} has no remaining daily capacity ({remaining_daily_capacity} est. chapters).")
                continue

            # Calculate an efficiency score: Remaining Capacity * TPM / Priority (lower priority is better)
            efficiency = (remaining_daily_capacity * limits.tpm) / (model_info.get('priority', 1) + 0.001) if model_info.get('priority', 1) > 0 else 0
            efficiency_scores.append((model_type, efficiency))


        # Sort models by efficiency score (higher is better)
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(f"Model efficiency ranking for {estimated_chapters} chapters:")
        if not efficiency_scores:
            self.logger.warning("No models available or with remaining capacity for chapter translation.")
            self.model_priority_order = [] # Clear priority order if no models are available
            return []

        ranked_models = [model_type for model_type, _ in efficiency_scores]

        # Log detailed efficiency information
        for model_type, efficiency in efficiency_scores:
             limits = self.models[model_type]['limits']
             remaining_rpd = max(0, limits.rpd - limits.daily_requests)
             remaining_chapters_by_chapers_per_day = max(0, limits.max_chapters_per_day - limits.daily_requests)
             remaining_daily_capacity = min(remaining_rpd, remaining_chapters_by_chapers_per_day)
             self.logger.info(
                 f" - {model_type.value}: Efficiency: {efficiency:.2f}, "
                 f"Rem Daily Req: {remaining_rpd}, Rem Config Chapters: {remaining_chapters_by_chapers_per_day}, "
                 f"Est Rem Daily: {remaining_daily_capacity}"
             )

        self.model_priority_order = ranked_models # Update the main priority order
        return self.model_priority_order

    def _get_best_available_model(self, estimated_tokens: int) -> Optional[Tuple[ModelType, str, ModelLimits]]:
        """
        Selects the best available model based on current rate limits and priority.
        Checks minute and daily limits and resets them if their windows have passed.
        Returns (model_type, model_instance, limits) or None if no model is available.
        """
        current_time = time.time()

        for model_type in self.model_priority_order:
            model_info = self.models.get(model_type)
            if not model_info:
                self.logger.debug(f"Model type {model_type.value} in priority order but not in models dictionary.")
                continue

            limits = model_info['limits']
            model_instance = model_info.get('model_instance')

            # Reset minute limits if the window has passed
            if current_time - limits.last_reset_time >= 60:
                self.logger.debug(f"Resetting minute limits for {model_type.value}")
                limits.requests_made = 0
                limits.tokens_used = 0
                limits.last_reset_time = current_time

            # Reset daily limits if the window has passed
            if current_time - limits.last_daily_reset >= 86400:
                self.logger.info(f"Resetting daily limits for {model_type.value} (fallback reset).")
                limits.daily_requests = 0
                limits.last_daily_reset = current_time

            # Check if the model is available and has capacity
            if (model_info.get('available', True) and
                model_instance is not None and
                limits.requests_made < limits.rpm and
                limits.tokens_used + estimated_tokens <= limits.tpm and
                limits.daily_requests < limits.rpd and
                estimated_tokens <= limits.max_input_tokens):

                self.logger.debug(
                    f"Selected {model_type.value}. "
                    f"Est Tokens: {estimated_tokens}. "
                    f"Usage: {limits.requests_made}/{limits.rpm} RPM, "
                    f"{limits.tokens_used}/{limits.tpm} TPM, "
                    f"{limits.daily_requests}/{limits.rpd} RPD."
                )
                self._save_history() # Save state after selecting a model
                return model_type, model_instance, limits

            # Log why a model wasn't selected (for debugging)
            status_parts = []
            if not model_info.get('available', True):
                 status_parts.append("not available flag")
            if model_instance is None:
                 status_parts.append("not initialized")
            if limits.requests_made >= limits.rpm:
                status_parts.append(f"RPM limit ({limits.rpm}) reached")
            if limits.tokens_used + estimated_tokens > limits.tpm:
                status_parts.append(f"TPM limit ({limits.tpm}) will be exceeded by {estimated_tokens}")
            if limits.daily_requests >= limits.rpd:
                status_parts.append(f"RPD limit ({limits.rpd}) reached")
            if estimated_tokens > limits.max_input_tokens:
                status_parts.append(f"chunk ({estimated_tokens}) exceeds max input ({limits.max_input_tokens})")

            if status_parts:
                 self.logger.debug(f"Model {model_type.value} not selected because: {', '.join(status_parts)}")


        self.logger.debug("No available model with sufficient capacity found currently.")
        return None

    def _wait_for_available_model(self, required_tokens: int) -> Optional[Tuple[ModelType, str, ModelLimits]]:
        """
        Enhanced model waiting with:
        - API retry-after header support
        - Precise rate limit window tracking
        - Progressive backoff with jitter
        - Comprehensive status reporting
        """
        max_wait_time = 600 # Maximum total wait time in seconds
        start_time = time.time()
        wait_sequence = [5, 10, 20, 40, 60, 90, 120] # Base wait times
        current_wait_index = 0
        specific_retry_after = 0 # Time until a specific model suggests retrying

        while time.time() - start_time < max_wait_time:
            if self.stop_event and self.stop_event.is_set():
                self.logger.info("Stopped while waiting for model.")
                return None
            current_time = time.time()

            # Compare with API-provided retry-after if set
            if specific_retry_after > current_time:
                sleep_time = max(1, specific_retry_after - current_time + random.uniform(0, 2)) # Add small jitter
                self.logger.info(f"Honoring API retry-after: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                specific_retry_after = 0 # Reset after waiting
                continue # Check models again immediately

            best_model = None
            # Check models in priority order
            for model_type in self.model_priority_order:
                model_info = self.models.get(model_type)
                if not model_info:
                    continue # Skip if model type is not in config

                limits = model_info['limits']
                model_instance = model_info.get('model_instance')

                # Reset minute limits if the window has passed
                if current_time - limits.last_reset_time >= 60:
                    limits.requests_made = 0
                    limits.tokens_used = 0
                    limits.last_reset_time = current_time

                # Reset daily limits if the window has passed
                if current_time - limits.last_daily_reset >= 86400:
                    limits.daily_requests = 0
                    limits.last_daily_reset = current_time

                # Check if the model is available and has capacity for the required tokens
                if (model_info.get('available', True) and
                    model_instance is not None and
                    limits.requests_made < limits.rpm and
                    limits.tokens_used + required_tokens <= limits.tpm and
                    limits.daily_requests < limits.rpd and
                    required_tokens <= limits.max_input_tokens):
                    best_model = (model_type, model_instance, limits)
                    break # Found a suitable model, exit loop

            if best_model:
                model_type, _, limits = best_model
                self.logger.info(
                    f"Selected {model_type.value} with capacity: "
                    f"{limits.rpm-limits.requests_made}RPM, "
                    f"{limits.tpm-limits.tokens_used}TPM, "
                    f"{limits.rpd-limits.daily_requests}RPD remaining"
                )
                return best_model

            # No model available, calculate wait time until the earliest minute window reset
            base_wait = wait_sequence[min(current_wait_index, len(wait_sequence)-1)]
            time_to_reset = 60 # Assume 60s window
            for model_info in self.models.values():
                 if model_info.get('model_instance'): # Only consider models that were initialized
                     # Time until the minute counter resets for this model
                     remaining = 60 - (current_time - model_info['limits'].last_reset_time) % 60
                     time_to_reset = min(time_to_reset, remaining)

            # Wait time is the minimum of the base backoff and the time until the earliest minute reset, with jitter
            sleep_time = min(base_wait, time_to_reset) * (0.9 + 0.2 * random.random()) # Add 10-20% jitter
            self.logger.warning(
                f"No available models (attempt {current_wait_index+1}), "
                f"waiting {sleep_time:.1f}s. "
                f"Next window reset in {time_to_reset:.1f}s"
            )
            time.sleep(sleep_time)
            current_wait_index += 1

            # Save history periodically while waiting to persist updated limits
            if current_wait_index % 3 == 0: # Save history every few attempts
                 self._save_history()


        self.logger.error(
            f"Failed to acquire model after {max_wait_time}s. "
            f"Last required tokens: {required_tokens}"
        )
        return None

    def estimate_tokens(self, text: str) -> int:
        """
        Estimates tokens using a heuristic (characters / X + buffer).
        This is a rough estimate and not as accurate as model-specific tokenizers, but useful for chunking
        and initial limit checks. Character/3 is common for Latin text. For mixed scripts (like CJK + Latin),
        a higher divisor might be more accurate, but 3 is a generally safe lower bound estimate ensuring we
        don't underestimate too much. Adding a buffer accounts for prompt structure, few-shot examples, etc.
        """
        if not isinstance(text, str):
            text = ""
        if not text:
            return 0
        # Heuristic: Characters / 3 + a buffer for prompt overhead, etc.
        return max(1, len(text) // 3 + 150) # Ensure minimum of 1 token for non-empty strings

    def count_tokens(self, text: str, model: str = None) -> int:
        if model:
            try:
                response = self.client.models.count_tokens(
                    model=model,
                    contents=text
                )
                return response.total_tokens
            except Exception as e:
                self.logger.warning(f"Token count failed: {e}")
                return self.estimate_tokens(text)
        return self.estimate_tokens(text)

    def _split_large_chunk(self, text: str, max_tokens: int) -> List[str]:
        """Splits text into chunks within token limits, prioritizing paragraph breaks then sentences."""
        if not isinstance(text, str) or not text.strip():
            return []
        estimated_tokens = self.estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return [text]
        self.logger.info(f"Splitting large chunk: estimated {estimated_tokens} tokens > max {max_tokens} tokens.")
        chunks = []
        current_chunk_parts = []
        current_chunk_tokens = 0
        safety_buffer = 50

        # Protect ellipses
        text = re.sub(r'\.{3,}', '___ELLIPSIS___', text)
        text = re.sub(r'…', '___ELLIPSIS___', text)

        # Split by paragraphs first
        paragraphs = re.split(r'(\n\s*\n+)', text)
        paragraph_texts = []
        for i in range(0, len(paragraphs), 2):
            para = paragraphs[i]
            separator = paragraphs[i + 1] if i + 1 < len(paragraphs) else ''
            paragraph_texts.append((para, separator))

        for para, separator in paragraph_texts:
            para_tokens = self.estimate_tokens(para)
            separator_tokens = self.estimate_tokens(separator)
            if current_chunk_tokens > 0 and (current_chunk_tokens + para_tokens + separator_tokens + safety_buffer > max_tokens):
                chunk_text = ''.join(current_chunk_parts).strip()
                if chunk_text:
                    chunk_text = chunk_text.replace('___ELLIPSIS___', '...')
                    chunks.append(chunk_text)
                current_chunk_parts = [para + separator]
                current_chunk_tokens = para_tokens + separator_tokens
            else:
                current_chunk_parts.append(para + separator)
                current_chunk_tokens += para_tokens + separator_tokens

        if current_chunk_parts:
            chunk_text = ''.join(current_chunk_parts).strip()
            if chunk_text:
                chunk_text = chunk_text.replace('___ELLIPSIS___', '...')
                chunks.append(chunk_text)

        # Further split chunks by sentences if still too large
        final_chunks = []
        for chunk in chunks:
            chunk_est_tokens = self.estimate_tokens(chunk)
            if chunk_est_tokens <= max_tokens:
                final_chunks.append(chunk)
                continue
            self.logger.info(f"Chunk (estimated {chunk_est_tokens} tokens) too large, splitting by sentences.")
            # Re-protect ellipses
            chunk = re.sub(r'\.{3,}', '___ELLIPSIS___', chunk)
            chunk = re.sub(r'…', '___ELLIPSIS___', chunk)
            # Split by sentence-ending punctuation
            sentence_parts = re.split(r'([.!?。？！]+)', chunk)
            sentences = []
            for i in range(0, len(sentence_parts) - 1, 2):
                sentence = sentence_parts[i] + (sentence_parts[i + 1] if i + 1 < len(sentence_parts) else "")
                if sentence.strip():
                    sentences.append(sentence)
            if len(sentence_parts) % 2 == 1 and sentence_parts[-1].strip():
                sentences.append(sentence_parts[-1])

            current_sub_chunk = []
            sub_chunk_tokens = 0
            for sentence in sentences:
                sentence_tokens = self.estimate_tokens(sentence)
                if sub_chunk_tokens > 0 and (sub_chunk_tokens + sentence_tokens + safety_buffer > max_tokens):
                    sub_chunk = ''.join(current_sub_chunk).strip()
                    if sub_chunk:
                        sub_chunk = sub_chunk.replace('___ELLIPSIS___', '...')
                        final_chunks.append(sub_chunk)
                    current_sub_chunk = [sentence]
                    sub_chunk_tokens = sentence_tokens
                else:
                    current_sub_chunk.append(sentence)
                    sub_chunk_tokens += sentence_tokens

            if current_sub_chunk:
                sub_chunk = ''.join(current_sub_chunk).strip()
                if sub_chunk:
                    sub_chunk = sub_chunk.replace('___ELLIPSIS___', '...')
                    final_chunks.append(sub_chunk)

        # Clean final chunks
        cleaned_final_chunks = [c.strip() for c in final_chunks if c and c.strip()]
        self.logger.debug(f"Split into {len(cleaned_final_chunks)} chunks.")
        return cleaned_final_chunks


    def _reset_chunk_count(self):
        """Resets chunk count and chapter ID for a new chapter."""
        self.current_chapter_id = None
        self.chunk_count_in_chapter = 0

    def _identify_and_add_entities(self, text: str, source_lang: str, target_lang: str, is_japanese_webnovel: bool = False) -> None:
        """
        Identifies named entities (PERSON, ORGANIZATION) in text and adds their translations/transliterations
        to the glossary if different from source. Uses an available model, potentially waiting for capacity.
        """
        if self.stop_event and self.stop_event.is_set():
            self.logger.debug("Stop signal received, skipping entity identification.")
            return

        # Skip NER for very short texts or invalid types
        if not isinstance(text, str) or len(text.strip()) < 20:
            self.logger.debug("Text too short or invalid type for NER, skipping entity identification.")
            return

        # Estimate tokens for the NER prompt + the text
        ner_prompt_base = create_ner_prompt("", "") # Estimate prompt overhead
        estimated_ner_prompt_tokens = self.estimate_tokens(ner_prompt_base) + self.estimate_tokens(text)

        # Acquire a model for the NER request
        model_info = self._wait_for_available_model(estimated_ner_prompt_tokens)
        if not model_info:
            self.logger.warning("No model available (even after waiting) for NER request. Skipping entity identification for this chunk.")
            return

        model_type, model, limits = model_info
        self.logger.debug(f"Using {model_type.value} for NER.")

        ner_prompt = create_ner_prompt(text, source_lang)
        ner_response = None

        # Retry loop for the NER API call
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                # Check rate limits again before the API call
                current_time = time.time()
                if current_time - limits.last_reset_time >= 60:
                     self.logger.debug(f"Resetting minute limits for {model_type.value} before NER call.")
                     limits.requests_made = 0
                     limits.tokens_used = 0
                     limits.last_reset_time = current_time

                # Re-evaluate token estimate for the actual prompt if needed
                estimated_prompt_tokens_actual = self.estimate_tokens(ner_prompt) # Use estimate

                if limits.requests_made >= limits.rpm or limits.tokens_used + estimated_prompt_tokens_actual > limits.tpm or limits.daily_requests >= limits.rpd:
                    self.logger.warning(f"Model {model_type.value} hit rate limit just before NER attempt {attempt+1}. Waiting...")
                    wait_result = self._wait_for_available_model(estimated_prompt_tokens_actual)
                    if wait_result:
                        model_type, model, limits = wait_result # Use the newly acquired model and limits
                        self.logger.info(f"Resuming NER attempt {attempt+1} with model {model_type.value} after waiting.")
                        continue # Continue with the API call after waiting
                    else:
                        self.logger.error(f"No model available after waiting for NER attempt {attempt+1}. Skipping NER.")
                        return # Cannot get a model, give up on NER for this chunk

                # Make the API call
                if self.stop_event and self.stop_event.is_set():
                    return None
                response = self.client.models.generate_content(
                        model=model,
                        contents=ner_prompt,
                        config={
                            "temperature": 0.1,
                            "maxOutputTokens": 8192,
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                        "type": "ARRAY",
                                        "items": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "entity": {"type": "STRING"},
                                                "type": {"type": "STRING"}
                                            },
                                            "required": ["entity", "type"]
                                        }
                                    },
                            "safety_settings": [
                                        {
                                            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                            "threshold": HarmBlockThreshold.BLOCK_NONE,
                                        },
                                        {
                                            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                            "threshold": HarmBlockThreshold.BLOCK_NONE,
                                        },
                                        {
                                            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                                            "threshold": HarmBlockThreshold.BLOCK_NONE,
                                        },
                                        {
                                            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                            "threshold": HarmBlockThreshold.BLOCK_NONE,
                                        },
                                    ],
                        }
                    )

                # Check for block reasons
                if not response or not response.text:
                        raise ProhibitedContentError("NER response blocked or empty.")

                ner_response = response # Store the successful response
                actual_output_tokens = self.estimate_tokens(ner_response.text if ner_response.text else "") # Estimate output tokens
                limits.tokens_used += estimated_prompt_tokens_actual + actual_output_tokens # Update token usage
                limits.requests_made += 1 # Update request count
                limits.daily_requests += 1 # Update daily request count
                self._save_history() # Save history after successful call
                time.sleep(self.BURST_DELAY)
                break # Exit retry loop on success

            except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, errors.ClientError) as e:
                self.logger.warning(f"API rate limit or service error during NER attempt {attempt+1}: {e}")
                retry_delay = 0
                if hasattr(e, 'retry_delay') and e.retry_delay is not None:
                    retry_delay = e.retry_delay.total_seconds()
                    self.logger.info(f"Waiting for API retry_delay: {retry_delay:.2f} seconds.")
                    time.sleep(retry_delay + random.uniform(0, 1)) # Add jitter to retry_delay
                else:
                    # Implement exponential backoff with jitter
                    sleep_duration = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"No specific retry_delay. Waiting with backoff: {sleep_duration:.2f} seconds.")
                    self._interruptible_sleep(sleep_duration)

            except ProhibitedContentError:
                 ner_response = None # Mark response as blocked, do not retry
                 break # Exit retry loop

            except Exception as e:
                self.logger.warning(f"An error occurred during NER attempt {attempt+1}: {e}", exc_info=True)
                if attempt == self.RETRY_ATTEMPTS - 1:
                    self.logger.error(f"NER failed after all retries due to unexpected error. Skipping NER.")
                    ner_response = None # Mark response as failed
                else:
                    # Implement exponential backoff with jitter for other errors
                    sleep_duration = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Waiting with backoff: {sleep_duration:.2f} seconds.")
                    self._interruptible_sleep(sleep_duration)

        # Process NER response if successful
        if ner_response and ner_response.text and ner_response.text.strip():
            response_text = re.sub(r"```json|```", "", ner_response.text).strip()

            entities = []
            try:
                if response_text: # Ensure response text is not empty after cleaning
                    entities = json.loads(response_text)
                    if not isinstance(entities, list):
                         self.logger.warning(f"NER response was not a JSON list. Response: {response_text[:200]}...")
                         entities = [] # Reset to empty list if not a list
                else:
                    self.logger.debug("NER response text was empty after stripping/cleaning.")

            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse NER response as JSON: {e}. Response: {response_text[:200]}...")
                entities = [] # Reset to empty list on JSON error
            except Exception as e:
                 self.logger.warning(f"An unexpected error occurred processing NER JSON: {e}. Response: {response_text[:200]}...")
                 entities = [] # Reset to empty list on other errors


            valid_entities = []
            for entity_data in entities:
                if isinstance(entity_data, dict) and isinstance(entity_data.get('entity'), str) and entity_data.get('type') in ['PERSON', 'ORGANIZATION']:
                    valid_entities.append(entity_data)
                else:
                    self.logger.warning(f"Skipping invalid entity format or type returned by NER: {entity_data}")


            if not valid_entities:
                 self.logger.debug("No valid entities found or identified by NER after processing.")
                 return # No valid entities to process

            self.logger.debug(f"Identified {len(valid_entities)} potential entities for translation.")

            # Translate identified entities and add to glossary
            for entity in valid_entities:
                entity_text = entity['entity']
                entity_type = entity['type']

                # Check if already in glossary
                if self.glossary_manager.get_target_term(entity_text) is not None:
                    self.logger.debug(f"Entity '{entity_text}' already in glossary. Skipping translation.")
                    continue

                translated_entity = self._find_equivalent_translation(
                    entity_text, entity_type, source_lang, target_lang, model, limits, is_japanese_webnovel=is_japanese_webnovel
                )

                if (translated_entity and translated_entity.strip() and 
                    translated_entity.strip().lower() != entity_text.strip().lower() and 
                    len(translated_entity) < 50 and not translated_entity.startswith("This")):
                    added = self.glossary_manager.add_entry(entity_text, translated_entity, force_update=False)
                    if added:
                        self.logger.info(f"Added NER entity to glossary: '{entity_text}' -> '{translated_entity}' (Type: {entity_type})")
                elif translated_entity and translated_entity.strip().lower() == entity_text.strip().lower():
                     self.logger.debug(f"Entity translation was same as source '{entity_text}' (or differed only by case/whitespace), not adding to glossary.")
                else:
                    self.logger.warning(f"Entity translation failed or returned empty/whitespace for '{entity_text}'. Not adding to glossary.")


        elif ner_response is None:
             self.logger.error("NER call failed after all retries or was blocked.")
        else:
            self.logger.debug("NER response was empty or contained only whitespace.")

    def _find_equivalent_translation(self, entity: str, entity_type: str, source_lang: str, target_lang: str, model: str, limits: ModelLimits, is_japanese_webnovel: bool = False) -> Optional[str]:
        """
        Finds translation/transliteration for a specific named entity using the provided model.
        Updates model usage stats for the passed limits object.
        This function assumes a model has already been acquired and passed in.
        """
        if self.stop_event and self.stop_event.is_set():
            self.logger.debug(f"Stop signal received, skipping entity translation for '{entity}'.")
            return entity # Return original entity on stop

        try:
            prompt = create_equivalent_translation_prompt(entity, entity_type, source_lang, target_lang, is_japanese_webnovel)
            estimated_prompt_tokens = self.estimate_tokens(prompt)

            if estimated_prompt_tokens > limits.max_input_tokens:
                 self.logger.warning(f"Entity translation prompt for '{entity}' too long ({estimated_prompt_tokens} tokens) for model max input ({limits.max_input_tokens}). Skipping translation.")
                 return entity # Return original if prompt is too large

            # Check rate limits before the API call
            current_time = time.time()
            if current_time - limits.last_reset_time >= 60:
                 self.logger.debug(f"Resetting minute limits for {model} before entity translation call.")
                 limits.requests_made = 0
                 limits.tokens_used = 0
                 limits.last_reset_time = current_time


            if limits.requests_made >= limits.rpm or limits.tokens_used + estimated_prompt_tokens > limits.tpm or limits.daily_requests >= limits.rpd:
                self.logger.warning(f"Model {model.model_name} hit rate limit just before translating entity '{entity}'. Skipping translation.")
                self._save_history() # Save state before returning due to limit
                return entity # Return original entity if rate limit is hit


            if self.stop_event and self.stop_event.is_set():
                return None
            response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "maxOutputTokens": 100,
                        "topP": 0.95,
                        "safety_settings": [
                                    {
                                        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                                    },
                                    {
                                        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                                    },
                                    {
                                        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                                    },
                                    {
                                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                                    },
                                ],
                    }
                )

            # Update block check:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                self.logger.warning(f"Entity translation prompt for '{entity}' blocked: {response.prompt_feedback.block_reason}. Returning original.")
                return entity

            translated_entity = response.text.strip() if response.text else ""
            actual_output_tokens = self.estimate_tokens(response.text if response.text else "") # Estimate output tokens

            # Update model usage stats for this entity translation request
            limits.tokens_used += estimated_prompt_tokens + actual_output_tokens
            limits.requests_made += 1
            limits.daily_requests += 1
            self._save_history() # Save history after successful entity translation
            time.sleep(self.BURST_DELAY)

            if not translated_entity:
                self.logger.warning(f"Empty translation response for entity: '{entity}'. Returning original.")
                return entity # Return original if translation is empty

            # Basic check for nonsensical output (e.g., just punctuation)
            if len(translated_entity.strip()) < 2 and re.fullmatch(r'[\W_]+', translated_entity.strip()):
                self.logger.warning(f"Translation for '{entity}' appears to be only punctuation/symbols: '{translated_entity}'. Returning original.")
                return entity

            self.logger.debug(f"Translated entity '{entity}' -> '{translated_entity}'")
            return translated_entity

        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable) as e:
            self.logger.warning(f"API rate limit or service error translating entity '{entity}': {e}. Returning original entity.", exc_info=True)
            self._save_history() # Save state on API error
            return entity # Return original entity on API error

        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, errors.ClientError) as e:
            self.logger.warning(f"API rate limit or service error translating entity '{entity}': {e}. Returning original entity.", exc_info=True)
            self._save_history() 
            return entity

    def _post_process_translation(self, translated_text: str) -> str:
        """Refines translation for common fluency issues."""
        processed_text = translated_text

        if not isinstance(processed_text, str):
             self.logger.warning(f"Post-processing received non-string input: {type(processed_text)}. Skipping post-processing.")
             return translated_text

        # Fix common model artifacts like "a -" or "a - a"
        # This pattern targets single letters followed by optional space and hyphen, then optional space and the same letter.
        # It might need refinement based on observed model behavior.
        processed_text = re.sub(r'\b([a-zA-Z])- ?\1\b', r'\1', processed_text)
        processed_text = re.sub(r'\b([a-zA-Z]) + \1\b', r'\1', processed_text)

        # Reduce multiple spaces that are not explicitly intended (e.g., not \n\n)
        processed_text = re.sub(r'(?<!\s)\s{2,}(?!\s)', ' ', processed_text)


        return processed_text

    def translate_text(self, text: str, source_lang: str, target_lang: str, is_japanese_webnovel: bool = False, chapter_id: str = None) -> str:
            if self.stop_event and self.stop_event.is_set():
                self.logger.info("Translator stopped signal received. Skipping translation.")
                return text

            original_text = text
            try:
                if not isinstance(original_text, str) or not original_text.strip():
                    self.logger.debug("Input text is empty or contains only whitespace or is not a string. Skipping translation.")
                    return original_text

                cleaned_text, whitespace_info = extract_whitespace_info(original_text)
                if not cleaned_text or len(cleaned_text) < self.MIN_RESPONSE_LENGTH:
                    self.logger.debug(f"Cleaned text too short ({len(cleaned_text)} chars) or empty. Skipping translation.")
                    return reconstruct_whitespace("", whitespace_info)

                if self.glossary_manager is None:
                    self.logger.error("GlossaryManager is not initialized. Cannot perform NER.")
                else:
                    self._identify_and_add_entities(cleaned_text, source_lang, target_lang, is_japanese_webnovel)

                if self.glossary_manager is None:
                    self.logger.error("GlossaryManager is not initialized. Cannot create placeholders.")
                    processed_text_with_placeholders = cleaned_text
                    self.active_placeholders = {}
                else:
                    processed_text_with_placeholders, self.active_placeholders = self.glossary_manager.create_placeholders(cleaned_text)

                estimated_total_tokens = self.estimate_tokens(processed_text_with_placeholders)
                model_info = self._wait_for_available_model(estimated_total_tokens)
                if not model_info:
                    self.logger.error("Translation skipped: No models available after waiting for the main translation request.")
                    return reconstruct_whitespace(cleaned_text, whitespace_info)

                model_type, model, limits = model_info
                self.logger.info(f"Using model {model_type.value} for translation.")

                chunks_with_placeholders = self._split_large_chunk(processed_text_with_placeholders, limits.max_input_tokens)
                translated_chunks =[]
                
                if chapter_id != self.current_chapter_id:
                    self._reset_chunk_count()
                self.current_chapter_id = chapter_id

                for i, chunk_with_placeholders in enumerate(chunks_with_placeholders):
                    if self.stop_event and self.stop_event.is_set():
                        self.logger.info(f"Stop signal received. Stopping translation after chunk {i}/{len(chunks_with_placeholders)}.")
                        translated_chunks.extend(chunks_with_placeholders[i:])
                        break

                    if not isinstance(chunk_with_placeholders, str) or not chunk_with_placeholders.strip() or len(chunk_with_placeholders.strip()) < self.MIN_RESPONSE_LENGTH:
                        translated_chunks.append(chunk_with_placeholders)
                        continue

                    chunk_translated_successfully = False
                    translated_chunk_text = chunk_with_placeholders
                    
                    use_continuation = False
                    if i > 0:
                        if len(chunks_with_placeholders) > 5 and i % 3 == 2:
                            self.logger.debug(f"Using full prompt for chunk {i+1} (index {i}) in chapter {chapter_id} as it's the {i//3 + 1}th third chunk.")
                        else:
                            use_continuation = True

                    for attempt in range(self.RETRY_ATTEMPTS):
                        if self.stop_event and self.stop_event.is_set():
                            break
                        try:
                            current_time = time.time()
                            if current_time - limits.last_reset_time >= 60:
                                limits.requests_made = 0
                                limits.tokens_used = 0
                                limits.last_reset_time = current_time

                            estimated_chunk_tokens = self.estimate_tokens(chunk_with_placeholders)
                            if limits.requests_made >= limits.rpm or limits.tokens_used + estimated_chunk_tokens > limits.tpm or limits.daily_requests >= limits.rpd:
                                wait_result = self._wait_for_available_model(estimated_chunk_tokens)
                                if wait_result:
                                    model_type, model, limits = wait_result
                                    continue
                                else:
                                    break

                            prompt = create_translation_prompt(chunk_with_placeholders, source_lang, target_lang, self.active_placeholders, is_continuation=use_continuation)
                            estimated_prompt_tokens = self.estimate_tokens(prompt)
                            
                            if self.stop_event and self.stop_event.is_set():
                                return None
                            response = self.client.models.generate_content(
                                model=model,
                                contents=prompt,
                                config={
                                    "temperature": 0.3,
                                    "maxOutputTokens": estimated_prompt_tokens * 2,
                                    "safety_settings": [
                                                {
                                                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                                                },
                                                {
                                                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                                                },
                                                {
                                                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                                                },
                                                {
                                                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                                                },
                                            ],
                                }
                            )

                            if response.prompt_feedback and response.prompt_feedback.block_reason:
                                translated_chunk_text = chunk_with_placeholders
                                chunk_translated_successfully = True
                                break

                            response_text = response.text if response.text is not None else ""
                            current_translated_chunk = response_text.strip()

                            placeholder_missing = False
                            for placeholder in self.active_placeholders:
                                if placeholder in chunk_with_placeholders and placeholder not in current_translated_chunk:
                                    placeholder_missing = True
                                    break
                                    
                            if placeholder_missing:
                                if attempt == self.RETRY_ATTEMPTS - 1:
                                    translated_chunk_text = chunk_with_placeholders
                                    chunk_translated_successfully = True
                                    break
                                continue

                            if not current_translated_chunk or len(current_translated_chunk) < self.MIN_RESPONSE_LENGTH:
                                if attempt == self.RETRY_ATTEMPTS - 1:
                                    translated_chunk_text = chunk_with_placeholders
                                    chunk_translated_successfully = True
                                else:
                                    raise ValueError("Empty or short response received.")
                                continue

                            if self.validator.validate_chunk(chunk_with_placeholders, current_translated_chunk):
                                translated_chunk_text = current_translated_chunk
                                chunk_translated_successfully = True
                                actual_output_tokens = self.estimate_tokens(response_text)
                                total_tokens = estimated_prompt_tokens + actual_output_tokens
                                
                                limits.tokens_used += total_tokens
                                limits.requests_made += 1
                                limits.daily_requests += 1
                                limits.last_requests_made_per_request = 1
                                limits.last_tokens_used_per_request = total_tokens
                                self._save_history()
                                time.sleep(self.BURST_DELAY)
                                break
                            else:
                                if attempt == self.RETRY_ATTEMPTS - 1:
                                    translated_chunk_text = chunk_with_placeholders
                                    chunk_translated_successfully = True

                        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, errors.ClientError) as e:
                            
                            retry_delay = (2 ** attempt)
                          
                            if hasattr(e, 'retry_delay') and e.retry_delay:
                                retry_delay = e.retry_delay.total_seconds()
                            
                            elif isinstance(e, errors.ClientError):
                                try:
                                    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(e))
                                    if match:
                                        retry_delay = float(match.group(1))
                                except Exception:
                                    pass 
                            
                            self.logger.warning(f"Rate limit hit. Waiting {retry_delay:.2f}s before attempt {attempt + 2}")
                            time.sleep(retry_delay + random.uniform(0, 1))
                            
                            if attempt == self.RETRY_ATTEMPTS - 1:
                                translated_chunk_text = chunk_with_placeholders
                                chunk_translated_successfully = True
                        except Exception as e:
                            sleep_duration = (2 ** attempt) + random.uniform(0, 1)
                            self._interruptible_sleep(sleep_duration)
                            if attempt == self.RETRY_ATTEMPTS - 1:
                                translated_chunk_text = chunk_with_placeholders
                                chunk_translated_successfully = True

                    translated_chunks.append(translated_chunk_text)
                    self.chunk_count_in_chapter += 1

                combined_translated_text_with_placeholders = '\n\n'.join(translated_chunks)

                if self.glossary_manager:
                    self.glossary_manager._validate_glossary_restoration(processed_text_with_placeholders, combined_translated_text_with_placeholders, self.active_placeholders)
                    final_translated_text_restored = self.glossary_manager.restore_placeholders(combined_translated_text_with_placeholders, self.active_placeholders)
                else:
                    final_translated_text_restored = combined_translated_text_with_placeholders

                self.active_placeholders = {}
                post_processed_text = self._post_process_translation(final_translated_text_restored)
                final_text = reconstruct_whitespace(post_processed_text, whitespace_info)
                self.translation_count += 1
                
                return final_text

            except Exception as e:
                self.logger.error(f"An unhandled error occurred during translation of a text block: {e}", exc_info=True)
                self._save_history()
                self.active_placeholders = {}
                _, whitespace_info_fallback = extract_whitespace_info(original_text)
                return reconstruct_whitespace(original_text.strip(), whitespace_info_fallback)

    def get_model_status(self) -> Dict:
        """Returns detailed status of all models including current rate limits and estimates."""
        status = {}
        current_time = time.time()

        for model_type, model_info in self.models.items():
            limits = model_info['limits']

            # Calculate remaining limits considering the reset window
            time_since_reset = current_time - limits.last_reset_time
            remaining_rpm = max(0, limits.rpm - limits.requests_made) if time_since_reset < 60 else limits.rpm
            remaining_tpm = max(0, limits.tpm - limits.tokens_used) if time_since_reset < 60 else limits.tpm

            time_since_daily_reset = current_time - limits.last_daily_reset
            remaining_rpd = max(0, limits.rpd - limits.daily_requests) if time_since_daily_reset < 86400 else limits.rpd


            # Estimate remaining chapters based on RPD and configured max_chapters_per_day
            estimated_chapters_remaining_by_rpd = max(0, limits.rpd - limits.daily_requests) # Assuming 1 request per chapter
            estimated_chapters_remaining_by_config = max(0, limits.max_chapters_per_day - limits.daily_requests) # Based on configured chapters/day
            estimated_chapters_remaining = min(
                estimated_chapters_remaining_by_rpd,
                estimated_chapters_remaining_by_config
            )
            estimated_chapters_remaining = max(0, estimated_chapters_remaining) # Ensure it's not negative


            status[model_type.value] = {
                'model_name': model_info.get('model_name', 'N/A'),
                'priority': model_info.get('priority', 'N/A'),
                'available': model_info.get('available', False) and model_info.get('model_instance') is not None, # True if initialized and available flag is True
                'limits': {
                    'tpm': limits.tpm,
                    'rpd': limits.rpd,
                    'rpm': limits.rpm,
                    'max_input_tokens': limits.max_input_tokens,
                    'max_chapters_per_day': limits.max_chapters_per_day,
                    'tokens_per_chapter': limits.tokens_per_chapter,
                },
                'current_usage': {
                    'requests_made_current_minute': limits.requests_made,
                    'tokens_used_current_minute': limits.tokens_used,
                    'daily_requests': limits.daily_requests,
                },
                'remaining': {
                    'remaining_rpm': remaining_rpm,
                    'remaining_tpm': remaining_tpm,
                    'remaining_rpd': remaining_rpd,
                    'estimated_chapters_remaining_today': estimated_chapters_remaining,
                },
                'last_reset_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(limits.last_reset_time)),
                'last_daily_reset': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(limits.last_daily_reset)),
                'last_requests_made_per_request': limits.last_requests_made_per_request, # Usage from the last successful request
                'last_tokens_used_per_request': limits.last_tokens_used_per_request,
            }
        return status

    def get_daily_chapter_capacity(self) -> int:
        """
        Returns total theoretical daily chapter capacity across all available models based on their configured `max_chapters_per_day`.
        Note: The actual capacity is limited by the sum of *remaining* capacities.
        """
        total_configured_capacity = 0
        for model_info in self.models.values():
            if model_info.get('available', True) and model_info.get('model_instance') is not None:
                total_configured_capacity += model_info['limits'].max_chapters_per_day
        return total_configured_capacity

    def optimize_for_chapters(self, estimated_chapters: int):
        """
        Optimizes model priority order based on estimated chapter count and models' *remaining* daily capacity (RPD and max_chapters_per_day).
        This function also performs the daily reset for all models if needed.
        """
        self.model_priority_order = self._calculate_optimal_model_for_chapters(estimated_chapters)

        # Provide feedback on remaining capacity after optimization and reset
        status = self.get_model_status()
        total_remaining_capacity_today = sum(
            model_status['remaining']['estimated_chapters_remaining_today']
            for model_status in status.values() if model_status['available']
        )

        self.logger.info(f"Optimization complete for estimated {estimated_chapters} chapters.")
        self.logger.info(f"Current total remaining estimated chapter capacity today across available models: {total_remaining_capacity_today}.")
        self.logger.info(f"New Model Priority Order: {[model.value for model in self.model_priority_order]}")

        if estimated_chapters > total_remaining_capacity_today and total_remaining_capacity_today > 0:
             # Find the earliest time a daily limit will reset among initialized models
             earliest_reset_time = float('inf')
             any_initialized = False
             for model_info in self.models.values():
                 if model_info.get('model_instance') is not None:
                     any_initialized = True
                     earliest_reset_time = min(earliest_reset_time, model_info['limits'].last_daily_reset + 86400) # 86400 seconds in a day

             if any_initialized:
                time_until_earliest_reset = max(0, earliest_reset_time - time.time())
                self.logger.warning(
                    f"Estimated chapters ({estimated_chapters}) exceeds total *remaining* estimated chapter capacity today ({total_remaining_capacity_today}). "
                    f"Translation of all chapters may require waiting for daily limits to reset (approx {time_until_earliest_reset:.0f}s until earliest model reset)."
                 )
             else:
                 self.logger.warning(f"Estimated chapters ({estimated_chapters}) exceeds total *remaining* estimated chapter capacity today ({total_remaining_capacity_today}). No models initialized to estimate reset time.")

        elif estimated_chapters > 0 and total_remaining_capacity_today == 0:
             self.logger.warning(
                 f"Estimated chapters ({estimated_chapters}) requires translation, but no models have remaining daily capacity. "
                 f"Translation cannot proceed until daily limits reset. Check get_model_status() for reset times."
             )
        elif estimated_chapters > 0:
             self.logger.info(f"Estimated chapters ({estimated_chapters}) is within current total remaining capacity ({total_remaining_capacity_today}).")

    def save_state(self):
        """Save glossary and history safely."""
        try:
            if self.glossary_manager:
                self.glossary_manager.save_glossary()
            self._save_history()
            self.logger.info("Translator state saved.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
        
    def reset(self):
        """Resets session counters and all model usage statistics, and optionally clears glossary."""
        self.translation_count = 0
        current_time = time.time()
        for model_info in self.models.values():
            limits = model_info['limits']
            limits.requests_made = 0
            limits.tokens_used = 0
            limits.daily_requests = 0
            limits.last_reset_time = current_time
            limits.last_daily_reset = current_time
        self.logger.info("Translator session and model usage history reset complete.")
        self._save_history()

    def get_translation_count(self) -> int:
        """Returns count of text blocks (not chunks) translated in current session."""
        return self.translation_count

    def get_remaining_translations(self) -> int:
        """Returns estimated remaining translations in current session based on session limit."""
        return max(0, self.max_translations_per_session - self.translation_count)