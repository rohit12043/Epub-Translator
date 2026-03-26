import html as html_module
import re
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Set
from bs4 import BeautifulSoup, NavigableString, Tag, Comment, Doctype, ProcessingInstruction
import ebooklib
from ebooklib import epub
from checkpoint import CheckpointManager
from text_utils import extract_whitespace_info, reconstruct_whitespace
from exceptions import ProhibitedContentError

# Double-newline is natural for the LLM to preserve and easy to split on.
BLOCK_SEPARATOR = "\n\n"

CLEANUP_REGEXES = [
    (re.compile(r'<p>\s*<\/p>', re.MULTILINE), ''),
    (re.compile(r'<(\w+)(\s[^>]*)?>\s*</\1>', re.MULTILINE), ''),
    (re.compile(r'^\s*[>\|]+\s*$', re.MULTILINE), ''),
    (re.compile(r'\[document\]'), ''),
    (re.compile(r'<\s*>'), ''),
    (re.compile(r'\n{3,}'), '\n\n'),
    (re.compile(r'<p\s*/>'), ''),
    (re.compile(r'<ruby>\s*</ruby>', re.MULTILINE), ''),
    (re.compile(r'<div>\s*</div>', re.MULTILINE), ''),
    (re.compile(r'<body><p>xml version=[\'"].*?</p>', re.MULTILINE), '<body>'),
]

#  TAG PRESERVER
#  replace inline HTML
#  tags with short [T0],[T1],… tokens before sending to LLM, restore after.
#  This guarantees <em>, <strong>, <ruby>, <span>, etc. survive translation intact
class TagPreserver:
    """
    Replaces inline HTML tags with [T0],[T1],… tokens so the LLM only sees
    plain text + tokens.  After translation the tokens are swapped back to
    their original tags, preserving ALL inline formatting perfectly.
    """

    INLINE_TAGS: frozenset = frozenset([
        'em', 'strong', 'b', 'i', 'u', 's', 'del', 'ins', 'mark',
        'small', 'sub', 'sup', 'code', 'span', 'a', 'ruby', 'rt', 'rp',
        'abbr', 'cite', 'dfn', 'kbd', 'samp', 'var', 'bdo', 'q',
    ])

    _INLINE_RE = re.compile(
        r'<(/?)(?:' + '|'.join(sorted(INLINE_TAGS, key=len, reverse=True)) + r')(\s[^>]*)?>',
        re.IGNORECASE,
    )
    _TOKEN_RE = re.compile(r'\[T(\d+)\]')

    @classmethod
    def preserve(cls, html_fragment: str) -> Tuple[str, Dict[str, str]]:
        """Replace inline tags with [T0],[T1],… Return (tokenised_text, tag_map)."""
        tag_map: Dict[str, str] = {}
        counter = [0]

        def _replace(m: re.Match) -> str:
            token = f"[T{counter[0]}]"
            tag_map[token] = m.group(0)
            counter[0] += 1
            return token

        result = cls._INLINE_RE.sub(_replace, html_fragment)
        return result, tag_map

    @classmethod
    def restore(cls, text_with_tokens: str, tag_map: Dict[str, str]) -> str:
        """Swap [Tn] tokens back to their original HTML tags."""
        if not tag_map:
            return text_with_tokens
        result = text_with_tokens
        for token in sorted(tag_map.keys(), key=lambda t: int(t[2:-1])):
            result = result.replace(token, tag_map[token])
        return result

    @classmethod
    def missing(cls, text: str, tag_map: Dict[str, str]) -> List[str]:
        """Return list of tokens from tag_map that are absent in text."""
        return [t for t in tag_map if t not in text]

    @classmethod
    def repair(cls, translated: str, original_with_tokens: str,
               tag_map: Dict[str, str]) -> str:
        """
        Proportional-reinsertion fallback 
        When the LLM drops a token, reinsert it at the proportionally equivalent
        position in the translated text, snapped to the nearest word boundary.
        """
        missing_tokens = cls.missing(translated, tag_map)
        if not missing_tokens:
            return translated

        orig_pure = cls._TOKEN_RE.sub('', original_with_tokens)
        orig_len = max(len(orig_pure), 1)
        trans_len = max(len(translated), 1)

        insertions: List[Tuple[int, int, str]] = []
        for token in missing_tokens:
            idx = original_with_tokens.find(token)
            if idx < 0:
                continue
            text_before = cls._TOKEN_RE.sub('', original_with_tokens[:idx])
            rel = len(text_before) / orig_len
            abs_pos = min(int(rel * trans_len), trans_len)
            # Snap to nearest word boundary
            while abs_pos > 0 and translated[abs_pos - 1] not in ' \t\n\r.,!?;:。，！？':
                abs_pos -= 1
            insertions.append((abs_pos, int(token[2:-1]), token))

        # Insert right-to-left so earlier positions stay valid
        insertions.sort(key=lambda x: (-x[0], x[1]))
        for pos, _, token in insertions:
            translated = translated[:pos] + token + translated[pos:]

        return translated

class EPUBProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.book = None
        self.total_chapters = 0
        self.current_chapter_index = 0
        self.excluded_keywords = ['toc', 'nav', 'cover', 'title', 'copyright', 'index', 'info']
        self.checkpoint_manager = CheckpointManager()
        self.translated_chunks_history: Dict[str, Dict[str, str]] = {}
        self.used_translated_chunks: Set[str] = set()
        self.target_lang = "English"

    def load_epub(self, filepath: str) -> bool:
        try:
            self.logger.info(f"Loading EPUB: {filepath}")
            self.book = epub.read_epub(filepath)
            self.content_chapters = self._get_content_items()
            self.total_chapters = len(self.content_chapters)
            self.current_chapter_index = 0
            if self.total_chapters == 0:
                raise ValueError("No content chapters found in EPUB")
            self.logger.info(f"Found {self.total_chapters} content chapters.")
            self.checkpoint_manager._load_checkpoint()
            self.checkpoint_data = self.checkpoint_manager.checkpoint_data
            self.translated_chunks_history.clear()
            for _, item_data in self.checkpoint_data.items():
                for item_id, item_chk_data in item_data.items():
                    if 'chunks' in item_chk_data:
                        if item_id not in self.translated_chunks_history:
                            self.translated_chunks_history[item_id] = {}
                        for chunk_key, chunk_chk_data in item_chk_data['chunks'].items():
                            if chunk_chk_data.get('completed_chunk'):
                                self.translated_chunks_history[item_id][chunk_key] = 'translated'
            return True
        except Exception as e:
            self.logger.error(f"Failed to load EPUB: {e}")
            return False

    def _get_content_items(self) -> List[Tuple[int, any]]:
        content_items = []
        included_patterns = [r'index_split_\d+\.html']
        toc_items_order = []

        def extract_hrefs(toc_item):
            hrefs = []
            if isinstance(toc_item, epub.Link):
                hrefs.append(toc_item.href.split('#')[0])
            elif isinstance(toc_item, (list, tuple)):
                for element in toc_item:
                    if isinstance(element, tuple) and len(element) == 2:
                        hrefs.extend(extract_hrefs(element[1]))
                    elif isinstance(element, list):
                        hrefs.extend(extract_hrefs(element))
                    elif isinstance(element, epub.Link):
                        hrefs.append(element.href.split('#')[0])
                    elif isinstance(element, epub.EpubHtml):
                        hrefs.append(element.href.split('#')[0])
            elif isinstance(toc_item, epub.EpubHtml):
                hrefs.append(toc_item.href.split('#')[0])
            return hrefs

        try:
            if self.book.toc:
                toc_items_order = extract_hrefs(self.book.toc)
                toc_items_order = list(dict.fromkeys(toc_items_order))
        except Exception as e:
            self.logger.warning(f"Could not parse TOC: {e}. Falling back to file-based ordering.")

        item_map = {
            item.get_name(): item
            for item in self.book.get_items()
            if hasattr(item, "get_content") and item.get_name().lower().endswith((".html", ".xhtml", ".htm"))
        }
        processed_items_names: Set[str] = set()
        chapter_counter = 0

        for href in toc_items_order:
            item_name_from_href = Path(href).name
            if item_name_from_href in item_map and item_name_from_href not in processed_items_names:
                item = item_map[item_name_from_href]
                item_name_lower = item.get_name().lower()
                if not any(keyword in item_name_lower for keyword in self.excluded_keywords):
                    chapter_counter += 1
                    content_items.append((chapter_counter, item))
                    processed_items_names.add(item_name_lower)

        for item in self.book.get_items():
            name = item.get_name().lower()
            if not (hasattr(item, "get_content") and name.endswith((".html", ".xhtml", ".htm"))):
                continue
            if name in processed_items_names:
                continue
            if any(re.match(pattern, name) for pattern in included_patterns):
                chapter_counter += 1
                content_items.append((chapter_counter, item))
                processed_items_names.add(name)
                continue
            if any(keyword in name for keyword in self.excluded_keywords):
                continue
            chapter_counter += 1
            content_items.append((chapter_counter, item))
            processed_items_names.add(name)

        content_items.sort(key=lambda x: x[0])
        return content_items

    def extract_text_with_structure(self, html_content: str) -> Tuple[str, Dict]:
        """
        Block-level extraction with inline tag preservation.

        Strategy (inspired by hydropix tag_preserver approach):
          1. Parse HTML, find all translatable block elements (p, h1-h6, li, …)
          2. For each block, extract its inner HTML and tokenise inline tags with
             TagPreserver → produces text_with_tokens + tag_map
          3. Return the concatenated block texts joined by BLOCK_SEPARATOR so the
             LLM sees clean paragraphs separated by double-newlines.
          4. Store element references + tag_maps in structure_info for later
             in-place reconstruction.

        This preserves <em>, <strong>, <ruby>, <span class="...">, etc. perfectly
        instead of losing them through the old heuristic approach.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for el in soup(['script', 'style', 'meta', 'link', 'br']):
                el.decompose()
            for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()

            div_classes = []
            for div in soup.find_all('div', class_=True):
                cls = div.get('class') or []
                div_classes.extend(cls if isinstance(cls, list) else [cls])

            structure_info: Dict = {
                'soup': soup,
                'original_html': html_content,
                'blocks': [],
                'chunk_block_counts': [],
                'paragraphs': [],
                'original_structure': [],
                'whitespace': {},
                'div_classes': div_classes,
                'comments': [],
                'processing_instructions': [],
            }

            BLOCK_TAGS = frozenset([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'li', 'blockquote', 'td', 'th', 'dt', 'dd',
            ])

            blocks: List[Dict] = []
            text_lines: List[str] = []
            seen_ids: Set[int] = set()

            for element in soup.find_all(BLOCK_TAGS):
                eid = id(element)
                if eid in seen_ids:
                    continue

                # Skip blocks nested inside another BLOCK_TAG that need processing
                # (exception: p/li/td inside blockquote/li/td are kept)
                skip = False
                for ancestor in element.parents:
                    if ancestor.name in BLOCK_TAGS and id(ancestor) not in seen_ids:
                        if ancestor.name not in {'blockquote', 'li', 'td', 'th', 'dt', 'dd'}:
                            skip = True
                            break
                if skip:
                    continue

                seen_ids.add(eid)
                inner_html = element.decode_contents()
                if not inner_html.strip():
                    continue

                text_with_tokens, tag_map = TagPreserver.preserve(inner_html)
                clean_text = re.sub(r'\s+', ' ', text_with_tokens).strip()
                if not clean_text:
                    continue

                blocks.append({
                    'element': element,
                    'text_with_tokens': clean_text,
                    'tag_map': tag_map,
                    'block_tag': element.name,
                    'attrs': dict(element.attrs) if element.attrs else {},
                    'original_inner_html': inner_html,
                })
                text_lines.append(clean_text)

            structure_info['blocks'] = blocks
            full_text = BLOCK_SEPARATOR.join(text_lines)
            self.logger.debug(f"Extracted {len(blocks)} blocks ({len(full_text)} chars).")
            return full_text, structure_info

        except Exception as e:
            self.logger.error(f"Error in extract_text_with_structure: {e}")
            return "", {
                'soup': None, 'original_html': html_content, 'blocks': [],
                'chunk_block_counts': [], 'paragraphs': [], 'original_structure': [],
                'whitespace': {}, 'div_classes': [], 'comments': [],
                'processing_instructions': [],
            }

    def intelligent_chunk_text(self, text: str, structure_info: Dict,
                                max_chars: int = 12000) -> List[Tuple[str, Dict]]:
        """
        Block-aware chunking: always split at BLOCK_SEPARATOR boundaries so a
        block is never torn in half.  Records chunk_block_counts in structure_info
        so reconstruction knows which translated blocks belong to which chunk.
        """
        try:
            if not text.strip():
                return []

            raw_blocks = [b for b in text.split(BLOCK_SEPARATOR) if b.strip()]
            if not raw_blocks:
                return []

            target_capacity = int(max_chars * 0.85)
            chunks: List[Tuple[str, Dict]] = []
            current_blocks: List[str] = []
            current_len = 0
            chunk_block_counts: List[int] = []

            for block in raw_blocks:
                block_len = len(block)
                sep_len = len(BLOCK_SEPARATOR) if current_blocks else 0

                if current_blocks and current_len + sep_len + block_len > target_capacity:
                    chunk_text = BLOCK_SEPARATOR.join(current_blocks)
                    chunks.append((chunk_text, structure_info))
                    chunk_block_counts.append(len(current_blocks))
                    current_blocks = [block]
                    current_len = block_len
                else:
                    current_blocks.append(block)
                    current_len += sep_len + block_len

            if current_blocks:
                chunks.append((BLOCK_SEPARATOR.join(current_blocks), structure_info))
                chunk_block_counts.append(len(current_blocks))

            structure_info['chunk_block_counts'] = chunk_block_counts
            return chunks if chunks else [(text, structure_info)]

        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            return [(text, structure_info)] if text else []

    def reconstruct_html_with_structure(self, original_content: str,
                                         translated_chunks: List[str],
                                         structure_info: Dict,
                                         stop_flag: Optional[Callable] = None) -> str:
        """
        Reconstruct the chapter HTML by:
          1. Splitting each translated chunk back into individual block texts
             (using BLOCK_SEPARATOR and the stored chunk_block_counts).
          2. Restoring [Tn] tokens: original HTML tags per block via TagPreserver.
          3. Updating each block's soup element in-place (clear + re-add children).
          4. Serialising the modified soup, re-attaching the original XML/DOCTYPE
             preamble so EPUB structure stays intact.
        """
        try:
            soup = structure_info.get('soup')
            blocks: List[Dict] = structure_info.get('blocks', [])
            chunk_block_counts: List[int] = structure_info.get('chunk_block_counts', [])

            valid_chunks = [c for c in translated_chunks if c and isinstance(c, str) and c.strip()]
            if not blocks or soup is None or not valid_chunks:
                self.logger.warning("reconstruct: missing blocks/soup/chunks — returning original.")
                return original_content

            block_offset = 0
            for chunk_i, translated_chunk in enumerate(valid_chunks):
                if stop_flag and stop_flag():
                    self.logger.info("Reconstruction stopped by stop_flag.")
                    return original_content

                # Split translated chunk back into per-block pieces
                translated_block_texts = [
                    b.strip() for b in translated_chunk.split(BLOCK_SEPARATOR) if b.strip()
                ]

                # How many original blocks does this chunk represent?
                n_orig = (chunk_block_counts[chunk_i]
                          if chunk_i < len(chunk_block_counts)
                          else len(translated_block_texts))

                for j in range(n_orig):
                    if block_offset + j >= len(blocks):
                        break

                    block_info = blocks[block_offset + j]
                    tag_map = block_info['tag_map']
                    orig_with_tokens = block_info['text_with_tokens']
                    element = block_info['element']

                    # Use translated text if available, else fall back to original
                    if j < len(translated_block_texts):
                        translated_text = translated_block_texts[j]
                    else:
                        translated_text = orig_with_tokens

                    # Repair missing tokens before restoration
                    if TagPreserver.missing(translated_text, tag_map):
                        translated_text = TagPreserver.repair(
                            translated_text, orig_with_tokens, tag_map
                        )

                    restored_inner = TagPreserver.restore(translated_text, tag_map)

                    # Update element contents in-place (preserves tag + attrs)
                    try:
                        element.clear()
                        frag = BeautifulSoup(
                            f'<{element.name}>{restored_inner}</{element.name}>',
                            'html.parser'
                        ).find(element.name)
                        if frag:
                            for child in list(frag.contents):
                                element.append(child)
                        else:
                            # Fallback: just set string content
                            element.string = restored_inner
                    except Exception as e:
                        self.logger.warning(f"Block update failed at offset {block_offset+j}: {e}")

                block_offset += n_orig

            body_tag = soup.body
            if body_tag is None:
                result = str(soup)
            else:
                body_html = str(body_tag)
                # Extract preamble (everything before <body …>)
                body_start_m = re.search(r'<body[^>]*>', original_content, re.IGNORECASE)
                if body_start_m:
                    preamble = original_content[:body_start_m.start()]
                    # Update lang attributes for target language
                    preamble = re.sub(
                        r'(?<=\s)lang=["\'][^"\']*["\']',
                        f'lang="{self.target_lang}"', preamble
                    )
                    preamble = re.sub(
                        r'(?<=\s)xml:lang=["\'][^"\']*["\']',
                        f'xml:lang="{self.target_lang}"', preamble
                    )
                    result = preamble + body_html + '\n</html>'
                else:
                    result = str(soup)

            # Apply cleanup regexes
            for pattern, replacement in CLEANUP_REGEXES:
                result = pattern.sub(replacement, result)

            return result

        except Exception as e:
            self.logger.error(f"HTML reconstruction failed: {e} — returning original HTML")
            return original_content

    def _load_chunk_from_checkpoint(self, checkpoint_key_prefix: str, item_id: str,
                                    chunk_index: int) -> Optional[str]:
        chunk_key = f"chunk{chunk_index}"
        chunk_data = self.checkpoint_manager.get_chunk_data(checkpoint_key_prefix, item_id, chunk_key)
        if chunk_data and chunk_data.get('completed_chunk', False):
            translated_blocks: List[str] = []
            line_index = 0
            while f"line{line_index}" in chunk_data['lines']:
                try:
                    line_data = chunk_data['lines'][f"line{line_index}"]
                    if isinstance(line_data, str):
                        line_data = json.loads(line_data)
                    translated_blocks.append(line_data['text'])
                except (json.JSONDecodeError, TypeError, KeyError):
                    self.logger.warning(
                        f"Invalid checkpoint data for {chunk_key}_line{line_index}. Skipping."
                    )
                    break
                line_index += 1
            # Join with BLOCK_SEPARATOR so reconstruction can split correctly
            return BLOCK_SEPARATOR.join(translated_blocks)
        return None

    def _translate_and_save_chunk(self, translator, chunk_text: str, source_lang: str,
                                   target_lang: str, is_japanese_webnovel: bool,
                                   checkpoint_key_prefix: str, item_id: str,
                                   chunk_key: str, chunk_index: int,
                                   item_name: str, total_chunks: int) -> str:
        original_chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()

        checkpointed_chunk_data = self.checkpoint_manager.get_chunk_data(
            checkpoint_key_prefix, item_id, chunk_key
        )
        if checkpointed_chunk_data and checkpointed_chunk_data.get('completed_chunk', False):
            if checkpointed_chunk_data.get('original_hash') == original_chunk_hash:
                self.logger.info(
                    f"Loading translated chunk {chunk_index + 1}/{total_chunks} of {item_name}."
                )
                loaded_text = self._load_chunk_from_checkpoint(
                    checkpoint_key_prefix, item_id, chunk_index
                )
                if loaded_text:
                    self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
                    return loaded_text
            else:
                self.logger.warning(
                    f"Hash mismatch for chunk {chunk_index + 1} of {item_name}. Re-translating."
                )

        self.logger.info(f"Translating chunk {chunk_index + 1}/{total_chunks} of {item_name}.")
        try:
            translated_text = translator.translate_text(
                chunk_text, source_lang, target_lang, is_japanese_webnovel
            )

            tokens_used, requests_made = 0, 0
            for _, info in translator.get_model_status().items():
                if info['last_requests_made_per_request'] > 0:
                    tokens_used = info['last_tokens_used_per_request']
                    requests_made = info['last_requests_made_per_request']
                    break

            if isinstance(translated_text, str) and translated_text.strip():
                # Detect untranslated (identical) chunks and fall back
                src_clean = re.sub(r'\[T\d+\]|__[A-Z_]+_\d+__', '', chunk_text).strip()
                tgt_clean = re.sub(r'\[T\d+\]|__[A-Z_]+_\d+__', '', translated_text).strip()
                if src_clean == tgt_clean:
                    self.logger.warning(
                        f"Translation identical to original for chunk {chunk_index + 1}, using original."
                    )
                    translated_text = chunk_text

                # Save each block as a checkpoint "line"
                raw_blocks = [b.strip() for b in translated_text.split(BLOCK_SEPARATOR) if b.strip()]
                for block_index, block in enumerate(raw_blocks):
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_key_prefix, item_id, chunk_key,
                        translated_line=block,
                        line_index=block_index,
                        is_dialogue=False,
                        tokens_used=tokens_used,
                        requests_made=requests_made,
                        original_chunk_hash=original_chunk_hash,
                    )
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_key_prefix, item_id, chunk_key, completed=True
                )
                self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
                return translated_text

        except ProhibitedContentError as pce:
            self.logger.error(f"Prohibited content in chunk {chunk_index + 1}: {pce}")
        except Exception as e:
            self.logger.error(f"Translation failed for chunk {chunk_index + 1}: {e}. Using original.")

        # Fallback: save original as checkpoint so we don't re-attempt indefinitely
        self.checkpoint_manager.save_checkpoint(
            checkpoint_key_prefix, item_id, chunk_key,
            original_chunk_hash=original_chunk_hash, completed=True
        )
        self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
        return chunk_text


    def translate_epub(self, translator, source_lang: str, target_lang: str,
                       output_path: str,
                       progress_callback: Optional[Callable] = None,
                       status_callback: Optional[Callable] = None,
                       stop_flag: Optional[Callable] = None,
                       is_japanese_webnovel: bool = False,
                       start_chapter: int = 1,
                       end_chapter: Optional[int] = None) -> bool:
        try:
            out_path_obj = Path(output_path)
            temp_output_path = out_path_obj.with_suffix('.temp.epub')
            self.target_lang = target_lang
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)

            use_temp_epub = False
            if temp_output_path.exists():
                try:
                    temp_book = epub.read_epub(str(temp_output_path))
                    for _, item_data in self.checkpoint_manager.checkpoint_data.items():
                        for _, item_chk_data in item_data.items():
                            if item_chk_data.get('completed', False):
                                use_temp_epub = True
                                break
                        if use_temp_epub:
                            break
                    if use_temp_epub:
                        self.book = temp_book
                        self.logger.info(f"Resuming from temp EPUB: {temp_output_path}")
                    else:
                        self.logger.info("Temp EPUB exists but no completed translations; using original.")
                except Exception as e:
                    self.logger.warning(f"Failed to load temp EPUB: {e}. Using original.")

            if not use_temp_epub and not self.book:
                raise ValueError("No EPUB loaded.")
            if not use_temp_epub:
                self.logger.info("Starting from original EPUB.")

            content_chapters_to_process = [
                (chap_idx, item) for chap_idx, item in self.content_chapters
                if start_chapter <= chap_idx and (end_chapter is None or chap_idx <= end_chapter)
            ]
            if not content_chapters_to_process:
                self.logger.warning(f"No chapters in range {start_chapter}-{end_chapter}.")
                return True

            if status_callback:
                status_callback("Starting translation…")

            total_selected = len(content_chapters_to_process)
            processed_count = 0

            try:
                for chapter_counter, item in content_chapters_to_process:
                    if stop_flag and stop_flag():
                        self.logger.info("Translation stopped by user.")
                        self._save_intermediate_epub(str(temp_output_path))
                        return False

                    self.current_chapter_index = chapter_counter
                    item_id = item.get_id() or f"item_{chapter_counter}"
                    checkpoint_key_prefix = f"{out_path_obj.stem}_{item_id}"

                    if self.checkpoint_manager.get_item_completion_status(checkpoint_key_prefix, item_id):
                        self.logger.info(
                            f"Skipping already translated chapter {chapter_counter}/{self.total_chapters}: {item_id}."
                        )
                        processed_count += 1
                        if progress_callback:
                            progress_callback((processed_count / total_selected) * 100)
                        continue

                    if status_callback:
                        status_callback(
                            f"Translating Chapter {chapter_counter}/{self.total_chapters}: {item.get_name()}…"
                        )

                    content = item.get_content().decode('utf-8')
                    cleaned_text, structure_info = self.extract_text_with_structure(content)

                    if not cleaned_text.strip():
                        self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                        processed_count += 1
                        if progress_callback:
                            progress_callback((processed_count / total_selected) * 100)
                        continue

                    chunks = self.intelligent_chunk_text(cleaned_text, structure_info)
                    if not chunks:
                        self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                        processed_count += 1
                        if progress_callback:
                            progress_callback((processed_count / total_selected) * 100)
                        continue

                    translated_chunks: List[Optional[str]] = [None] * len(chunks)
                    for j, (chunk_text, _) in enumerate(chunks):
                        translated_chunks[j] = self._translate_and_save_chunk(
                            translator=translator,
                            chunk_text=chunk_text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            is_japanese_webnovel=is_japanese_webnovel,
                            checkpoint_key_prefix=checkpoint_key_prefix,
                            item_id=item_id,
                            chunk_key=f"chunk{j}",
                            chunk_index=j,
                            item_name=item.get_name(),
                            total_chunks=len(chunks),
                        )
                        if j % 5 == 0:
                            self._save_intermediate_epub(str(temp_output_path))

                    new_content = self.reconstruct_html_with_structure(
                        content, translated_chunks, structure_info, stop_flag
                    )
                    item.set_content(new_content.encode('utf-8'))
                    self._save_intermediate_epub(str(temp_output_path))
                    self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                    processed_count += 1
                    if progress_callback:
                        progress_callback((processed_count / total_selected) * 100)

            except KeyboardInterrupt:
                self.logger.info("Interrupted by user. Temp EPUB saved.")
                self._save_intermediate_epub(str(temp_output_path))
                return False

            self._save_final_epub(str(out_path_obj))
            temp_output_path.unlink(missing_ok=True)
            self.logger.info("EPUB translation completed successfully.")
            return True

        except Exception as e:
            self.logger.error(f"EPUB translation failed: {e}")
            return False

    def _save_intermediate_epub(self, filepath: str):
        try:
            temp_file = Path(filepath).with_suffix('.saving.epub')
            epub.write_epub(str(temp_file), self.book)
            temp_file.replace(filepath)
            self.logger.debug(f"Intermediate EPUB saved: {filepath}")
        except TypeError as e:
            if "'Link' object is not iterable" in str(e):
                self.logger.warning("Malformed TOC — flattening for save.")
                try:
                    flat_links = []
                    for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                        title = getattr(item, 'title', None) or item.get_name()
                        flat_links.append(epub.Link(item.get_name(), title, item.get_name()))
                    self.book.toc = flat_links
                    temp_file = Path(filepath).with_suffix('.saving.epub')
                    epub.write_epub(str(temp_file), self.book)
                    temp_file.replace(filepath)
                except Exception as fe:
                    self.logger.warning(f"Failed after TOC flatten: {fe}")
            else:
                self.logger.warning(f"Intermediate save failed (TypeError): {e}")
        except Exception as e:
            self.logger.warning(f"Intermediate save failed: {e}")

    def _save_final_epub(self, filepath: str):
        try:
            epub.write_epub(filepath, self.book)
            self.logger.info(f"Final EPUB saved: {filepath}")
            try:
                temp_path = Path(filepath).with_suffix('.temp.epub')
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Failed to save final EPUB: {e}")
            raise
    def cleanup(self):
        self.book = None
        self.total_chapters = 0
        self.current_chapter_index = 0
        self.translated_chunks_history.clear()
        self.used_translated_chunks.clear()
        if hasattr(self, "chapter_ner_cache"):
            self.chapter_ner_cache.clear()

    def get_translation_progress(self) -> Tuple[int, int]:
        return self.current_chapter_index, self.total_chapters

    def get_total_chapters(self) -> int:
        return self.total_chapters