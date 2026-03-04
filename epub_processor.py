import html
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
from text_utils import parse_line, extract_whitespace_info, reconstruct_whitespace
from exceptions import ProhibitedContentError

CLEANUP_REGEXES =[
    (re.compile(r'<p>\s*<\/p>', re.MULTILINE), ''),
    (re.compile(r'<(\w+)(\s[^>]*)?>\s*</\1>', re.MULTILINE), ''),
    (re.compile(r'^\s*[>\|]+\s*$', re.MULTILINE), ''),
    (re.compile(r'(?:>|>)\s*(?=</(?:div|body|html)>)'), ''),
    (re.compile(r'\[document\]'), ''),
    (re.compile(r'<[document]\s*>'), ''),
    (re.compile(r'<\s*>'), ''),
    (re.compile(r'\n{2,}'), '\n'),
    (re.compile(r'<p\s*/>'), ''),
    (re.compile(r'<ruby>\s*</ruby>', re.MULTILINE), ''),
    (re.compile(r'<div>\s*</div>', re.MULTILINE), ''),
    (re.compile(r'<body><p>xml version=[\'"].*?</p>', re.MULTILINE), '<body>')
]

class EPUBProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.book = None
        self.total_chapters = 0
        self.current_chapter_index = 0
        self.excluded_keywords =['toc', 'nav', 'cover', 'title', 'copyright', 'index', 'info']
        self.checkpoint_manager = CheckpointManager()
        self.translated_chunks_history: Dict[str, Dict[str, str]] = {}
        self.used_translated_chunks: Set[str] = set()

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
            
            for checkpoint_key_prefix, item_data in self.checkpoint_data.items():
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
        content_items =[]
        included_patterns = [r'index_split_\d+\.html']
        toc_items_order =[]
        
        # --- NEW SAFE TOC PARSING HELPER ---
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
        # -----------------------------------

        try:
            if self.book.toc:
                toc_items_order = extract_hrefs(self.book.toc)
                # Remove duplicates while preserving order
                toc_items_order = list(dict.fromkeys(toc_items_order))
        except Exception as e:
            self.logger.warning(f"Could not parse TOC: {e}. Falling back to file-based ordering.")
            
        item_map = {
            item.get_name(): item
            for item in self.book.get_items()
            if hasattr(item, "get_content") and item.get_name().lower().endswith((".html", ".xhtml", ".htm"))
        }
        processed_items_names = set()
        chapter_counter = 0
        
        for href in toc_items_order:
            item_name_from_href = Path(href).name
            if item_name_from_href in item_map and item_name_from_href not in processed_items_names:
                item = item_map[item_name_from_href]
                item_name_lower = item.get_name().lower()
                if not any(keyword in item_name_lower for keyword in self.excluded_keywords):
                    chapter_counter += 1
                    content_items.append((chapter_counter, item))
                    self.logger.debug(f"Including (TOC, Chapter {chapter_counter}): {item_name_lower}")
                processed_items_names.add(item_name_lower)
                
        for item in self.book.get_items():
            name = item.get_name().lower()
            if hasattr(item, "get_content") and name.endswith((".html", ".xhtml", ".htm")):
                item_name = item.get_name().lower()
                if item_name in processed_items_names:
                    continue
                if any(re.match(pattern, item_name) for pattern in included_patterns):
                    chapter_counter += 1
                    content_items.append((chapter_counter, item))
                    self.logger.debug(f"Including (Pattern Match, Chapter {chapter_counter}): {item_name}")
                    processed_items_names.add(item_name)
                    continue
                if any(keyword in item_name for keyword in self.excluded_keywords):
                    self.logger.debug(f"Skipping excluded item: {item_name}")
                    continue
                chapter_counter += 1
                content_items.append((chapter_counter, item))
                self.logger.debug(f"Including (Default Order, Chapter {chapter_counter}): {item_name}")
                processed_items_names.add(item_name)
                
        content_items.sort(key=lambda x: x[0])
        return content_items

    def extract_text_with_structure(self, html_content: str) -> Tuple[str, Dict]:
        try:
            soup = BeautifulSoup(html_content, features="html.parser")
            for element in soup(['script', 'style', 'meta', 'link', 'br']):
                element.decompose()
                
            structure_info = {
                'paragraphs': [],
                'original_structure':[],
                'whitespace': {},
                'div_classes': [],
                'comments':[],
                'processing_instructions': []
            }
            text_segments =[]

            def extract_from_element(element, depth=0):
                if isinstance(element, Comment):
                    structure_info['comments'].append({
                        'content': str(element),
                        'position': len(text_segments)
                    })
                    return
                if isinstance(element, ProcessingInstruction):
                    structure_info['processing_instructions'].append({
                        'content': str(element),
                        'position': len(text_segments)
                    })
                    return
                if isinstance(element, Doctype):
                    return
                if isinstance(element, NavigableString):
                    text = str(element)
                    if text.strip():
                        cleaned_text = re.sub(r'\s+', ' ', text).strip()
                        text_segments.append(cleaned_text)
                        position = len(text_segments) - 1
                        parent_tag = element.parent.name if element.parent else None
                        parent_attrs = dict(element.parent.attrs) if element.parent and element.parent.attrs else {}
                        structure_info['original_structure'].append({
                            'type': 'text',
                            'content': cleaned_text,
                            'original': text,
                            'position': position,
                            'parent_tag': parent_tag,
                            'parent_attrs': parent_attrs
                        })
                        structure_info['whitespace'][position] = {
                            'leading': text.startswith(' '),
                            'trailing': text.endswith(' ')
                        }
                        if parent_tag == 'p':
                            structure_info['paragraphs'].append({
                                'type': 'text',
                                'position': position,
                                'content': cleaned_text
                            })
                elif isinstance(element, Tag):
                    tag_name = element.name.lower()
                    attrs = dict(element.attrs) if element.attrs else {}
                    if tag_name == 'div' and 'class' in attrs:
                        if isinstance(attrs['class'], list):
                            structure_info['div_classes'].extend(attrs['class'])
                        else:
                            structure_info['div_classes'].append(attrs['class'])
                    structure_info['original_structure'].append({
                        'type': 'tag_start',
                        'tag': tag_name,
                        'position': len(text_segments),
                        'attrs': attrs
                    })
                    for child in element.children:
                        extract_from_element(child)
                    structure_info['original_structure'].append({
                        'type': 'tag_end',
                        'tag': tag_name,
                        'position': len(text_segments)
                    })
                    if tag_name == 'p':
                        structure_info['paragraphs'].append({
                            'type': 'tag_end',
                            'tag': 'p',
                            'position': len(text_segments)
                        })

            extract_from_element(soup)
            full_text = '\n'.join(text_segments)
            self.logger.debug(f"Extracted {len(full_text)} chars, {len(text_segments)} segments.")
            return full_text, structure_info
        except Exception as e:
            self.logger.error(f"Error extracting text with structure: {e}")
            return "", {'paragraphs':[], 'original_structure': [], 'whitespace': {}, 'div_classes': [], 'comments': [], 'processing_instructions':[]}

    def intelligent_chunk_text(self, text: str, structure_info: Dict, max_chars: int = 10000) -> List[Tuple[str, Dict]]:
        try:
            if not text.strip():
                self.logger.debug("No text provided for chunking.")
                return[]
            if len(text) <= max_chars:
                self.logger.debug("Text fits within max_chars, returning single chunk.")
                return [(text, structure_info)]
                
            chunks =[]
            current_chunk_lines =[]
            current_length = 0
            current_chunk_structure = {
                'paragraphs': [],
                'original_structure':[],
                'whitespace': {},
                'div_classes': list(set(structure_info.get('div_classes',[])))
            }
            temp_original_structure = []
            
            for struct_item in structure_info['original_structure']:
                if struct_item['type'] == 'text':
                    line_content = struct_item['content']
                    line_length = len(line_content) + 1
                    
                    if current_length + line_length > max_chars and current_chunk_lines:
                        chunks.append(('\n'.join(current_chunk_lines), {
                            'paragraphs': current_chunk_structure['paragraphs'],
                            'original_structure': temp_original_structure,
                            'whitespace': current_chunk_structure['whitespace'],
                            'div_classes': current_chunk_structure['div_classes'],
                            'comments': structure_info.get('comments',[]),
                            'processing_instructions': structure_info.get('processing_instructions',[])
                        }))
                        self.logger.debug(f"Created chunk {len(chunks)} with {current_length} chars.")
                        current_chunk_lines =[]
                        current_length = 0
                        current_chunk_structure['paragraphs'] = []
                        current_chunk_structure['whitespace'] = {}
                        temp_original_structure =[]
                        
                    current_chunk_lines.append(line_content)
                    current_length += line_length
                    temp_original_structure.append(struct_item)
                    current_chunk_structure['whitespace'][struct_item['position']] = structure_info['whitespace'].get(struct_item['position'], {})
                    
                    if any(p['type'] == 'text' and p['position'] == struct_item['position'] for p in structure_info['paragraphs']):
                        current_chunk_structure['paragraphs'].append({
                            'type': 'text',
                            'position': struct_item['position'],
                            'content': struct_item['content']
                        })
                elif struct_item['type'] in ['tag_start', 'tag_end']:
                    temp_original_structure.append(struct_item)
                    if struct_item['type'] == 'tag_end' and struct_item['tag'] == 'p':
                        current_chunk_structure['paragraphs'].append(struct_item)
                        
            if current_chunk_lines:
                chunks.append(('\n'.join(current_chunk_lines), {
                    'paragraphs': current_chunk_structure['paragraphs'],
                    'original_structure': temp_original_structure,
                    'whitespace': current_chunk_structure['whitespace'],
                    'div_classes': current_chunk_structure['div_classes'],
                    'comments': structure_info.get('comments',[]),
                    'processing_instructions': structure_info.get('processing_instructions',[])
                }))
                self.logger.debug(f"Created final chunk {len(chunks)} with {current_length} chars.")
                
            if not chunks:
                self.logger.warning("No chunks generated, returning original text as a single chunk.")
                return [(text, structure_info)]
                
            self.logger.debug(f"Created {len(chunks)} chunks with max_chars={max_chars}.")
            return chunks
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            return[(text, structure_info)] if text else[]

    def reconstruct_html_with_structure(self, original_content: str, translated_chunks: List[str], structure_info: Dict, stop_flag: Optional[Callable] = None) -> str:
        try:
            valid_chunks =[chunk for chunk in translated_chunks if chunk and isinstance(chunk, str) and chunk.strip()]
            if not valid_chunks:
                self.logger.warning("No valid translated chunks found, returning original content")
                return original_content

            soup = BeautifulSoup(original_content, 'html.parser')
            doc_type = f"<!DOCTYPE {soup.contents[0]}>" if soup.contents and isinstance(soup.contents[0], Doctype) else ""
            
            head_content = str(soup.head) if soup.head else '<head><title>Translated Content</title><meta charset="utf-8"/></head>'
            head_soup = BeautifulSoup(head_content, 'html.parser')
            if not head_soup.find('meta', {'charset': 'utf-8'}):
                head_soup.head.append(BeautifulSoup('<meta charset="utf-8"/>', 'html.parser').meta)
            head_content = str(head_soup)
            
            div_class = structure_info.get('div_classes', [''])[0] if structure_info.get('div_classes') else ''
            lines =[
                '<?xml version="1.0" encoding="utf-8"?>',
                doc_type,
                f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{self.target_lang}" xml:lang="{self.target_lang}">',
                head_content,
                '<body>'
            ]
            
            for pi in structure_info.get('processing_instructions', []):
                lines.append(pi['content'])
            for comment in structure_info.get('comments',[]):
                lines.append(f'')
                
            chunk_index = 0
            in_div = False
            title_added = False
            open_tags = []
            tag_stack =[]
            
            if div_class:
                lines.append(f'<div class="{div_class}">')
                in_div = True
                open_tags.append('div')
                tag_stack.append('div')
                
            for struct_item in structure_info.get('original_structure',[]):
                if stop_flag and stop_flag():
                    self.logger.debug("Reconstruction stopped by stop_flag. Returning original content to prevent data loss.")
                    return original_content
                    
                if isinstance(struct_item, dict):
                    if struct_item['type'] == 'tag_start':
                        tag = struct_item['tag']
                        if tag == 'div' and in_div and tag_stack and tag_stack[0] == 'div':
                            continue
                        attrs = struct_item.get('attrs', {})
                        attrs_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items()]) if attrs else ''
                        if tag == 'h1' and not title_added and valid_chunks:
                            title_text = valid_chunks[0].split('\n')[0].strip()
                            lines.append(f'<h1>{html.escape(title_text)}</h1>')
                            title_added = True
                        else:
                            lines.append(f'<{tag}{" " + attrs_str if attrs_str else ""}>')
                            open_tags.append(tag)
                            tag_stack.append(tag)
                    elif struct_item['type'] == 'tag_end':
                        tag = struct_item['tag']
                        if tag == 'div' and in_div and tag_stack and tag_stack[0] == 'div':
                            continue
                        while tag_stack and tag_stack[-1] != tag:
                            mismatched_tag = tag_stack.pop()
                            lines.append(f'</{mismatched_tag}>')
                            if mismatched_tag in open_tags:
                                open_tags.remove(mismatched_tag)
                        if tag_stack and tag_stack[-1] == tag:
                            tag_stack.pop()
                            lines.append(f'</{tag}>')
                            if tag in open_tags:
                                open_tags.remove(tag)
                    elif struct_item['type'] == 'text' and chunk_index < len(valid_chunks):
                        text = valid_chunks[chunk_index]
                        if not text.strip():
                            chunk_index += 1
                            continue
                            
                        _, whitespace_info = extract_whitespace_info(text)
                        if struct_item['position'] in structure_info.get('whitespace', {}):
                            orig_whitespace = structure_info['whitespace'][struct_item['position']]
                            if orig_whitespace.get('leading'):
                                text = ' ' + text
                            if orig_whitespace.get('trailing'):
                                text = text + ' '
                                
                        paragraphs = re.split(r'\n\s*\n', text.strip())
                        for para_idx, paragraph in enumerate(paragraphs):
                            if not paragraph.strip():
                                continue
                            paragraph = re.sub(r'\.{3,}', '___ELLIPSIS___', paragraph)
                            paragraph = re.sub(r'…', '___ELLIPSIS___', paragraph)
                            sentence_parts = re.split(r'([.!?。？！]+)', paragraph.strip())
                            sentences =[]
                            for i in range(0, len(sentence_parts) - 1, 2):
                                sentence = sentence_parts[i] + (sentence_parts[i + 1] if i + 1 < len(sentence_parts) else "")
                                if sentence.strip():
                                    sentences.append(sentence)
                            if len(sentence_parts) % 2 == 1 and sentence_parts[-1].strip():
                                sentences.append(sentence_parts[-1])
                                
                            sentences =[s.replace('___ELLIPSIS___', '...') for s in sentences]
                            for sentence in sentences:
                                if not sentence.strip():
                                    continue
                                parsed_line = parse_line(sentence)
                                if parsed_line:
                                    lines.append(parsed_line)
                            if para_idx < len(paragraphs) - 1:
                                lines.append('<p></p>')
                                
                        text = reconstruct_whitespace(text, whitespace_info)
                        chunk_index += 1
                        
            while tag_stack:
                tag = tag_stack.pop()
                lines.append(f'</{tag}>')
            if in_div:
                lines.append('</div>')
            lines.append('</body></html>')
            
            reconstructed = '\n'.join([line for line in lines if line])
            for pattern, replacement in CLEANUP_REGEXES:
                reconstructed = pattern.sub(replacement, reconstructed)
            return reconstructed
        except Exception as e:
            self.logger.error(f"HTML reconstruction failed: {e}, returning original HTML")
            return original_content

    def _load_chunk_from_checkpoint(self, checkpoint_key_prefix: str, item_id: str, chunk_index: int) -> Optional[str]:
        chunk_key = f"chunk{chunk_index}"
        chunk_data = self.checkpoint_manager.get_chunk_data(checkpoint_key_prefix, item_id, chunk_key)
        
        if chunk_data and chunk_data.get('completed_chunk', False):
            translated_lines =[]
            line_index = 0
            while f"line{line_index}" in chunk_data['lines']:
                try:
                    line_data = chunk_data['lines'][f"line{line_index}"]
                    # Handle legacy stringified JSON data gracefully
                    if isinstance(line_data, str):
                        line_data = json.loads(line_data)
                    translated_lines.append(line_data['text'])
                except (json.JSONDecodeError, TypeError, KeyError):
                    self.logger.warning(f"Invalid checkpoint data for {chunk_key}_line{line_index}. Skipping.")
                    break
                line_index += 1
            return '\n'.join(translated_lines)
        return None

    def _translate_and_save_chunk(self, translator, chunk_text: str, source_lang: str, target_lang: str, is_japanese_webnovel: bool, checkpoint_key_prefix: str, item_id: str, chunk_key: str, chunk_index: int, item_name: str, total_chunks: int) -> str:
        original_chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
        checkpointed_chunk_data = self.checkpoint_manager.get_chunk_data(checkpoint_key_prefix, item_id, chunk_key)
        
        if checkpointed_chunk_data and checkpointed_chunk_data.get('completed_chunk', False):
            if checkpointed_chunk_data.get('original_hash') == original_chunk_hash:
                self.logger.info(f"Loading translated chunk {chunk_index + 1}/{total_chunks} of {item_name}.")
                loaded_text = self._load_chunk_from_checkpoint(checkpoint_key_prefix, item_id, chunk_index)
                if loaded_text:
                    self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
                    return loaded_text
            else:
                self.logger.warning(f"Hash mismatch for chunk {chunk_index + 1} of {item_name}. Re-translating.")

        self.logger.info(f"Translating chunk {chunk_index + 1}/{total_chunks} of {item_name}.")
        try:
            translated_text = translator.translate_text(chunk_text, source_lang, target_lang, is_japanese_webnovel)
            tokens_used, requests_made = 0, 0
            
            for mt, info in translator.get_model_status().items():
                if info['last_requests_made_per_request'] > 0:
                    tokens_used = info['last_tokens_used_per_request']
                    requests_made = info['last_requests_made_per_request']
                    break
                    
            if isinstance(translated_text, str) and translated_text.strip():
                if re.sub(r'__ENTITY_\d+__', '', chunk_text).strip() == re.sub(r'__ENTITY_\d+__', '', translated_text).strip():
                    self.logger.warning(f"Translation identical to original for chunk {chunk_index + 1}, using original.")
                    translated_text = chunk_text
                    
                raw_lines =[line.strip() for line in translated_text.split('\n') if line.strip()]
                for line_index, line in enumerate(raw_lines):
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_key_prefix, item_id, chunk_key, 
                        translated_line=line, line_index=line_index, 
                        is_dialogue=bool(re.match(r'^[\'\"“”‘’].+[\'\"“”‘’]\s*$', line)), 
                        tokens_used=tokens_used, requests_made=requests_made, 
                        original_chunk_hash=original_chunk_hash
                    )
                self.checkpoint_manager.save_checkpoint(checkpoint_key_prefix, item_id, chunk_key, completed=True)
                self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
                return translated_text
        except ProhibitedContentError as pce:
            self.logger.error(f"Prohibited content detected for chunk {chunk_index + 1}: {pce}")
        except Exception as e:
            self.logger.error(f"Translation failed for chunk {chunk_index + 1}: {e}. Using original.")

        self.checkpoint_manager.save_checkpoint(checkpoint_key_prefix, item_id, chunk_key, original_chunk_hash=original_chunk_hash, completed=True)
        self.translated_chunks_history.setdefault(item_id, {})[chunk_key] = 'translated'
        return chunk_text

    def translate_epub(self, translator, source_lang: str, target_lang: str, output_path: str, progress_callback: Optional[Callable] = None, status_callback: Optional[Callable] = None, stop_flag: Optional[Callable] = None, is_japanese_webnovel=False, start_chapter: int = 1, end_chapter: Optional[int] = None) -> bool:
        try:
            out_path_obj = Path(output_path)
            temp_output_path = out_path_obj.with_suffix('.temp.epub')
            self.target_lang = target_lang
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            use_temp_epub = False
            if temp_output_path.exists():
                try:
                    temp_book = epub.read_epub(str(temp_output_path))
                    for checkpoint_key_prefix, item_data in self.checkpoint_manager.checkpoint_data.items():
                        for item_id, item_chk_data in item_data.items():
                            if item_chk_data.get('completed', False):
                                use_temp_epub = True
                                break
                        if use_temp_epub: break
                        
                    if use_temp_epub:
                        self.book = temp_book
                        self.logger.info(f"Resuming from existing temp EPUB: {temp_output_path}")
                    else:
                        self.logger.info("Temp EPUB exists but no completed translations found, starting from original EPUB")
                except Exception as e:
                    self.logger.warning(f"Failed to load temp EPUB: {e}. Falling back to original EPUB.")
                    
            if not use_temp_epub:
                if not self.book:
                    raise ValueError("No EPUB loaded.")
                self.logger.info("Starting from original EPUB.")
                
            content_chapters_to_process =[]
            for chap_idx, item in self.content_chapters:
                if start_chapter <= chap_idx and (end_chapter is None or chap_idx <= end_chapter):
                    content_chapters_to_process.append((chap_idx, item))
                    
            if not content_chapters_to_process:
                self.logger.warning(f"No chapters found in the specified range {start_chapter}-{end_chapter}.")
                return True
                
            if status_callback:
                status_callback("Starting translation with multi-model approach...")
                
            total_selected_chapters = len(content_chapters_to_process)
            processed_chapters_count = 0

            for chapter_counter, item in content_chapters_to_process:
                item_id = item.get_id() or f"item_{chapter_counter}"
                checkpoint_key_prefix = f"{out_path_obj.stem}_{item_id}"
                checkpointed_chunk_data = self.checkpoint_manager.checkpoint_data.get(checkpoint_key_prefix, {}).get(item_id, {})
                
                if checkpointed_chunk_data.get('completed', False):
                    self.logger.info(f"Applying checkpointed translation for chapter {chapter_counter}/{self.total_chapters}: {item_id}")
                    content = item.get_content().decode('utf-8')
                    cleaned_text, structure_info = self.extract_text_with_structure(content)
                    chunks = self.intelligent_chunk_text(cleaned_text, structure_info)
                    translated_chunks = [None] * len(chunks)
                    
                    for j, (chunk_text, chunk_structure) in enumerate(chunks):
                        loaded_text = self._load_chunk_from_checkpoint(checkpoint_key_prefix, item_id, j)
                        if loaded_text is not None:
                            translated_chunks[j] = loaded_text
                            
                    new_content = self.reconstruct_html_with_structure(content, translated_chunks, structure_info, stop_flag)
                    item.set_content(new_content.encode('utf-8'))
                    self._save_intermediate_epub(str(temp_output_path))

            for chapter_counter, item in content_chapters_to_process:
                if stop_flag and stop_flag():
                    self.logger.info("Translation stopped by user.")
                    return False
                    
                self.current_chapter_index = chapter_counter
                processed_chapters_count += 1
                item_id = item.get_id() or f"item_{chapter_counter}"
                checkpoint_key_prefix = f"{out_path_obj.stem}_{item_id}"
                
                if self.checkpoint_manager.get_item_completion_status(checkpoint_key_prefix, item_id):
                    self.logger.info(f"Skipping already translated chapter {chapter_counter}/{self.total_chapters}: {item_id}.")
                    if progress_callback:
                        progress_callback((processed_chapters_count / total_selected_chapters) * 100)
                    continue
                    
                if status_callback:
                    status_callback(f"Translating Chapter {chapter_counter}/{self.total_chapters}: {item.get_name()} with available models...")
                    
                content = item.get_content().decode('utf-8')
                cleaned_text, structure_info = self.extract_text_with_structure(content)
                
                if not cleaned_text.strip():
                    self.logger.info(f"No translatable text in {item.get_name()}, skipping.")
                    self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                    if progress_callback:
                        progress_callback((processed_chapters_count / total_selected_chapters) * 100)
                    continue
                    
                chunks = self.intelligent_chunk_text(cleaned_text, structure_info)
                if not chunks:
                    self.logger.warning(f"No chunks generated for {item.get_name()}, skipping.")
                    self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                    if progress_callback:
                        progress_callback((processed_chapters_count / total_selected_chapters) * 100)
                    continue
                    
                translated_chunks = [None] * len(chunks)
                for j, (chunk_text, chunk_structure) in enumerate(chunks):
                    translated_chunks[j] = self._translate_and_save_chunk(
                        translator=translator, chunk_text=chunk_text, source_lang=source_lang, target_lang=target_lang,
                        is_japanese_webnovel=is_japanese_webnovel, checkpoint_key_prefix=checkpoint_key_prefix,
                        item_id=item_id, chunk_key=f"chunk{j}", chunk_index=j, item_name=item.get_name(), total_chunks=len(chunks)
                    )
                    
                new_content = self.reconstruct_html_with_structure(content, translated_chunks, structure_info, stop_flag)
                item.set_content(new_content.encode('utf-8'))
                self._save_intermediate_epub(str(temp_output_path))
                self.checkpoint_manager.set_completed(checkpoint_key_prefix, item_id)
                
                if progress_callback:
                    progress_callback((processed_chapters_count / total_selected_chapters) * 100)

            self._save_final_epub(str(out_path_obj))
            temp_output_path.unlink(missing_ok=True)
            self.logger.info("EPUB translation completed successfully with multi-model approach.")
            return True
        except Exception as e:
            self.logger.error(f"EPUB translation failed: {e}")
            return False

    def _save_intermediate_epub(self, filepath: str):
        try:
            epub.write_epub(filepath, self.book)
            self.logger.debug(f"Intermediate EPUB saved: {filepath}")
        except TypeError as e:
            if "'Link' object is not iterable" in str(e):
                self.logger.warning("Malformed TOC detected during intermediate save. Flattening TOC to force save...")
                try:
                    # Extract flat list of links
                    flat_links = []
                    for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                        title = item.title if getattr(item, 'title', None) else item.get_name()
                        flat_links.append(epub.Link(item.get_name(), title, item.get_name()))
                    
                    # Overwrite the broken TOC with the flat one
                    self.book.toc = flat_links
                    
                    # Retry save
                    epub.write_epub(filepath, self.book)
                    self.logger.debug(f"Intermediate EPUB saved after TOC flattening: {filepath}")
                except Exception as fallback_e:
                    self.logger.warning(f"Failed to save intermediate EPUB even after flattening TOC: {fallback_e}")
            else:
                self.logger.warning(f"Failed to save intermediate EPUB (TypeError): {e}")
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate EPUB: {e}")

    def _save_final_epub(self, filepath: str):
        try:
            epub.write_epub(filepath, self.book)
            self.logger.info(f"Final EPUB saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save final EPUB: {e}")
            raise

    def cleanup(self):
        self.book = None
        self.total_chapters = 0
        self.current_chapter_index = 0
        self.translated_chunks_history.clear()
        self.used_translated_chunks.clear()

    def get_translation_progress(self) -> Tuple[int, int]:
        return self.current_chapter_index, self.total_chapters

    def get_total_chapters(self) -> int:
        return self.total_chapters