from dataclasses import dataclass, field
import re
import html
from typing import List, Tuple

@dataclass
class WhitespaceInfo:
    """Stores detailed whitespace information for text reconstruction."""
    leading_spaces: str
    trailing_spaces: str
    line_breaks: List[int] = field(default_factory=list)  # Positions of \n
    paragraph_breaks: List[int] = field(default_factory=list)  # Positions of \n\n


def parse_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ''
    if line == '---':
        return '<hr/>'
    if re.match(r'^(Chapter|챕터|章)\s*\d+(\s*[:-]?\s*.+)?$', line, re.IGNORECASE):
        return f'<p><strong>{html.escape(line)}</strong></p>'
    is_dialogue = bool(re.match(r'^(?:[\'\"“”‘’])?.+[\'\"“”‘’]\s*$', line))
    if is_dialogue:
        line = html.escape(line)
        line = re.sub(r'^[\'"]', '“', line)
        line = re.sub(r'[\'"]([.!?…]?\s*)$', '”\1', line)
        return f'<p><em>{line}</em></p>'
    line = html.escape(line)
    line = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', line)
    line = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', line)
    return f'<p>{line}</p>'


def extract_whitespace_info(text: str) -> Tuple[str, WhitespaceInfo]:
    whitespace_info = WhitespaceInfo(
        leading_spaces="",
        trailing_spaces="",
        line_breaks=[],
        paragraph_breaks=[]
    )
    
    leading_match = re.match(r'^(\s*)', text)
    if leading_match:
        whitespace_info.leading_spaces = leading_match.group(1)
        
    trailing_match = re.search(r'(\s*)$', text)
    if trailing_match:
        whitespace_info.trailing_spaces = trailing_match.group(1)
        
    for match in re.finditer(r'\n', text):
        whitespace_info.line_breaks.append(match.start())
        
    for match in re.finditer(r'\n\s*\n', text):
        whitespace_info.paragraph_breaks.append(match.start())
        
    cleaned_text = text.strip()
    return cleaned_text, whitespace_info


def reconstruct_whitespace(translated_text: str, whitespace_info: WhitespaceInfo) -> str:
    """Reconstructs whitespace, including paragraph breaks, for a translated chunk."""
    if not isinstance(translated_text, str):
        translated_text = ""
        
    # Preserve paragraph breaks from translated text
    paragraphs = re.split(r'\n\s*\n', translated_text)
    result = '\n\n'.join(p.strip() for p in paragraphs if p.strip())
    
    # Safely restore only the leading and trailing spaces
    result = whitespace_info.leading_spaces + result + whitespace_info.trailing_spaces
    
    return result