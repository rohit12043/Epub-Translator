from dataclasses import dataclass, field
import re
import html as html_module
from typing import List, Tuple


@dataclass
class WhitespaceInfo:
    """Stores leading/trailing whitespace and line-break positions."""
    leading_spaces:   str = ""
    trailing_spaces:  str = ""
    line_breaks:      List[int] = field(default_factory=list)
    paragraph_breaks: List[int] = field(default_factory=list)

def parse_line(line: str) -> str:
    """
    Fallback for block-level reconstruction
    Wrap a single translated line in an appropriate HTML tag.
    """
    line = line.strip()
    if not line:
        return ''

    if line == '---':
        return '<hr/>'

    # Chapter / part headings
    if re.match(r'^(?:Chapter|챕터|章|Part|Volume|卷|편)\s*\d+', line, re.IGNORECASE):
        return f'<p><strong>{html_module.escape(line)}</strong></p>'

    # Escape HTML entities and wrap in paragraph
    escaped = html_module.escape(line)
    # Restore any [Tn] tokens that may still be present
    escaped = re.sub(r'\[T(\d+)\]', '', escaped)  # strip stray tokens
    return f'<p>{escaped}</p>'


def extract_whitespace_info(text: str) -> Tuple[str, WhitespaceInfo]:
    """
    Strip and record leading/trailing whitespace and newline positions so they
    can be faithfully restored after translation.
    """
    info = WhitespaceInfo()

    leading_m = re.match(r'^(\s*)', text)
    if leading_m:
        info.leading_spaces = leading_m.group(1)

    trailing_m = re.search(r'(\s*)$', text)
    if trailing_m:
        info.trailing_spaces = trailing_m.group(1)

    for m in re.finditer(r'\n', text):
        info.line_breaks.append(m.start())

    for m in re.finditer(r'\n\s*\n', text):
        info.paragraph_breaks.append(m.start())

    cleaned = text.strip()
    return cleaned, info


def reconstruct_whitespace(translated_text: str, whitespace_info: WhitespaceInfo) -> str:
    """
    Reattach original leading/trailing whitespace to a translated block.
    Preserves internal paragraph breaks from the translated text itself.
    """
    if not isinstance(translated_text, str):
        translated_text = ""

    # Preserve paragraph breaks already in the translated text
    paragraphs = re.split(r'\n\s*\n', translated_text)
    body = '\n\n'.join(p.strip() for p in paragraphs if p.strip())

    return whitespace_info.leading_spaces + body + whitespace_info.trailing_spaces