from typing import Dict, Tuple

def create_ner_prompt(text: str, source_lang: str, target_lang: str, is_japanese_webnovel: bool = False) -> str:
    """ Creates a combined prompt for Named Entity Recognition AND Translation in a single pass. """
    
    shared_alphabet_rule = (
        f"**CRITICAL ALPHABET RULE:** If {source_lang} and {target_lang} use the same alphabet/script "
        f"(e.g., German to English, Spanish to French), you MUST return the exact original entity text unchanged "
        f"in the 'translation' field. Do NOT abbreviate names, do NOT use initials, and do NOT drop last names."
    )

    jpn_instruction = ""
    if is_japanese_webnovel and source_lang.lower() in ['chinese', 'zh', 'cn']:
        jpn_instruction = "- **CRITICAL FOR JAPANESE:** Provide the original Japanese reading (romaji) of the name, not the Chinese reading (e.g., '日葵' -> 'Himawari')."

    return f"""Identify specific named entities in the following {source_lang} text and provide their highly accurate {target_lang} equivalent.

Focus exclusively on:
- **PERSON** names: Names of specific individuals or fictional characters.
- **ORGANIZATION** names: Companies, institutions, schools, teams, brands, etc.

**Strict Rules for Extraction and Translation:**
1. Extract the **exact and complete text** of the entity as it appears in the input text (e.g., full first and last name).
2. Provide the absolute best {target_lang} equivalent. Focus on standard transliteration/pronunciation ONLY if changing alphabets (e.g., Korean to English).
3. {shared_alphabet_rule}
4. {jpn_instruction}
5. For ORGANIZATION names, use the official {target_lang} name if widely known, otherwise provide a natural translation. Do not truncate or arbitrarily shorten organization names.
6. **Exclude** common words, generic nouns, or locations unless part of a specific organization name.

**Output Format:**
Return a **JSON list** of objects. Each object must have these exactly three keys:
- `"entity"` (string): The exact original text found in the source.
- `"type"` (string): Must be exactly `"PERSON"` or `"ORGANIZATION"`.
- `"translation"` (string): The translated, transliterated, or preserved equivalent in {target_lang}.

**TEXT:**
{text}
"""

def create_translation_prompt(text: str, source_lang: str, target_lang: str, active_placeholders: Dict[str, Tuple[str, str]], is_continuation: bool = False) -> str:
    """Creates optimized translation prompt with comprehensive guidance, strong placeholder protection, and context awareness for novel settings."""
    placeholder_section = ""
    if active_placeholders:
        placeholders = sorted(active_placeholders.keys())
        placeholder_section = f"""### Placeholder Instructions (DO NOT TRANSLATE THIS SECTION)
* The text contains special placeholders in the format __GLOSSARY_XX__ (e.g., {placeholders[0] if placeholders else '__GLOSSARY_0__'}).
* These placeholders represent pre-translated terms or entities.
* **ABSOLUTE RULES FOR PLACEHOLDERS**:
  - **DO NOT** translate, modify, remove, or reorder these placeholders.
  - Preserve their **exact spelling**, **case**, and **numeric IDs** (e.g., __GLOSSARY_2__ must remain __GLOSSARY_2__).
  - Maintain their **exact positions** relative to the surrounding text.
  - Do not add or remove spaces around placeholders.
  - Treat placeholders as immutable tokens.
* Current placeholders in this text: {", ".join(placeholders)}.
* **WARNING**: Altering placeholders will invalidate the translation.
### End Placeholder Instructions
"""

    if is_continuation:
        return f"""### Translation Task
Continue translating the following text from {source_lang} to {target_lang}, maintaining the same style and rules as previously established:

{placeholder_section}

### Context Reminder
- This is a continuation of a novel set in a school environment (e.g., science club, school festivals).
- Maintain a conversational, youthful tone for young characters in a casual, modern setting.
- Preserve emotional nuances (e.g., hesitations, informal speech) and dialogue style consistent with the novel.

### Output Requirements
- Return **ONLY** the translated text with all placeholders preserved exactly as provided.
- **DO NOT** include this prompt, introductory phrases, explanations, or extra formatting unless present in the original text.
- Preserve original line breaks (\\n) and spacing.
- Ensure the translation is complete without truncation.

### Text to Translate
{text}
"""

    return f"""### Translation Task
Translate the following text from {source_lang} to {target_lang}:

{placeholder_section}

### Context and Tone
- This text is part of a novel, likely set in a school environment (e.g., involving clubs like a science club or school festivals).
- Adopt a conversational, youthful tone suitable for young characters (e.g., students) in a casual, modern setting.
- Capture the emotional nuance, such as hesitation (e.g., "我、我是" should convey a shy or nervous tone like "Um, we're"), and informal speech patterns (e.g., "喔……" as a casual trail-off like "…").
- Ensure the translation feels natural and engaging for readers of a webnovel or light novel.

### Translation Guidelines
1. Produce a natural, fluent translation that reads well in {target_lang}.
2. Accurately capture the original meaning, tone, style, and emotional nuance, reflecting the novel's context and character dynamics.
3. Avoid literal or word-for-word translations when they sound unnatural.
4. Adapt idiomatic expressions to {target_lang} conventions, prioritizing natural dialogue for young characters.
5. **Preserve the original paragraph structure** (use **\\n\\n** for **paragraph breaks**).
6. Be concise; do not add unnecessary words or embellishments unless they enhance the novel's tone.
7. Pay attention to context from the surrounding text to ensure coherence and consistency with the story's setting.
8. Reflect character personality in dialogue (e.g., hesitations, trailing speech, or enthusiasm should be conveyed naturally).

### Terminology Notes
- School events (e.g., '校庆', '文化祭') → Translate as 'cultural festival' or 'school festival' based on context.
- Accessories (e.g., '饰品', 'アクセサリー') → Prefer 'accessories' over 'jewelry' when referring to casual or decorative items (e.g., flower accessories sold by a school club), unless the context explicitly indicates precious metals or gems.
- Names and proper nouns → Preserve unless explicitly replaced by placeholders.
- For cultural or botanical terms (e.g., "枝垂桜", "二輪草"), prepend a concise one-line description in parentheses, e.g., "(Weeping cherry: trees with drooping branches in Japan)".
- Club-related terms (e.g., '科学社', '科学部') → Translate as 'science club' or similar, ensuring it fits a school club context.
- Informal speech markers (e.g., '喔', '～～') → Use natural equivalents like '...', 'um', or 'you know' to reflect casual, youthful dialogue.

### Output Requirements
- Return **ONLY** the translated text with all placeholders preserved exactly as provided.
- **DO NOT** include this prompt, introductory phrases, explanations, or extra formatting unless present in the original text.
- Preserve original line breaks (\\n) and spacing.
- Ensure the translation is complete without truncation.

### Text to Translate
{text}
"""