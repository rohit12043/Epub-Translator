from typing import Dict, Tuple

def create_ner_prompt(text: str, source_lang: str) -> str:
    """ Creates prompt for named entity recognition, focusing on PERSON and ORGANIZATION. """
    return f"""Identify specific named entities in the following {source_lang} text. Focus exclusively on:
- **PERSON** names: Names of specific individuals or fictional characters.
- **ORGANIZATION** names: Names of companies, institutions, schools, teams, brands, etc.

**Strict Rules for Identification:**
- Extract the **exact text** of the entity as it appears in the input text.
- **Only PERSON and ORGANIZATION entities should be identified for transliteration**.
- **Exclude** common words, generic nouns, adjectives, or verbs.
- **Exclude** simple geographic locations unless part of a specific organization name.
- **Exclude** cultural terms, plant names, or other proper nouns that are not PERSON or ORGANIZATION.
- Avoid including entities that are only common terms.
- Prioritize clear proper nouns over potential ambiguous terms.

**Translation and Transliteration Instructions:**
- For identified PERSON and ORGANIZATION entities, the entity text will be used for transliteration.
- For all other proper nouns, **do not include them in the output** as they should be fully translated into natural English in the main translation process.

**Output Format:**
- Return a **JSON list** of objects.
- Each object must have the following two keys:
  - `"entity"` (string): The exact text of the entity found in the input text.
  - `"type"` (string): Must be one of the following exact strings: `"PERSON"` or `"ORGANIZATION"`.
- Output **ONLY valid JSON**. Do NOT include any explanations, descriptions, introductory phrases, or extra text before or after the JSON code block.
- Ensure the JSON is correctly formatted.
- If no entities matching the criteria are found, return an empty JSON list `[]`.

**TEXT:**
{text}
"""

def create_equivalent_translation_prompt(entity: str, entity_type: str, source_lang: str, target_lang: str, is_japanese_webnovel: bool = False) -> str:
    """Creates prompt for translating/transliterating a specific named entity."""
    if is_japanese_webnovel and source_lang.lower() in['chinese', 'zh', 'cn']:
        instructions = (
            f"This is a Japanese webnovel translated to Chinese. For PERSON names: "
            f"Provide the original Japanese reading (romaji) of the name, not the Chinese reading. "
            f"Example: '日葵' should be 'Himawari' (Japanese) not 'Rìkuí' (Chinese). "
            f"For ORGANIZATION names: Use the original Japanese name if known, "
            f"otherwise provide a natural transliteration."
        )
    elif entity_type == "PERSON":
        instructions = (
            f"For PERSON names: Provide the standard transliteration or most common recognized {target_lang} equivalent. "
            f"Focus on matching {target_lang} pronunciation conventions. Do NOT translate the literal meaning of the name."
        )
    elif entity_type == "ORGANIZATION":
        instructions = (
            f"For ORGANIZATION names: Provide the official or most commonly recognized {target_lang} name if it is widely known. "
            f"If no standard name exists, provide a natural-sounding transliteration or an appropriate, contextually relevant translation."
        )
    else:
        instructions = "Provide the most appropriate equivalent in the target language."

    return f"""Provide the best equivalent in {target_lang} for the following {source_lang} named entity.
**ENTITY:** '{entity}'
**TYPE:** {entity_type}
**Source Language:** {source_lang}
**Target Language:** {target_lang}

**Strict Rules for Output:**
1. {instructions}
2. Return **ONLY** the translated, transliterated, or equivalent entity text.
3. If the entity should remain unchanged in {target_lang}, return the original entity text **EXACTLY as provided**.
4. Ensure the output is plain text.

Equivalent in {target_lang}:"""

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