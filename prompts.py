from typing import Dict, Tuple


def create_ner_prompt(text: str, source_lang: str, target_lang: str,
                      is_japanese_webnovel: bool = False) -> str:
    """
    Compact NER prompt for PERSON / ORGANIZATION entity extraction.
    """
    same_script_rule = (
        f"If {source_lang} and {target_lang} share the same alphabet/script, "
        f"return the exact original text unchanged in 'translation'. "
        f"Do NOT abbreviate, drop parts of names, or use initials."
    )

    jpn_note = ""
    if is_japanese_webnovel and source_lang.lower() in ('chinese', 'zh', 'cn'):
        jpn_note = (
            "\n- For Japanese names written in Chinese characters, provide the "
            "Japanese reading (e.g. '日葵' → 'Himawari'), not the Chinese reading."
        )

    return f"""Extract named entities from the {source_lang} text below and give their {target_lang} equivalents.

Focus ONLY on:
- PERSON: specific individuals / fictional characters (full name, exact text).
- ORGANIZATION: companies, schools, teams, institutions, brands.

Rules:
1. Extract the EXACT and COMPLETE entity text as it appears.
2. PERSON: use standard transliteration when changing scripts. {same_script_rule}
3. ORGANIZATION: use the official {target_lang} name if widely known; otherwise translate naturally.
4. Exclude common nouns, locations, generic terms (unless part of an org name).{jpn_note}

Return a JSON array only. Each object must have exactly:
  {{"entity": "<original text>", "type": "PERSON" | "ORGANIZATION", "translation": "<{target_lang} equivalent>"}}

Text:
{text}"""


def create_translation_prompt(text: str, source_lang: str, target_lang: str,
                               active_placeholders: Dict[str, Tuple[str, str]],
                               is_continuation: bool = False) -> str:
    """
    Compact translation prompt.
    """
    # ── Glossary placeholder section (only shown when placeholders are active) ─
    glossary_section = ""
    if active_placeholders:
        ph_list = ", ".join(sorted(active_placeholders.keys()))
        glossary_section = (
            f"\nGLOSSARY PLACEHOLDERS — keep these VERBATIM: {ph_list}\n"
            f"Do NOT translate, alter, or remove them.\n"
        )

    continuation_note = " Continue in the same style as established." if is_continuation else ""

    return f"""Translate the following {source_lang} text to {target_lang}.{continuation_note}

STRICT RULES — follow all of them:
1. Return ONLY the translated text. No explanations, no preamble.
2. [Tn] TOKENS (e.g. [T0], [T1], [T2]…): These mark inline HTML formatting \
(bold, italic, ruby, etc.). Preserve every [Tn] token EXACTLY in the position \
relative to the surrounding words — do NOT drop, move, or alter any token.
3. PARAGRAPH BREAKS: Preserve all double-newlines (\\n\\n) that separate \
paragraphs — they mark block boundaries and must not be removed or collapsed.
4. NATURAL TRANSLATION: Produce fluent, idiomatic {target_lang}. \
Adapt expressions; do not translate word-for-word. \
Preserve dialogue tone, character voice, hesitations, and emotional nuance.
5. NAMES & PROPER NOUNS: Keep unchanged unless a glossary placeholder replaces them.
{glossary_section}
--- TEXT TO TRANSLATE ---
{text}"""