import re
import logging

logger = logging.getLogger(__name__)

class TranslationValidator:
    def __init__(self):
        self.hallucination_patterns =[
            r"as an AI language model",
            r"additional(?:ly| information)",
            r"I am a large language model, trained by Google",
            r"I cannot provide assistance with that request",
            r"I'm unable to help with that"
        ]
        self.error_patterns =[
            r"I cannot translate",
            r"Error:",
            r"Translation failed",
            r"I am unable to translate content that may violate",
            r"I cannot fulfill this request",
            r"I can't provide a translation for this content",
            r"I do not have access to the internet",
            r"content that violates policy",
            r"this request may violate guidelines"
        ]


    def validate_chunk(self, source_chunk_with_placeholders: str, translated_chunk: str) -> bool:
        """
        Comprehensive validation of a single translated chunk (which may still contain placeholders).
        Does NOT validate glossary terms themselves, only general translation quality.
        Length check failures are treated as warnings and do not cause validation to fail.
        """
        checks = {
            "hallucinations": self.check_hallucinations(translated_chunk),
            "errors": self.check_errors(translated_chunk),
        }

        length_passed = self.check_length(source_chunk_with_placeholders, translated_chunk) # This is a warning, not a failure

        failed_checks = [name for name, passed in checks.items() if not passed]

        if failed_checks:
            logger.warning(f"Chunk validation failed on: {', '.join(failed_checks)}")
            return False

        return True # Return True even if length check warned

    def check_length(self, source: str, translation: str) -> bool:
        """Checks length ratio between source and translation, logging warnings for out-of-range ratios."""
        source_clean = re.sub(r'__GLOSSARY_\d+__', '', source).strip()
        translation_clean = translation.strip()

        if not source_clean:
            return True  # No source text to compare against

        if not translation_clean:
            logger.warning("Length check failed: Empty translation received.")
            return False

        src_len = len(source_clean)
        tgt_len = len(translation_clean)

        min_ratio = 0.05  # Allow for significant compression/expansion, but catch extreme cases
        max_ratio = 8.0

        if src_len == 0:
            # This case should be handled by the `if not source_clean` check above,
            # but as a fallback, avoid division by zero.
            return True

        ratio = tgt_len / src_len

        if not (min_ratio <= ratio <= max_ratio):
            logger.warning(
                f"Length check warning: Source length (cleaned)={src_len}, "
                f"Target length (stripped)={tgt_len}, Ratio={ratio:.2f}. "
                f"Expected ratio between {min_ratio} and {max_ratio}. Translation will proceed."
            )
            return True # Return True because length check is a warning, not a strict failure

        return True

    def check_hallucinations(self, text: str) -> bool:
        """Detects model hallucinations or canned responses using regex patterns."""
        if not isinstance(text, str):
            return True # Cannot check non-strings

        text_lower = text.lower()
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Hallucination pattern found: '{pattern}' in translation.")
                return False
        return True

    def check_errors(self, text: str) -> bool:
        """Detects explicit model errors or refusal messages using regex patterns."""
        if not isinstance(text, str):
            return True # Cannot check non-strings

        text_lower = text.lower()
        for pattern in self.error_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Error pattern found: '{pattern}' in translation.")
                return False
        return True