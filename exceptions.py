# ===== FILE: exceptions.py =====
class EPUBTranslatorError(Exception):
    """Base exception for the application."""
    pass

class ProhibitedContentError(EPUBTranslatorError):
    """Raised when the API blocks content for safety reasons."""
    pass

class TranslationFailedError(EPUBTranslatorError):
    """Raised when translation fails after all retries."""
    pass