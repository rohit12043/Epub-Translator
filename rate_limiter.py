import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enum representing supported Gemini model types."""
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_FLASH = "gemini-3-flash"
    GEMINI_3_1_PRO = "gemini-3.1-pro"

@dataclass
class ModelLimits:
    """Tracks model rate limits (configured) and usage (runtime)."""
    tpm: int
    tpd: int
    rpd: int
    rpm: int
    max_chapters_per_day: int
    tokens_per_chapter: int
    max_input_tokens: int
    requests_made: int = field(default=0)
    tokens_used: int = field(default=0)
    daily_requests: int = field(default=0)
    daily_tokens: int = field(default=0)
    last_reset_time: float = field(default_factory=time.time)
    last_daily_reset: float = field(default_factory=time.time)
    last_requests_made_per_request: int = field(default=0)
    last_tokens_used_per_request: int = field(default=0)

    def to_dict(self) -> Dict:
        """Converts ModelLimits to a dictionary for saving."""
        return {
            "tpm": self.tpm,
            "tpd": self.tpd,
            "rpd": self.rpd,
            "rpm": self.rpm,
            "max_chapters_per_day": self.max_chapters_per_day,
            "tokens_per_chapter": self.tokens_per_chapter,
            "max_input_tokens": self.max_input_tokens,
            "requests_made": self.requests_made,
            "tokens_used": self.tokens_used,
            "daily_requests": self.daily_requests,
            "daily_tokens": self.daily_tokens,
            "last_reset_time": self.last_reset_time,
            "last_daily_reset": self.last_daily_reset,
            "last_requests_made_per_request": self.last_requests_made_per_request,
            "last_tokens_used_per_request": self.last_tokens_used_per_request,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelLimits':
        """Creates ModelLimits from a dictionary, handling missing keys and types."""
        try:
            tpm = max(0, int(data.get("tpm", 250000)))
            tpd = max(0, int(data.get("tpd", 1000000)))
            rpd = max(0, int(data.get("rpd", 500)))
            rpm = max(0, int(data.get("rpm", 15)))
            max_chapters_per_day = max(0, int(data.get("max_chapters_per_day", 78)))
            tokens_per_chapter = max(1, int(data.get("tokens_per_chapter", 3200)))
            max_input_tokens = max(1000, int(data.get("max_input_tokens", 30000)))
            limits = cls(
                tpm=tpm,
                tpd=tpd,
                rpd=rpd,
                rpm=rpm,
                max_chapters_per_day=max_chapters_per_day,
                tokens_per_chapter=tokens_per_chapter,
                max_input_tokens=max_input_tokens
            )
            limits.requests_made = max(0, int(data.get("requests_made", 0)))
            limits.tokens_used = max(0, int(data.get("tokens_used", 0)))
            limits.daily_requests = max(0, int(data.get("daily_requests", 0)))
            limits.daily_tokens = max(0, int(data.get("daily_tokens", 0)))
            limits.last_requests_made_per_request = max(0, int(data.get("last_requests_made_per_request", 0)))
            limits.last_tokens_used_per_request = max(0, int(data.get("last_tokens_used_per_request", 0)))
            last_reset_time = data.get("last_reset_time", time.time())
            limits.last_reset_time = last_reset_time if isinstance(last_reset_time, (int, float)) and last_reset_time >= 0 else time.time()
            last_daily_reset = data.get("last_daily_reset", time.time())
            limits.last_daily_reset = last_daily_reset if isinstance(last_daily_reset, (int, float)) and last_daily_reset >= 0 else time.time()
            return limits
        except Exception as e:
            logger.error(f"Failed to load ModelLimits from history data: {e}. Data: {data}. Using default limits.")
            return cls(
                tpm=data.get("tpm", 250000),
                tpd=data.get("tpd", 1000000),
                rpd=data.get("rpd", 500),
                rpm=data.get("rpm", 15),
                max_chapters_per_day=data.get("max_chapters_per_day", 78),
                tokens_per_chapter=data.get("tokens_per_chapter", 3200),
                max_input_tokens=data.get("max_input_tokens", 30000)
            )
            