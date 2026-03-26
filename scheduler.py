import time
import logging


class AdaptiveTranslationScheduler:
    """
    Advanced quota-aware translation scheduler.

    Design philosophy:
    - Burst-safe free-tier utilization
    - Token pressure smoothing
    - Entropy-aware chunk dispatch
    - Multi-model fallback selection
    """

    def __init__(self, translator):
        self.translator = translator
        self.logger = logging.getLogger(__name__)

        self.BURST_WINDOW = 60
        self.SAFE_RPM_MARGIN = 1
        self.SAFE_TPM_MARGIN = 5000

        self.last_request_time = 0

    def estimate_chunk_entropy(self, text: str) -> float:
        """Simple heuristic entropy proxy."""
        if not text:
            return 0

        unique_chars = len(set(text))
        return unique_chars / max(len(text), 1)

    def should_delay(self, limits) -> float:
        """
        Returns sleep duration required before next call.
        """

        now = time.time()
        elapsed = now - self.last_request_time

        # Burst protection
        if elapsed < self.BURST_WINDOW / limits.rpm:
            return (self.BURST_WINDOW / limits.rpm) - elapsed

        return 0

    def select_best_model(self, token_estimate: int, text: str):
        """
        Model selection policy:

        Priority order:
        1. Low latency model if pressure is low
        2. Otherwise fallback to cheapest safe model
        """

        entropy = self.estimate_chunk_entropy(text)

        candidates = sorted(
            self.translator.models.items(),
            key=lambda x: x[1]["priority"]
        )

        for model_type, info in candidates:
            limits = info["limits"]

            if not info["available"]:
                continue

            if token_estimate > limits.max_input_tokens:
                continue

            # entropy heuristic filter
            if entropy < 0.2 and limits.rpm < 5:
                continue

            if limits.requests_made < limits.rpm and \
               limits.tokens_used + token_estimate < limits.tpm - self.SAFE_TPM_MARGIN:

                return model_type, info["model_instance"], limits

        return None

    def schedule(self, prompt_text: str, prompt_tokens: int):
        """Main scheduling entry."""

        model_choice = self.select_best_model(prompt_tokens, prompt_text)

        if not model_choice:
            self.logger.warning("Scheduler could not find suitable model.")
            return None

        model_type, model, limits = model_choice

        delay = self.should_delay(limits)
        if delay > 0:
            self.logger.debug(f"Scheduler burst guard sleep {delay:.2f}s")
            time.sleep(delay)

        self.last_request_time = time.time()

        return model_type, model, limits