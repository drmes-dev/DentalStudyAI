"""Per-model backoff shared by chat, verification and paper import."""
import math
import re
import time
from datetime import datetime, timedelta
from threading import Lock
from zoneinfo import ZoneInfo


class AIUnavailable(RuntimeError):
    def __init__(self, result):
        super().__init__(result.get("response", "AI services are temporarily unavailable."))
        self.retry_after = max(1, int(result.get("retry_after", 60)))
        self.reason = result.get("error_code", "ai_unavailable")


class ModelAvailability:
    def __init__(self):
        self._lock = Lock()
        self._blocked = {}

    def remaining(self, provider, model):
        with self._lock:
            until, _ = self._blocked.get((provider, model), (0, ""))
        return max(0, math.ceil(until - time.time()))

    def failed(self, provider, model, exc):
        # Exception text is inspected, never returned or logged: SDK errors can
        # include request bodies, account identifiers and authorization details.
        message = str(exc).lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if not isinstance(status, int):
            status = getattr(getattr(exc, "response", None), "status_code", None)
        rate_limited = status == 429 or "resource_exhausted" in message
        denied = status in (401, 403, 404) or "unpurchased" in message
        seconds, reason = (60, "temporary_failure")
        if denied:
            seconds, reason = 3600, "model_access_unavailable"
        elif rate_limited:
            seconds, reason = 60, "quota_reached"
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            try:
                seconds = max(seconds, math.ceil(float(headers.get("retry-after", 0))))
            except (ValueError, TypeError):
                pass
            # Groq also supplies durations such as "5m12.4s" in its message.
            duration = re.search(r"try again in\s+([\d.hms ]+)", message)
            if duration:
                units = {"h": 3600, "m": 60, "s": 1}
                seconds = max(seconds, math.ceil(sum(float(n) * units[u] for n, u in
                    re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", duration.group(1)))))
            if provider == "Gemini" and ("perday" in message or "per day" in message):
                # Gemini's daily quota does not reset after its short RetryInfo.
                now = datetime.fromtimestamp(time.time(), ZoneInfo("America/Los_Angeles"))
                reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
                seconds = max(seconds, math.ceil(reset.timestamp() - time.time()))
            elif "per day" in message or "tokens/day" in message or "tpd" in message:
                seconds = max(seconds, 900)
        with self._lock:
            self._blocked[(provider, model)] = (time.time() + seconds, reason)
        print(f"AI model paused: {provider}/{model}; reason={reason}; retry_after={seconds}s")

    def unavailable(self, models):
        with self._lock:
            blocks = [self._blocked.get(key, (time.time() + 60, "temporary_failure")) for key in models]
        retry_after = max(1, math.ceil(min(until for until, _ in blocks) - time.time()))
        quota = any(reason == "quota_reached" for _, reason in blocks)
        return {
            "provider": "Error",
            "error_code": "quota_reached" if quota else "ai_unavailable",
            "retry_after": retry_after,
            "response": ("Cloud AI quota is temporarily exhausted or providers are busy. " if quota else
                         "Cloud AI providers are temporarily unavailable. ") +
                        "Please retry shortly. Your saved progress is safe.",
        }
