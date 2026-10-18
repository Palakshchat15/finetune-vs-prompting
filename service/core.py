"""Framework-free core of the FastAPI service: settings from environment variables, background model
loading (so /health answers at once and /ready flips when the model is in memory), input validation and
request counters. Kept free of FastAPI and torch so it is unit-testable with a fake classifier."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_dir: str = "models/distilbert"
    max_len: int = 64
    torch_threads: int = 1
    max_text_chars: int = 1000
    max_top_k: int = 10

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        e = os.environ if env is None else env
        return cls(model_dir=e.get("MODEL_DIR", cls.model_dir),
                   max_len=int(e.get("MAX_LEN", cls.max_len)),
                   torch_threads=int(e.get("TORCH_NUM_THREADS", cls.torch_threads)),
                   max_text_chars=int(e.get("MAX_TEXT_CHARS", cls.max_text_chars)),
                   max_top_k=int(e.get("MAX_TOP_K", cls.max_top_k)))


def default_loader(s: Settings):
    from src.distilbert import DistilBertIntentClassifier  # noqa: PLC0415  (torch import is slow; done in a thread)
    return DistilBertIntentClassifier(s.model_dir, device="cpu", max_len=s.max_len, num_threads=s.torch_threads)


class ModelHolder:
    """Holds the classifier. state: 'loading' -> 'ready' | 'failed'."""

    def __init__(self, settings: Settings, loader=default_loader):
        self.settings, self._loader = settings, loader
        self.model, self.error, self.state = None, None, "loading"
        self.load_seconds = None
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        t0 = time.perf_counter()
        try:
            self.model = self._loader(self.settings)
            self.state = "ready"
        except Exception as e:  # noqa: BLE001  (reported by /ready; the pod never becomes ready)
            self.error, self.state = f"{type(e).__name__}: {e}", "failed"
        self.load_seconds = time.perf_counter() - t0

    def load_in_background(self) -> threading.Thread:
        t = threading.Thread(target=self.load, name="model-loader", daemon=True)
        t.start()
        return t

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def validate(self, text: str, top_k: int) -> str | None:
        """Return an error message for bad input, else None."""
        if not isinstance(text, str) or not text.strip():
            return "text must be a non-empty string"
        if len(text) > self.settings.max_text_chars:
            return f"text longer than {self.settings.max_text_chars} characters"
        if not 1 <= int(top_k) <= self.settings.max_top_k:
            return f"top_k must be between 1 and {self.settings.max_top_k}"
        return None

    def predict(self, text: str, top_k: int = 3) -> dict:
        if not self.ready:
            raise RuntimeError(f"model not ready ({self.state})")
        t0 = time.perf_counter()
        with self._lock:  # one forward pass at a time per pod; scale out with replicas (HPA), not threads
            self.requests += 1
            try:
                r = self.model.predict([text.strip()], top_k=top_k)[0]
            except Exception:
                self.errors += 1
                raise
        return {"label": r["label"], "confidence": r["confidence"], "top_k": r["top_k"],
                "latency_ms": (time.perf_counter() - t0) * 1000}

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "load_seconds": self.load_seconds,
                "uptime_s": time.time() - self.started, "requests": self.requests, "errors": self.errors,
                "model_dir": self.settings.model_dir}
