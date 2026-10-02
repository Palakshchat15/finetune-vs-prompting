"""Classification metrics shared by every system."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def summarize(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """Accuracy and macro-F1 over `labels`. Invalid/None predictions count as wrong."""
    yp = [p if p in set(labels) else "__invalid__" for p in y_pred]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, yp)),
        "macro_f1": float(f1_score(y_true, yp, labels=labels, average="macro", zero_division=0)),
        "invalid_rate": float(np.mean([p == "__invalid__" for p in yp])) if yp else 0.0,
    }


def per_class_f1(y_true, y_pred, labels) -> dict[str, float]:
    yp = [p if p in set(labels) else "__invalid__" for p in y_pred]
    f = f1_score(y_true, yp, labels=labels, average=None, zero_division=0)
    return {lab: float(v) for lab, v in zip(labels, f)}


def top_confusions(y_true, y_pred, k: int = 10) -> list[dict]:
    """Most frequent (true -> predicted) errors."""
    c = Counter((t, p if p is not None else "__invalid__") for t, p in zip(y_true, y_pred) if t != p)
    return [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common(k)]


def latency_stats(seconds: list[float]) -> dict:
    a = np.asarray(seconds, dtype=float)
    if a.size == 0:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None, "throughput_rps": None}
    return {"p50_ms": float(np.percentile(a, 50) * 1000), "p95_ms": float(np.percentile(a, 95) * 1000),
            "mean_ms": float(a.mean() * 1000), "throughput_rps": float(1.0 / a.mean()) if a.mean() > 0 else None}


def cost_per_1k(mean_seconds: float, usd_per_hour: float) -> float:
    """Cost of 1,000 sequential requests at `mean_seconds` each on an instance priced `usd_per_hour`."""
    return mean_seconds * 1000 / 3600 * usd_per_hour
