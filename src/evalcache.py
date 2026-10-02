"""Per-item JSONL caches for the LLM evaluations (resume after interruption)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import path


def cache_file(system: str, model: str, split: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return path("cache") / f"{system}__{safe}__{split}.jsonl"


def read_cache(p: Path) -> dict[int, dict]:
    """orig_idx -> record. Tolerates a truncated last line (killed mid-write)."""
    out = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(rec["orig_idx"])] = rec
    return out


def append(p: Path, rec: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
