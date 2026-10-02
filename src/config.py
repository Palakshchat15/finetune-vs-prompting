"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = ROOT / load_config()["paths"][key]
    if key != "traces_db":
        p.mkdir(parents=True, exist_ok=True)
    return p


def detect_device() -> str:
    """'cuda' if torch sees a GPU, else 'mps' on Apple silicon, else 'cpu'. torch is optional."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
