"""Paired significance tests on the full test set (exact McNemar), from the per-item caches.

Usage:  .venv\\Scripts\\python scripts\\significance.py
Writes: results/significance.json
"""
from __future__ import annotations

import csv
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

PAIRS = [("finetuned_1p5b", "tfidf_logreg"), ("finetuned_1p5b", "few_shot_7b"), ("few_shot_7b", "tfidf_logreg")]
# 2026-10 upgrade: extra pairs, computed only when both systems have full-test results (else skipped).
EXTRA_PAIRS = [("distilbert_pt", "tfidf_logreg"), ("finetuned_1p5b", "distilbert_pt"),
               ("svm_tfidf", "tfidf_logreg"), ("svm_char", "tfidf_logreg"), ("spacy_lemma_svm", "svm_tfidf")]
CLASSIC = set(load_config().get("classic", {}).get("systems", {}))
DISTILBERT = load_config().get("distilbert", {})


def correct_by_item(system: str) -> dict[int, bool]:
    """orig_idx -> prediction correct, on the full test split."""
    if system == "tfidf_logreg" or system in CLASSIC:
        name = "tfidf" if system == "tfidf_logreg" else system
        with open(ROOT / "results" / "predictions" / f"{name}_test.csv", encoding="utf-8") as f:
            return {int(r["orig_idx"]): r["pred"] == r["label"] for r in csv.DictReader(f)}
    if system == DISTILBERT.get("system_key"):
        model = DISTILBERT["base_model"]
    else:
        model = load_config()["llm"]["systems"][system]["model"].replace(":", "-")
    path = ROOT / "data" / "cache" / f"{system}__{model}__test.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {int(r["orig_idx"]): r["prediction"] == r["label"] for r in rows}


def exact_mcnemar(a: dict[int, bool], b: dict[int, bool]) -> dict:
    keys = sorted(set(a) & set(b))
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    n = only_a + only_b
    tail = sum(comb(n, j) for j in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return {"n_items": len(keys), "only_first_correct": only_a, "only_second_correct": only_b,
            "p_value_exact_two_sided": min(1.0, 2 * tail)}


def main() -> None:
    out = {}
    for first, second in PAIRS:
        res = exact_mcnemar(correct_by_item(first), correct_by_item(second))
        out[f"{first} vs {second}"] = res
        print(f"{first} vs {second}: {res}")
    for first, second in EXTRA_PAIRS:
        try:
            a, b = correct_by_item(first), correct_by_item(second)
        except (FileNotFoundError, KeyError):
            print(f"{first} vs {second}: skipped (not run yet)")
            continue
        if len(set(a) & set(b)) < 3080:
            print(f"{first} vs {second}: skipped (full test incomplete)")
            continue
        res = exact_mcnemar(a, b)
        out[f"{first} vs {second}"] = res
        print(f"{first} vs {second}: {res}")
    (ROOT / "results" / "significance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote results/significance.json")


if __name__ == "__main__":
    main()
