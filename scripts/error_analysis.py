r"""Bucket the errors of every scored system, with counts and real examples. No LLM calls.

Buckets (first match wins):
  invalid_label     output is not one of the 77 labels (prompted/fine-tuned systems only)
  same_topic        true and predicted intents share a topic (card, top-up, transfer, identity, ...)
  different_topic   anything else
Writes results/error_buckets.json.   .venv\Scripts\python scripts\error_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.evalcache import cache_file, read_cache  # noqa: E402
from src.systems import system_model  # noqa: E402

# Topic keywords, checked in order; a label's topic is the first keyword it contains.
TOPICS = ["top_up", "topping_up", "transfer", "cash", "withdrawal", "exchange", "currency", "identity", "verify",
          "pin", "passcode", "card", "refund", "payment", "charge", "fee", "balance", "account", "apple_pay",
          "virtual", "disposable", "contactless", "atm", "beneficiary", "direct_debit"]
CANON = {"topping_up": "top_up", "withdrawal": "cash", "currency": "exchange", "verify": "identity",
         "passcode": "pin", "fee": "charge"}


def topic(label: str) -> str:
    for k in TOPICS:
        if k in label:
            return CANON.get(k, k)
    return label


def bucket(true: str, pred, labels: set) -> str:
    if pred not in labels:
        return "invalid_label"
    return "same_topic" if topic(true) == topic(pred) else "different_topic"


def analyse(df: pd.DataFrame, labels: set) -> dict:
    err = df[df["label"] != df["pred"]].copy()
    err["bucket"] = [bucket(t, p, labels) for t, p in zip(err["label"], err["pred"])]
    out = {"n": len(df), "errors": len(err), "buckets": {}}
    for b, g in err.groupby("bucket"):
        out["buckets"][b] = {"count": len(g), "share_of_errors": len(g) / max(1, len(err)),
                             "examples": g.head(4)[["text", "label", "pred"]].to_dict("records")}
    return out


def main():
    labels = set(load_labels())
    res = {}
    for split in ("test", "test_subset"):
        p = path("results") / "predictions" / f"tfidf_{split}.csv"
        if p.exists():
            res[f"tfidf_logreg|{split}"] = analyse(pd.read_csv(p), labels)
    for key in load_config()["llm"]["systems"]:
        for split in ("test", "test_subset"):
            gold = load_split(split)
            cache = read_cache(cache_file(key, system_model(key), split))
            if len(cache) < len(gold) or not all(i in cache for i in gold["orig_idx"]):
                continue
            df = pd.DataFrame([{"text": cache[i]["text"], "label": cache[i]["label"], "pred": cache[i]["prediction"]}
                               for i in gold["orig_idx"]])
            res[f"{key}|{split}"] = analyse(df, labels)
    (path("results") / "error_buckets.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    for k, v in res.items():
        print(k, v["errors"], {b: x["count"] for b, x in v["buckets"].items()})


if __name__ == "__main__":
    main()
