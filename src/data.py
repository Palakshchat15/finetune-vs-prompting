"""Banking77 download, stratified validation split and stratified test subset.

Outputs (data/splits/):
  train.csv        90% of the original train (used for fitting and few-shot examples)
  val.csv          stratified 10% of the original train (tuning / early stopping only)
  test.csv         the full original test set (3,080), scored once for reported numbers
  test_subset.csv  stratified subset of test, `subset_per_class` per label (for slow LLM systems)
  val_subset.csv   tiny stratified subset of val, for prompt tuning on the LLMs
  labels.json      the 77 label strings, sorted
"""
from __future__ import annotations

import io
import json

import pandas as pd
import requests

from .config import load_config, path

# Neutral User-Agent: never send personal identifiers to external services.
UA = {"User-Agent": "banking77-portfolio-downloader/1.0"}


def download() -> dict:
    cfg = load_config()["data"]
    raw = path("raw")
    out = {}
    for name in ("train", "test"):
        dest = raw / f"{name}.csv"
        if not dest.exists():
            r = requests.get(cfg[f"{name}_url"], headers=UA, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
        out[name] = dest
    card = raw / "hf_dataset_card.md"
    if not card.exists():
        try:
            r = requests.get(cfg["hf_card_url"], headers=UA, timeout=30)
            if r.status_code == 200:
                card.write_text(r.text, encoding="utf-8")
        except requests.RequestException:
            pass
    return out


def load_raw(name: str) -> pd.DataFrame:
    df = pd.read_csv(path("raw") / f"{name}.csv")
    df = df.rename(columns={"category": "label"})
    df["text"] = df["text"].astype(str).str.strip()
    return df[["text", "label"]]


def stratified_take(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Take `per_class` rows per label (all rows if a label has fewer), reproducibly."""
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(parts).sort_index()


def stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rest, held_out) with round(frac * n_label) rows of each label held out (at least 1)."""
    held = []
    for _, g in df.groupby("label", sort=True):
        n = max(1, int(round(frac * len(g))))
        held.append(g.sample(n=n, random_state=seed))
    held_df = pd.concat(held).sort_index()
    rest = df.drop(index=held_df.index)
    return rest, held_df


def make_splits() -> dict:
    cfg = load_config()["data"]
    seed = cfg["seed"]
    train_full = load_raw("train")
    test = load_raw("test")
    train, val = stratified_split(train_full, cfg["val_frac"], seed)
    test_subset = stratified_take(test, cfg["subset_per_class"], seed)
    val_subset = stratified_take(val, cfg["dev_subset_per_class"], seed)
    labels = sorted(train_full["label"].unique())
    out = path("splits")
    for name, d in [("train", train), ("val", val), ("test", test),
                    ("test_subset", test_subset), ("val_subset", val_subset)]:
        d.reset_index().rename(columns={"index": "orig_idx"}).to_csv(out / f"{name}.csv", index=False)
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    summary = {
        "seed": seed,
        "n_labels": len(labels),
        "rows": {"train_original": len(train_full), "train": len(train), "val": len(val), "test": len(test),
                 "test_subset": len(test_subset), "val_subset": len(val_subset)},
        "per_class_min_max": {
            name: [int(d["label"].value_counts().min()), int(d["label"].value_counts().max())]
            for name, d in [("train", train), ("val", val), ("test", test), ("test_subset", test_subset)]
        },
        "train_val_text_overlap": int(len(set(train["text"]) & set(val["text"]))),
        "train_test_text_overlap": int(len(set(train_full["text"]) & set(test["text"]))),
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(path("splits") / f"{name}.csv")


def load_labels() -> list[str]:
    return json.loads((path("splits") / "labels.json").read_text(encoding="utf-8"))
