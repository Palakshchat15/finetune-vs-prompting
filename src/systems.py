"""The five systems behind one interface: classify(text) -> dict. Used by the eval scripts and the app."""
from __future__ import annotations

import time
from functools import lru_cache

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import llm, prompts
from .config import load_config, path
from .data import load_labels, load_split

TFIDF_KEY = "tfidf_logreg"


def system_keys() -> list[str]:
    return [TFIDF_KEY, *load_config()["llm"]["systems"].keys()]


def system_label(key: str) -> str:
    if key == TFIDF_KEY:
        return "(a) TF-IDF + logistic regression"
    return load_config()["llm"]["systems"][key]["label"]


def system_model(key: str) -> str:
    return "sklearn" if key == TFIDF_KEY else load_config()["llm"]["systems"][key]["model"]


@lru_cache(maxsize=1)
def _labels() -> tuple[list[str], set[str]]:
    labs = load_labels()
    return labs, set(labs)


@lru_cache(maxsize=1)
def _tfidf_model():
    p = path("models") / "tfidf_logreg.joblib"
    if not p.exists():
        raise FileNotFoundError("TF-IDF model missing. Run: .venv/Scripts/python scripts/run_baseline.py")
    return joblib.load(p)


class KnnExamples:
    """Few-shot example retrieval: the k nearest TRAIN messages by TF-IDF cosine similarity.
    Only the train split is indexed, so validation/test messages never leak into prompts."""

    def __init__(self, k: int):
        tr = load_split("train")
        self.texts, self.labels, self.k = list(tr["text"]), list(tr["label"]), k
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.texts)

    def get(self, text: str) -> list[tuple[str, str]]:
        sims = (self.X @ self.vec.transform([text]).T).toarray().ravel()
        idx = np.argsort(-sims, kind="stable")[: self.k]
        # Most similar example last (closest to the query in the prompt).
        return [(self.texts[i], self.labels[i]) for i in idx[::-1]]


@lru_cache(maxsize=1)
def _knn() -> KnnExamples:
    return KnnExamples(load_config()["llm"]["few_shot"]["k"])


def classify(text: str, key: str) -> dict:
    """Return {prediction, valid, raw, t_retrieve, t_model, t_total, prompt_tokens, completion_tokens, model}."""
    labels, label_set = _labels()
    t0 = time.perf_counter()
    if key == TFIDF_KEY:
        pred = str(_tfidf_model().predict([text])[0])
        t = time.perf_counter() - t0
        return {"prediction": pred, "valid": True, "raw": pred, "t_retrieve": 0.0, "t_model": t, "t_total": t,
                "prompt_tokens": 0, "completion_tokens": 0, "model": "sklearn"}
    sc = load_config()["llm"]["systems"][key]
    t_retrieve = 0.0
    if sc["mode"] == "zero_shot":
        msgs, fmt = prompts.zero_shot_messages(text, labels), prompts.intent_schema(labels)
    elif sc["mode"] == "few_shot":
        t1 = time.perf_counter()
        ex = _knn().get(text)
        t_retrieve = time.perf_counter() - t1
        msgs, fmt = prompts.few_shot_messages(text, labels, ex), prompts.intent_schema(labels)
    elif sc["mode"] == "finetuned":
        msgs, fmt = prompts.finetune_messages(text), None
    else:
        raise ValueError(f"unknown mode {sc['mode']}")
    r = llm.chat(msgs, sc["model"], fmt=fmt)
    if fmt is None:
        pred, valid = prompts.parse_plain(r["content"], label_set)
    else:
        pred, valid = prompts.parse_structured(r["content"], label_set)
    return {"prediction": pred, "valid": valid, "raw": r["content"], "t_retrieve": t_retrieve,
            "t_model": r["wall_s"], "t_total": time.perf_counter() - t0, "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "model": sc["model"],
            "prompt_eval_s": r["prompt_eval_s"], "eval_s": r["eval_s"], "load_s": r["load_s"]}
