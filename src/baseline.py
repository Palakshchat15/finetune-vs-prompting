"""(a) TF-IDF + logistic regression baseline."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import load_config


def make_pipeline(C: float) -> Pipeline:
    b = load_config()["baseline"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=tuple(b["ngram_range"]), sublinear_tf=b["sublinear_tf"],
                                  min_df=b["min_df"], lowercase=True)),
        ("clf", LogisticRegression(C=C, max_iter=3000)),
    ])
