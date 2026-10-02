import pytest

from src.config import path
from src.data import load_split

pytestmark = pytest.mark.skipif(not (path("models") / "tfidf_logreg.joblib").exists(),
                                reason="run scripts/prepare_data.py and scripts/run_baseline.py")


def test_tfidf_classify_returns_valid_label():
    from src.systems import classify
    r = classify("My new card still hasn't arrived", "tfidf_logreg")
    assert r["valid"] and r["prediction"] == "card_arrival" and r["t_total"] >= 0


def test_knn_examples_come_from_train_only():
    from src.systems import KnnExamples
    knn = KnnExamples(5)
    train = set(load_split("train")["text"])
    q = load_split("test")["text"].iloc[0]
    ex = knn.get(q)
    assert len(ex) == 5 and all(t in train for t, _ in ex)


def test_system_registry():
    from src.systems import system_keys
    assert system_keys()[:5] == ["tfidf_logreg", "zero_shot_7b", "few_shot_7b", "base_1p5b", "zero_shot_3b"]
