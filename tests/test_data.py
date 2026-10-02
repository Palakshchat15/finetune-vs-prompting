import pandas as pd
import pytest

from src.config import path
from src.data import load_labels, load_split, stratified_split, stratified_take


def toy():
    return pd.DataFrame({"text": [f"t{i}" for i in range(100)], "label": ["a"] * 70 + ["b"] * 30})


def test_stratified_split_proportions_and_disjoint():
    rest, held = stratified_split(toy(), 0.1, 42)
    assert held["label"].value_counts().to_dict() == {"a": 7, "b": 3}
    assert set(rest.index).isdisjoint(held.index) and len(rest) + len(held) == 100


def test_stratified_split_reproducible():
    assert list(stratified_split(toy(), 0.1, 42)[1].index) == list(stratified_split(toy(), 0.1, 42)[1].index)


def test_stratified_take_caps_small_classes():
    t = stratified_take(toy(), 40, 1)
    assert t["label"].value_counts().to_dict() == {"a": 40, "b": 30}


needs_data = pytest.mark.skipif(not (path("splits") / "labels.json").exists(), reason="run scripts/prepare_data.py")


@needs_data
def test_real_splits():
    labels = load_labels()
    tr, va, te, sub = (load_split(s) for s in ("train", "val", "test", "test_subset"))
    assert len(labels) == 77
    assert (len(tr), len(va), len(te), len(sub)) == (9000, 1003, 3080, 770)
    assert set(tr["orig_idx"]).isdisjoint(va["orig_idx"])
    assert set(sub["orig_idx"]) <= set(te["orig_idx"])
    assert (sub["label"].value_counts() == 10).all() and sub["label"].nunique() == 77
    assert set(va["label"]) == set(labels)
