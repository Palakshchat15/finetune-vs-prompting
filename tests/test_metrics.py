import pytest

from src.metrics import cost_per_1k, latency_stats, summarize, top_confusions

L = ["a", "b", "c"]


def test_summarize_counts_invalid_as_wrong():
    s = summarize(["a", "b", "c", "a"], ["a", "b", None, "zzz"], L)
    assert s["accuracy"] == 0.5 and s["invalid_rate"] == 0.5


def test_perfect():
    s = summarize(L, L, L)
    assert s["accuracy"] == 1 and s["macro_f1"] == 1


def test_confusions_and_latency_and_cost():
    assert top_confusions(["a", "a", "b"], ["b", "b", "b"])[0] == {"true": "a", "pred": "b", "count": 2}
    ls = latency_stats([0.1, 0.2, 0.3])
    assert ls["p50_ms"] == pytest.approx(200) and ls["throughput_rps"] == pytest.approx(5)
    assert cost_per_1k(3.6, 1.0) == pytest.approx(1.0)
