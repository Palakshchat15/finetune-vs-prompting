"""eval_llm.run_eval: per-item cache, resume after a crash, lock released - with a fake classifier (no Ollama)."""
import importlib.util

import pytest

from src import evalcache
from src.config import ROOT, path

pytestmark = pytest.mark.skipif(not (path("splits") / "labels.json").exists(), reason="run scripts/prepare_data.py")


def load_eval():
    spec = importlib.util.spec_from_file_location("eval_llm", ROOT / "scripts" / "eval_llm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_resume_after_crash(tmp_path, monkeypatch):
    m = load_eval()
    monkeypatch.setattr(evalcache, "path", lambda key: tmp_path)
    monkeypatch.setattr(m, "cache_file", lambda s, mo, sp: tmp_path / "c.jsonl")
    lock = tmp_path / ".llm_lock"
    calls = {"n": 0}

    def flaky(text, system):
        calls["n"] += 1
        if calls["n"] == 5:  # warm-up + 3 items, then "power cut"
            raise KeyboardInterrupt
        return {"prediction": "card_arrival", "valid": True, "raw": "x", "t_total": 0.01}

    with pytest.raises(KeyboardInterrupt):
        m.run_eval("zero_shot_7b", "test_subset", limit=8, classify_fn=flaky, check=False, footprint=False,
                   lock_path=lock)
    assert not lock.exists()  # released on failure
    assert len(evalcache.read_cache(tmp_path / "c.jsonl")) == 3

    seen = []

    def ok(text, system):
        seen.append(text)
        return {"prediction": "card_arrival", "valid": True, "raw": "x", "t_total": 0.01}

    m.run_eval("zero_shot_7b", "test_subset", limit=8, classify_fn=ok, check=False, footprint=False, lock_path=lock)
    assert len(seen) == 1 + 5  # warm-up + only the 5 missing items
    assert len(evalcache.read_cache(tmp_path / "c.jsonl")) == 8
    assert not lock.exists()


def test_item_errors_are_recorded_not_fatal(tmp_path, monkeypatch):
    m = load_eval()
    monkeypatch.setattr(m, "cache_file", lambda s, mo, sp: tmp_path / "c.jsonl")
    calls = {"n": 0}

    def bad_after_warmup(text, system):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"prediction": "card_arrival", "valid": True}
        raise ValueError("bad json")

    m.run_eval("zero_shot_7b", "test_subset", limit=3, classify_fn=bad_after_warmup, check=False, footprint=False,
               use_lock=False)
    recs = evalcache.read_cache(tmp_path / "c.jsonl")
    assert len(recs) == 3
    assert all(not r["valid"] and r["error"].startswith("ValueError") for r in recs.values())
