import json
import threading
import time

import pytest

from src.evalcache import append, read_cache
from src.llm_lock import PROJECT, llm_lock, owner


def test_lock_created_and_removed(tmp_path):
    lock = tmp_path / ".llm_lock"
    with llm_lock("t", lock=lock):
        assert owner(lock) == PROJECT
    assert not lock.exists()


def test_lock_removed_on_failure(tmp_path):
    lock = tmp_path / ".llm_lock"
    with pytest.raises(RuntimeError):
        with llm_lock("t", lock=lock):
            raise RuntimeError("boom")
    assert not lock.exists()


def test_lock_waits_for_other_project(tmp_path):
    lock = tmp_path / ".llm_lock"
    lock.write_text("rag_financebench\n", encoding="utf-8")
    with pytest.raises(TimeoutError):
        with llm_lock("t", lock=lock, poll_s=0.05, max_wait_s=0.2):
            pass
    assert owner(lock) == "rag_financebench"  # never deletes another project's lock
    threading.Timer(0.2, lock.unlink).start()
    t0 = time.time()
    with llm_lock("t", lock=lock, poll_s=0.05, max_wait_s=5):
        assert owner(lock) == PROJECT and time.time() - t0 >= 0.15
    assert not lock.exists()


def test_reentrant_same_project_leaves_lock(tmp_path):
    lock = tmp_path / ".llm_lock"
    with llm_lock("outer", lock=lock):
        with llm_lock("inner", lock=lock):
            pass
        assert lock.exists()
    assert not lock.exists()


def test_cache_resume_tolerates_truncated_line(tmp_path):
    p = tmp_path / "c.jsonl"
    append(p, {"orig_idx": 1, "prediction": "a"})
    append(p, {"orig_idx": 2, "prediction": "b"})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"orig_idx": 3, "predic')  # killed mid-write
    assert set(read_cache(p)) == {1, 2}
