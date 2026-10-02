"""Shared CPU/LLM lock across the AI Engineer portfolio projects (AI_Engineer_Portfolio/.llm_lock).

Before an Ollama-heavy job we create the file (project, start time, pid, job) and delete it when the job
ends, including on failure. If another project holds it we wait and poll. If this project already holds
it (e.g. an outer queue), we proceed and leave deletion to the holder.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT = "finetune_vs_prompting"
LOCK = Path(__file__).resolve().parents[2] / ".llm_lock"


def owner(lock: Path | None = None) -> str:
    try:
        return (lock or LOCK).read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


@contextmanager
def llm_lock(job: str, poll_s: int = 30, lock: Path | None = None, max_wait_s: float | None = None):
    lock = lock or LOCK
    waited, t0, created = False, time.time(), False
    while True:
        if not lock.exists():
            try:
                with open(lock, "x", encoding="utf-8") as f:
                    f.write(f"{PROJECT}\nstarted {datetime.now().isoformat(timespec='seconds')}\n"
                            f"pid {os.getpid()}\njob: {job}\n")
                created = True
                break
            except FileExistsError:
                continue
        if owner(lock) == PROJECT:
            break
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise TimeoutError(f"LLM lock still held by '{owner(lock)}' after {max_wait_s}s")
        if not waited:
            print(f"LLM lock held by '{owner(lock)}' ({lock}); waiting ...", flush=True)
            waited = True
        time.sleep(poll_s)
    try:
        yield
    finally:
        if created:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
