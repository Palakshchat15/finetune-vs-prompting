r"""Small load test for the DistilBERT service (local uvicorn or the Kubernetes Service).

Sends real test-set messages (data/splits/test.csv) to POST /predict from N concurrent threads and reports
client-side latency p50/p95/p99, throughput, error count and accuracy of the answers. Uses requests only.

  .venv\Scripts\python scripts\load_test.py --url http://localhost:8080 --requests 500 --concurrency 4
  .venv\Scripts\python scripts\load_test.py --url http://localhost:8080 --duration 120 --concurrency 8   # HPA demo
Writes results/load_test_<tag>.json (tag defaults to c<concurrency>).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def percentiles(ms: list[float]) -> dict:
    """p50/p95/p99/mean/max of a list of milliseconds (nearest-rank on the sorted list)."""
    if not ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None, "max_ms": None}
    s = sorted(ms)

    def q(p):
        return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]

    return {"p50_ms": q(50), "p95_ms": q(95), "p99_ms": q(99), "mean_ms": sum(s) / len(s), "max_ms": s[-1]}


def summarize_run(results: list[dict], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r["ok"]]
    lab = [r for r in ok if r.get("gold") is not None]
    return {"requests": len(results), "ok": len(ok), "errors": len(results) - len(ok),
            "error_samples": sorted({r["error"] for r in results if not r["ok"]})[:5],
            "concurrency": concurrency, "wall_s": wall_s,
            "throughput_rps": len(ok) / wall_s if wall_s > 0 else None,
            "accuracy_of_answers": (sum(r["pred"] == r["gold"] for r in lab) / len(lab)) if lab else None,
            **percentiles([r["ms"] for r in ok]),
            "server_latency_p50_ms": percentiles([r["server_ms"] for r in ok if r.get("server_ms")])["p50_ms"]}


def main():
    import pandas as pd  # noqa: PLC0415
    import requests  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=500, help="total requests (ignored if --duration is set)")
    ap.add_argument("--duration", type=float, default=None, help="run for this many seconds instead")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    base = a.url.rstrip("/")
    r = requests.get(f"{base}/ready", timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"service not ready: {r.status_code} {r.text[:200]}")
    df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    items = itertools.cycle(list(zip(df["text"].astype(str), df["label"])))
    lock = threading.Lock()
    deadline = time.time() + a.duration if a.duration else None
    budget = {"left": a.requests}
    session_local = threading.local()

    def next_item():
        with lock:
            if deadline is None:
                if budget["left"] <= 0:
                    return None
                budget["left"] -= 1
            elif time.time() > deadline:
                return None
            return next(items)

    def worker():
        out = []
        s = getattr(session_local, "s", None) or requests.Session()
        session_local.s = s
        while (it := next_item()) is not None:
            text, gold = it
            t0 = time.perf_counter()
            try:
                resp = s.post(f"{base}/predict", json={"text": text, "top_k": 1}, timeout=a.timeout)
                ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    j = resp.json()
                    out.append({"ok": True, "ms": ms, "pred": j["label"], "gold": gold, "server_ms": j["latency_ms"]})
                else:
                    out.append({"ok": False, "ms": ms, "error": f"HTTP {resp.status_code}"})
            except requests.RequestException as e:
                out.append({"ok": False, "ms": (time.perf_counter() - t0) * 1000, "error": type(e).__name__})
        return out

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(worker) for _ in range(a.concurrency)]
        results = [x for f in futs for x in f.result()]
    summ = {"url": base, **summarize_run(results, time.perf_counter() - t0, a.concurrency), "measured_at": time.time()}
    out = ROOT / "results" / f"load_test_{a.tag or f'c{a.concurrency}'}.json"
    out.write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
