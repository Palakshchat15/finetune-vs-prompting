r"""(j) Evaluate the fine-tuned DistilBERT checkpoint (models/distilbert/) on test_subset and the full test.

Same per-item outputs as the LLM systems: data/cache/distilbert_pt__distilbert-base-uncased__<split>.jsonl
(one record per message, resumable; prediction, valid, confidence, t_total = one-request wall time on the main
device). Summary with GPU and CPU latency: results/distilbert_pt.json. Then run summarize_results.py.

  .venv\Scripts\python scripts\eval_distilbert.py                    # main device = cuda if available
  .venv\Scripts\python scripts\eval_distilbert.py --device cpu       # everything on CPU
  .venv\Scripts\python scripts\eval_distilbert.py --split test_subset

Latency protocol (as for the other systems): one message at a time, model warm (3 warm-up requests excluded),
tokenisation included, CUDA synchronised. Main device: every item. CPU: the first `latency_sample_n` items of
each split with torch threads = distilbert.cpu_threads. Holds AI_Engineer_Portfolio/.llm_lock.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402

from src.config import ROOT, detect_device, load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.evalcache import append, cache_file, read_cache  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402
from src.metrics import cost_per_1k, latency_stats, per_class_f1, summarize, top_confusions  # noqa: E402

SPLITS = ["test_subset", "test"]
WARMUP = 3


def checkpoint_id(model_dir: Path) -> str:
    """Fingerprint of the checkpoint, stored in every cache record so a retrain cannot mix with old results."""
    meta = (model_dir / "label_map.json").read_text(encoding="utf-8")
    w = model_dir / "model.safetensors"
    stamp = f"{w.stat().st_size}:{int(w.stat().st_mtime)}" if w.exists() else ""
    return hashlib.sha1((meta + stamp).encode("utf-8")).hexdigest()[:12]


def score_split(clf, split: str, system: str, model_name: str, ckpt: str, device: str) -> Path:
    df = load_split(split)
    cpath = cache_file(system, model_name, split)
    done = read_cache(cpath)
    stale = {r.get("ckpt") for r in done.values()} - {ckpt}
    if stale:
        raise SystemExit(f"{cpath} holds results from another checkpoint {stale}. Delete it and re-run.")
    todo = df[~df["orig_idx"].isin(done)]
    print(f"{system} on {split} ({device}): {len(done)} cached, {len(todo)} to run -> {cpath.name}", flush=True)
    if todo.empty:
        return cpath
    for t in df["text"].head(WARMUP):
        clf.timed_predict_one(str(t))
    for i, row in enumerate(todo.itertuples(index=False), 1):
        try:
            r, dt = clf.timed_predict_one(str(row.text))
            rec = {"prediction": r["label"], "valid": True, "raw": r["label"], "confidence": r["confidence"],
                   "t_retrieve": 0.0, "t_model": dt, "t_total": dt, "error": None}
        except Exception as e:  # noqa: BLE001  (record and continue; counted as invalid)
            rec = {"prediction": None, "valid": False, "raw": None, "error": f"{type(e).__name__}: {e}"}
        append(cpath, {"orig_idx": int(row.orig_idx), "label": row.label, "text": row.text, "system": system,
                       "model": model_name, **rec, "prompt_tokens": 0, "completion_tokens": 0, "device": device,
                       "ckpt": ckpt, "ts": time.time()})
        if i % 500 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    return cpath


def batch_throughput(clf, texts: list[str], batch_size: int) -> float:
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        clf.predict(texts[i:i + batch_size])
    if clf.device == "cuda":
        clf._torch.cuda.synchronize()
    return len(texts) / (time.perf_counter() - t0)


def cpu_latency(clf_cpu, texts: list[str]) -> dict:
    for t in texts[:WARMUP]:
        clf_cpu.timed_predict_one(t)
    return latency_stats([clf_cpu.timed_predict_one(t)[1] for t in texts])


def main():
    from src.distilbert import DistilBertIntentClassifier  # noqa: PLC0415  (needs torch)

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, action="append", help="repeatable; default: both")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    c = cfg["distilbert"]
    system, model_name = c["system_key"], c["base_model"]
    model_dir = ROOT / c["out_dir"]
    device = a.device or detect_device()
    device = "cpu" if device == "mps" else device
    labels = load_labels()

    def run():
        clf = DistilBertIntentClassifier(model_dir, device=device)
        ckpt = checkpoint_id(model_dir)
        torch = clf._torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        clf_cpu = clf if device == "cpu" else DistilBertIntentClassifier(model_dir, "cpu", num_threads=c["cpu_threads"])
        if device == "cpu":
            torch.set_num_threads(c["cpu_threads"])
        out_p = path("results") / f"{system}.json"
        out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
        out.update({"system": system, "label": c["label"], "model": model_name, "checkpoint": ckpt,
                    "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "cpu_threads": c["cpu_threads"], "torch": torch.__version__,
                    "model_size_mb": clf.size_on_disk_mb(), "host_ram_total_gb": psutil.virtual_memory().total / 1e9})
        for split in (a.split or SPLITS):
            cpath = score_split(clf, split, system, model_name, ckpt, device)
            gold = load_split(split)
            cache = read_cache(cpath)
            recs = [cache[i] for i in gold["orig_idx"]]
            y, p = [r["label"] for r in recs], [r["prediction"] for r in recs]
            texts = [str(t) for t in gold["text"]]
            lat_dev = latency_stats([r["t_total"] for r in recs if r.get("t_total") is not None])
            lat_cpu = cpu_latency(clf_cpu, texts[: c["latency_sample_n"]])
            out[split] = {**summarize(y, p, labels),
                          "latency_main_device": {"device": device, **lat_dev, "n": len(recs)},
                          "latency_cpu": {**lat_cpu, "n": min(len(texts), c["latency_sample_n"]),
                                          "threads": c["cpu_threads"]},
                          "batch_throughput_rps": batch_throughput(clf, texts, c["eval_batch_size"]),
                          "cost_per_1k_usd_cpu_assumption": cost_per_1k(lat_cpu["mean_ms"] / 1000,
                                                                         cfg["cost"]["cpu_instance_usd_per_hour"]),
                          "per_class_f1": per_class_f1(y, p, labels), "top_confusions": top_confusions(y, p, 15)}
            s = out[split]
            print(f"{split}: n={s['n']} acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f} "
                  f"{device} p50={lat_dev['p50_ms']:.1f}ms p95={lat_dev['p95_ms']:.1f}ms | "
                  f"cpu p50={lat_cpu['p50_ms']:.1f}ms p95={lat_cpu['p95_ms']:.1f}ms", flush=True)
        out["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
        out["process_rss_mb"] = psutil.Process().memory_info().rss / 1e6
        out["measured_at"] = time.time()
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {out_p}", flush=True)

    if a.no_lock:
        run()
    else:
        with llm_lock(f"eval {system}"):
            run()


if __name__ == "__main__":
    main()
