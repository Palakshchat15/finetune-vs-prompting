r"""Evaluate one prompted/fine-tuned system on a split, caching every item (resumable).

  .venv\Scripts\python scripts\eval_llm.py --system zero_shot_7b --split test_subset
  .venv\Scripts\python scripts\eval_llm.py --system few_shot_7b --split val_subset      # prompt tuning (dev)
  .venv\Scripts\python scripts\eval_llm.py --system base_1p5b --split test_subset --limit 20   # speed probe

Systems and models come from config.yaml (llm.systems). Holds AI_Engineer_Portfolio/.llm_lock for the run
(waits if another project holds it). Items already in data/cache/<system>__<model>__<split>.jsonl are skipped.
A warm-up request (model load) is made first and excluded from latency. Afterwards the model's RAM/VRAM
footprint (Ollama /api/ps) and disk size (/api/tags) are written to results/footprint_<system>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402

from src import llm  # noqa: E402
from src.config import load_config, path  # noqa: E402
from src.data import load_split  # noqa: E402
from src.evalcache import append, cache_file, read_cache  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402
from src.systems import classify, system_model  # noqa: E402


def run_eval(system: str, split: str, limit: int | None = None, use_lock: bool = True,
             classify_fn=classify, check=True, footprint=True, lock_path=None) -> Path:
    """Score `system` on `split`, appending one JSONL record per item; skips items already cached."""
    model = system_model(system)
    tag = llm.check_model(model) if check else {}  # clear error if Ollama or the model is missing; no fallback
    df = load_split(split)
    if limit:
        df = df.head(limit)
    cpath = cache_file(system, model, split)
    done = read_cache(cpath)
    todo = df[~df["orig_idx"].isin(done)]
    print(f"{system} ({model}) on {split}: {len(done)} cached, {len(todo)} to run -> {cpath.name}", flush=True)
    if todo.empty:
        return cpath
    print(f"free RAM {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)

    def run():
        classify_fn(str(todo.iloc[0]["text"]), system)  # warm-up / model load, not recorded
        t_start = time.time()
        for i, row in enumerate(todo.itertuples(index=False), 1):
            try:
                r = classify_fn(str(row.text), system)
                err = None
            except llm.OllamaUnavailable:
                raise
            except Exception as e:  # noqa: BLE001  (record and continue; counted as invalid)
                r, err = {"prediction": None, "valid": False, "raw": None}, f"{type(e).__name__}: {e}"
            append(cpath, {"orig_idx": int(row.orig_idx), "label": row.label, "text": row.text,
                           "system": system, "model": model, **r, "error": err, "ts": time.time()})
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_start
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/item  eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
        if footprint:
            fp = {"system": system, "model": model, "disk_size_gb": tag.get("size", 0) / 1e9,
                  "details": tag.get("details", {}),
                  "loaded": [m for m in llm.loaded_models() if m.get("name", "").split(":")[0] == model.split(":")[0]],
                  "host_ram_total_gb": psutil.virtual_memory().total / 1e9, "measured_at": time.time()}
            (path("results") / f"footprint_{system}.json").write_text(json.dumps(fp, indent=1), encoding="utf-8")

    if use_lock:
        with llm_lock(f"eval {system} {split}", lock=lock_path):
            run()
    else:
        run()
    return cpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(load_config()["llm"]["systems"]))
    ap.add_argument("--split", default="test_subset", choices=["test_subset", "test", "val_subset", "val"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (speed probe)")
    ap.add_argument("--no-lock", action="store_true", help="skip the shared lock (only for a few-item smoke test)")
    a = ap.parse_args()
    run_eval(a.system, a.split, a.limit, use_lock=not a.no_lock)


if __name__ == "__main__":
    main()
