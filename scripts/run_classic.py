r"""(f)-(i) Classic NLP baselines: choose hyperparameters on validation, refit on train, score test once.

Same outputs and metrics as run_baseline.py, per system key (config.yaml classic.systems):
  results/classic_<key>.json, results/predictions/<key>_{test,test_subset}.csv, models/<key>.joblib

Usage:
  .venv\Scripts\python scripts\run_classic.py                      # all four systems
  .venv\Scripts\python scripts\run_classic.py --system svm_char     # one system
  .venv\Scripts\python scripts\run_classic.py --skip-existing       # resume: skip systems already scored
spacy_lemma_svm needs requirements-nlp.txt (spaCy + en_core_web_sm); the others need only scikit-learn.
Holds AI_Engineer_Portfolio/.llm_lock while running (CPU-heavy grid fits; keeps latency numbers clean).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from src.classic import clear_memo, grid_edge_note, make_pipeline, param_grid, select_on_val, system_keys  # noqa: E402
from src.config import load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402
from src.metrics import cost_per_1k, latency_stats, per_class_f1, summarize, top_confusions  # noqa: E402


def result_path(key: str) -> Path:
    return path("results") / f"classic_{key}.json"


def run_system(key: str) -> dict:
    cfg = load_config()
    c = cfg["classic"]
    sc = c["systems"][key]
    labels = load_labels()
    train, val = load_split("train"), load_split("val")
    print(f"== {key}: {sc['label']}", flush=True)

    # 1) Hyperparameters on validation only.
    grid = param_grid(key)
    best, rows = select_on_val(lambda p: make_pipeline(key, p, disk_cache=True), grid,
                               train["text"], train["label"], val["text"], val["label"], labels)
    note = grid_edge_note(best, grid)
    print(f"  best {best}" + (f"  NOTE: {note}" if note else ""), flush=True)

    # 2) Refit on train only (val stays unseen, as for (a) and the fine-tunes), time the fit.
    t0 = time.perf_counter()
    model = make_pipeline(key, best, disk_cache=True).fit(train["text"], train["label"])
    fit_s = time.perf_counter() - t0
    if sc["preprocess"] == "spacy":
        model.set_params(spacy__disk_cache=False)  # the saved model (app) never writes the lemma cache
    mpath = path("models") / f"{key}.joblib"
    joblib.dump(model, mpath)

    out = {"system": key, "label": sc["label"], "best_params": best, "val_grid": rows, "grid_note": note,
           "fit_seconds": fit_s, "model_size_mb": os.path.getsize(mpath) / 1e6,
           "threads_note": "single process, sklearn defaults"}
    if sc["preprocess"] == "spacy":
        import spacy  # noqa: PLC0415
        out["spacy"] = {"version": spacy.__version__, "model": c["spacy_model"],
                        "keep_words": c["spacy_keep_words"],
                        "note": "fit_seconds excludes lemmatisation of train (memoised during the val grid); "
                                "per-request latency includes spaCy on the raw message"}
    pred_dir = path("results") / "predictions"
    pred_dir.mkdir(exist_ok=True)
    proc = psutil.Process()
    for split in ("test", "test_subset"):
        df = load_split(split)
        t0 = time.perf_counter()
        preds = [str(p) for p in model.predict(df["text"])]
        batch_s = time.perf_counter() - t0
        lat = []
        clear_memo()  # spaCy: time the real lemmatisation of each message, not a memo lookup
        for t in df["text"].head(c["latency_sample_n"]):
            t1 = time.perf_counter()
            model.predict([t])
            lat.append(time.perf_counter() - t1)
        ls = latency_stats(lat)
        res = {**summarize(list(df["label"]), preds, labels), **ls,
               "batch_throughput_rps": len(df) / batch_s,
               "latency_sample_n": len(lat),
               "cost_per_1k_usd_cpu_assumption": cost_per_1k(ls["mean_ms"] / 1000,
                                                              cfg["cost"]["cpu_instance_usd_per_hour"]),
               "per_class_f1": per_class_f1(list(df["label"]), preds, labels),
               "top_confusions": top_confusions(list(df["label"]), preds, 15)}
        out[split] = res
        pd.DataFrame({"orig_idx": df["orig_idx"], "text": df["text"], "label": df["label"], "pred": preds}) \
            .to_csv(pred_dir / f"{key}_{split}.csv", index=False)
        print(f"  {split}: n={res['n']} acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
              f"p50={ls['p50_ms']:.2f}ms p95={ls['p95_ms']:.2f}ms", flush=True)
    out["process_rss_mb"] = proc.memory_info().rss / 1e6
    result_path(key).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    keys = system_keys()
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=keys, action="append", help="repeatable; default: all")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-lock", action="store_true", help="skip the shared lock (only for a quick smoke test)")
    a = ap.parse_args()
    todo = [k for k in (a.system or keys) if not (a.skip_existing and result_path(k).exists())]

    def run():
        for k in todo:
            run_system(k)

    if a.no_lock:
        run()
    else:
        with llm_lock(f"classic baselines {','.join(todo)}"):
            run()


if __name__ == "__main__":
    main()
