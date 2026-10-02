r"""(a) TF-IDF + logistic regression: choose C on validation, refit on train, score test once.

Outputs: results/baseline_tfidf.json, results/predictions/tfidf_{test,test_subset}.csv,
         models/tfidf_logreg.joblib (used by the app).
Usage: .venv\Scripts\python scripts\run_baseline.py
r"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from src.baseline import make_pipeline  # noqa: E402
from src.config import load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.metrics import cost_per_1k, latency_stats, per_class_f1, summarize, top_confusions  # noqa: E402


def main():
    cfg = load_config()
    labels = load_labels()
    train, val = load_split("train"), load_split("val")

    # 1) Tune C on validation only.
    grid = []
    for C in cfg["baseline"]["C_grid"]:
        m = make_pipeline(C).fit(train["text"], train["label"])
        s = summarize(list(val["label"]), list(m.predict(val["text"])), labels)
        grid.append({"C": C, **s})
        print(f"C={C:<5} val acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}", flush=True)
    best = max(grid, key=lambda r: (r["macro_f1"], r["accuracy"]))
    C = best["C"]

    # 2) Refit on train only (val stays unseen, same data the fine-tune sees), time the fit.
    t0 = time.perf_counter()
    model = make_pipeline(C).fit(train["text"], train["label"])
    fit_s = time.perf_counter() - t0
    mpath = path("models") / "tfidf_logreg.joblib"
    joblib.dump(model, mpath)

    out = {"system": "tfidf_logreg", "label": "(a) TF-IDF + logistic regression", "C": C,
           "val_grid": grid, "fit_seconds": fit_s, "model_size_mb": os.path.getsize(mpath) / 1e6,
           "threads_note": "single process, sklearn defaults, Intel i5-1235U build laptop"}
    pred_dir = path("results") / "predictions"
    pred_dir.mkdir(exist_ok=True)
    proc = psutil.Process()
    for split in ("test", "test_subset"):
        df = load_split(split)
        # Batch prediction (throughput) and per-request latency (one message at a time, like the app).
        t0 = time.perf_counter()
        preds = list(model.predict(df["text"]))
        batch_s = time.perf_counter() - t0
        lat = []
        for t in df["text"].head(500):
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
            .to_csv(pred_dir / f"tfidf_{split}.csv", index=False)
        print(f"{split}: n={res['n']} acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
              f"p50={ls['p50_ms']:.2f}ms p95={ls['p95_ms']:.2f}ms", flush=True)
    out["process_rss_mb"] = proc.memory_info().rss / 1e6
    (path("results") / "baseline_tfidf.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
