r"""Build the comparison from results/baseline_tfidf.json and the per-item LLM caches, plus (added in the
2026-10 upgrade) the classic NLP baselines (results/classic_<key>.json) and the PyTorch DistilBERT classifier
(per-item cache data/cache/distilbert_pt__*.jsonl + results/distilbert_pt.json for the CPU latency).
New systems only add rows (and the columns p50_ms_cpu, p95_ms_cpu, latency_device); the rows of (a)-(e)
are computed exactly as before.

Writes results/comparison.csv (one row per system x split), results/per_class_f1.csv,
and results/confusions.json. Systems without a cache are listed as
"not run yet" (fine-tuned: "pending: run the Colab notebook"). Safe to run at any time; no LLM calls.
  .venv\Scripts\python scripts\summarize_results.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.evalcache import cache_file, read_cache  # noqa: E402
from src.metrics import cost_per_1k, latency_stats, per_class_f1, summarize, top_confusions  # noqa: E402
from src.systems import TFIDF_KEY, classic_keys, distilbert_key, system_label, system_model  # noqa: E402

SPLITS = ["test_subset", "test"]


def main():
    cfg = load_config()
    labels = load_labels()
    rows, pcf, conf = [], {}, {}
    base = json.loads((path("results") / "baseline_tfidf.json").read_text(encoding="utf-8"))
    for split in SPLITS:
        b = base[split]
        rows.append({"system": TFIDF_KEY, "label": system_label(TFIDF_KEY), "model": "sklearn", "split": split,
                     "status": "done", "n": b["n"], "accuracy": b["accuracy"], "macro_f1": b["macro_f1"],
                     "invalid_rate": 0.0, "p50_ms": b["p50_ms"], "p95_ms": b["p95_ms"],
                     "throughput_rps": b["throughput_rps"],
                     "cost_per_1k_usd": b["cost_per_1k_usd_cpu_assumption"],
                     "size_on_disk_mb": base["model_size_mb"]})
        pcf[f"{TFIDF_KEY}|{split}"] = b["per_class_f1"]
        conf[f"{TFIDF_KEY}|{split}"] = b["top_confusions"]
    for key, sc in cfg["llm"]["systems"].items():
        model = system_model(key)
        fp_path = path("results") / f"footprint_{key}.json"
        fp = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.exists() else {}
        for split in SPLITS:
            gold = load_split(split)
            cache = read_cache(cache_file(key, model, split))
            recs = [cache[i] for i in gold["orig_idx"] if i in cache]
            if len(recs) < len(gold):
                status = ("pending: run the Colab notebook" if sc["mode"] == "finetuned" and not recs
                          else f"not run yet ({len(recs)}/{len(gold)} cached)")
                rows.append({"system": key, "label": sc["label"], "model": model, "split": split, "status": status,
                             "n": len(recs)})
                continue
            y, p = [r["label"] for r in recs], [r["prediction"] for r in recs]
            lat = latency_stats([r["t_total"] for r in recs if r.get("t_total") is not None])
            loaded = (fp.get("loaded") or [{}])[0]
            rows.append({"system": key, "label": sc["label"], "model": model, "split": split, "status": "done",
                         **summarize(y, p, labels), **lat,
                         "cost_per_1k_usd": cost_per_1k(lat["mean_ms"] / 1000, cfg["cost"]["cpu_instance_usd_per_hour"]),
                         "size_on_disk_mb": fp.get("disk_size_gb", 0) * 1000 or None,
                         "mem_loaded_mb": loaded.get("size", 0) / 1e6 or None,
                         "vram_mb": loaded.get("size_vram", 0) / 1e6 if loaded else None})
            pcf[f"{key}|{split}"] = per_class_f1(y, p, labels)
            conf[f"{key}|{split}"] = top_confusions(y, p, 15)
    add_new_systems(cfg, labels, rows, pcf, conf)
    df = pd.DataFrame(rows)
    df.to_csv(path("results") / "comparison.csv", index=False)
    pd.DataFrame(pcf).rename_axis("label").to_csv(path("results") / "per_class_f1.csv")
    (path("results") / "confusions.json").write_text(json.dumps(conf, indent=1), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in ["label", "split", "status", "n", "accuracy", "macro_f1", "invalid_rate", "p50_ms",
                               "p95_ms"] if c in df]])


def add_new_systems(cfg: dict, labels: list[str], rows: list, pcf: dict, conf: dict) -> None:
    """(f)-(i) classic baselines and (j) DistilBERT. Missing results -> a 'not run yet' row."""
    for key in classic_keys():
        p = path("results") / f"classic_{key}.json"
        res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        for split in SPLITS:
            if res is None or split not in res:
                rows.append({"system": key, "label": system_label(key), "model": "sklearn", "split": split,
                             "status": f"not run yet: scripts/run_classic.py --system {key}", "n": 0})
                continue
            b = res[split]
            rows.append({"system": key, "label": system_label(key), "model": "sklearn", "split": split,
                         "status": "done", "n": b["n"], "accuracy": b["accuracy"], "macro_f1": b["macro_f1"],
                         "invalid_rate": 0.0, "p50_ms": b["p50_ms"], "p95_ms": b["p95_ms"],
                         "throughput_rps": b["throughput_rps"], "mean_ms": b["mean_ms"],
                         "cost_per_1k_usd": b["cost_per_1k_usd_cpu_assumption"],
                         "size_on_disk_mb": res["model_size_mb"], "latency_device": "cpu",
                         "p50_ms_cpu": b["p50_ms"], "p95_ms_cpu": b["p95_ms"]})
            pcf[f"{key}|{split}"] = b["per_class_f1"]
            conf[f"{key}|{split}"] = b["top_confusions"]

    key = distilbert_key()
    if not key:
        return
    model = system_model(key)
    sp = path("results") / f"{key}.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for split in SPLITS:
        gold = load_split(split)
        cache = read_cache(cache_file(key, model, split))
        recs = [cache[i] for i in gold["orig_idx"] if i in cache]
        if len(recs) < len(gold) or split not in summ:
            status = ("pending: scripts/train_distilbert.py then scripts/eval_distilbert.py" if not recs
                      else f"not run yet ({len(recs)}/{len(gold)} cached)")
            rows.append({"system": key, "label": system_label(key), "model": model, "split": split,
                         "status": status, "n": len(recs)})
            continue
        y, p = [r["label"] for r in recs], [r["prediction"] for r in recs]
        lat = latency_stats([r["t_total"] for r in recs if r.get("t_total") is not None])
        cpu = summ[split]["latency_cpu"]
        rows.append({"system": key, "label": system_label(key), "model": model, "split": split, "status": "done",
                     **summarize(y, p, labels), **lat,
                     # Cost uses the CPU latency: the service (service/, k8s/) runs on CPU.
                     "cost_per_1k_usd": cost_per_1k(cpu["mean_ms"] / 1000, cfg["cost"]["cpu_instance_usd_per_hour"]),
                     "size_on_disk_mb": summ.get("model_size_mb"),
                     "vram_mb": summ.get("peak_gpu_memory_mb"),
                     "latency_device": summ.get("device"), "p50_ms_cpu": cpu["p50_ms"], "p95_ms_cpu": cpu["p95_ms"]})
        pcf[f"{key}|{split}"] = per_class_f1(y, p, labels)
        conf[f"{key}|{split}"] = top_confusions(y, p, 15)


if __name__ == "__main__":
    main()
