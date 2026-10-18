r"""(j) Fine-tune DistilBERT on Banking77 with a hand-written PyTorch loop (no HF Trainer).

AdamW (no decay on bias/LayerNorm), linear warm-up then linear decay, fp16 mixed precision on CUDA
(autocast + GradScaler), gradient clipping, early stopping on validation loss, full seed control.
Trains on train (9,000), early-stops on val (1,003). Test is NOT touched here (scripts/eval_distilbert.py).

Outputs:
  models/distilbert/                      best checkpoint (safetensors, tokenizer, label_map.json); gitignored
  results/distilbert_training_log.json    loss curves, val metrics per epoch, settings, time, GPU, peak memory

Usage (after requirements-torch.txt, see README):
  .venv\Scripts\python scripts\train_distilbert.py
  .venv\Scripts\python scripts\train_distilbert.py --epochs 1 --limit 512     # smoke test (minutes on CPU)
Holds AI_Engineer_Portfolio/.llm_lock while running (one heavy job at a time on the 8 GB laptop).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import distilbert as db  # noqa: E402
from src.config import ROOT, detect_device, load_config, path  # noqa: E402
from src.data import load_labels, load_split  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402
from src.metrics import summarize  # noqa: E402


def train(epochs: int | None = None, limit: int | None = None, device: str | None = None,
          out_dir: Path | None = None, log_path: Path | None = None) -> dict:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    c = load_config()["distilbert"]
    epochs = epochs or c["epochs"]
    device = device or detect_device()
    if device == "mps":
        device = "cpu"  # AMP path is CUDA-only; keep the CPU path simple elsewhere
    use_amp = bool(c["amp"]) and device == "cuda"
    out_dir = Path(out_dir or ROOT / c["out_dir"])
    log_path = Path(log_path or path("results") / "distilbert_training_log.json")
    db.set_seed(c["seed"])

    labels = load_labels()
    l2i, i2l = db.label_maps(labels)
    train_df, val_df = load_split("train"), load_split("val")
    if limit:
        train_df, val_df = train_df.head(limit), val_df.head(max(64, limit // 4))
    tok = AutoTokenizer.from_pretrained(c["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        c["base_model"], num_labels=len(labels), id2label=i2l, label2id=l2i).to(device)

    # Truncation check: how many messages are longer than max_len tokens?
    lengths = [len(x) for x in tok(list(train_df["text"]), truncation=False)["input_ids"]]
    n_trunc = sum(n > c["max_len"] for n in lengths)
    print(f"device={device} amp={use_amp} train={len(train_df)} val={len(val_df)} "
          f"max_len={c['max_len']} (truncates {n_trunc} train messages; longest {max(lengths)} tokens)", flush=True)

    tr_enc, va_enc = db.encode(tok, train_df["text"], c["max_len"]), db.encode(tok, val_df["text"], c["max_len"])
    y_tr, y_va = [l2i[x] for x in train_df["label"]], [l2i[x] for x in val_df["label"]]
    tr_loader = db.make_loader(tr_enc, y_tr, c["batch_size"], True, c["seed"], c["num_workers"])
    va_loader = db.make_loader(va_enc, y_va, c["eval_batch_size"], False, c["seed"], c["num_workers"])

    opt = db.make_optimizer(model, c["lr"], c["weight_decay"])
    total_steps = epochs * len(tr_loader)
    warmup = int(round(c["warmup_ratio"] * total_steps))
    sched = db.make_scheduler(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = db.EarlyStopping(patience=c["early_stopping_patience"], min_delta=c["min_delta"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_curve, val_curve, epochs_log = [], [], []
    state = {"step": 0}

    def on_step(_i, loss):
        state["step"] += 1
        if state["step"] % 25 == 0:
            train_curve.append({"step": state["step"], "loss": loss, "lr": sched.get_last_lr()[0]})

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        te = time.perf_counter()
        tr_loss = db.train_one_epoch(model, tr_loader, opt, sched, scaler, device, use_amp, c["max_grad_norm"],
                                     on_step)
        va_loss, va_pred = db.evaluate(model, va_loader, device, use_amp)
        m = summarize(list(val_df["label"]), [i2l[i] for i in va_pred], labels)
        improved = stopper.step(va_loss, epoch)
        val_curve.append({"step": state["step"], "loss": va_loss})
        epochs_log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_accuracy": m["accuracy"],
                           "val_macro_f1": m["macro_f1"], "seconds": time.perf_counter() - te, "best": improved})
        print(f"epoch {epoch}: train {tr_loss:.4f}  val {va_loss:.4f}  val acc {m['accuracy']:.4f}  "
              f"macroF1 {m['macro_f1']:.4f}  {'*saved*' if improved else f'no improvement ({stopper.bad})'}",
              flush=True)
        if improved:
            db.save_checkpoint(model, tok, out_dir, labels,
                               {"max_len": c["max_len"], "base_model": c["base_model"], "epoch": epoch,
                                "val_loss": va_loss, "val_accuracy": m["accuracy"]})
        if stopper.should_stop:
            print(f"early stopping after epoch {epoch} (best epoch {stopper.best_epoch})", flush=True)
            break
    train_s = time.perf_counter() - t0
    best = next(e for e in epochs_log if e["epoch"] == stopper.best_epoch)
    log = {
        "system": c["system_key"], "base_model": c["base_model"], "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "train_seconds": train_s, "gpu_hours": train_s / 3600 if device == "cuda" else 0.0,
        "epochs_completed": len(epochs_log), "best_epoch": stopper.best_epoch, "steps": state["step"],
        "val_loss_best": stopper.best, "val_accuracy": best["val_accuracy"], "val_macro_f1": best["val_macro_f1"],
        "n_train": len(train_df), "n_val": len(val_df), "truncated_train_messages": n_trunc,
        "smoke_test": bool(limit), "epochs": epochs_log, "train_loss": train_curve, "val_loss": val_curve,
        "settings": {k: c[k] for k in ("max_len", "batch_size", "lr", "weight_decay", "warmup_ratio",
                                        "max_grad_norm", "early_stopping_patience", "seed", "amp")}
                    | {"epochs_max": epochs, "warmup_steps": warmup, "total_steps_planned": total_steps,
                       "optimizer": "AdamW", "schedule": "linear warm-up + linear decay"},
        "checkpoint_dir": str(out_dir.relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir),
    }
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"best epoch {stopper.best_epoch}: val loss {stopper.best:.4f}, val acc {best['val_accuracy']:.4f}; "
          f"{train_s / 60:.1f} min; checkpoint -> {out_dir}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override config (max epochs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test on the first N train rows; writes to a separate dir and log")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    kw = {}
    if a.limit:  # never overwrite the real checkpoint/log with a smoke test
        kw = {"out_dir": ROOT / "models" / "distilbert_smoke",
              "log_path": path("results") / "distilbert_training_log_smoke.json"}
    if a.no_lock:
        train(a.epochs, a.limit, a.device, **kw)
    else:
        with llm_lock("train distilbert"):
            train(a.epochs, a.limit, a.device, **kw)


if __name__ == "__main__":
    main()
