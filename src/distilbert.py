"""(j) DistilBERT intent classifier with a hand-written PyTorch training loop (no HF Trainer).

Used by scripts/train_distilbert.py, scripts/eval_distilbert.py, src/systems.py (app) and the FastAPI service
(service/app.py, which copies only this file and src/__init__.py into its Docker image).
So this module must not import src.config or anything else from the project, and torch / transformers are
imported lazily: the pure-Python helpers (schedule, early stopping, label maps) are testable without torch.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

LABEL_MAP_FILE = "label_map.json"


# --------------------------------------------------------------------------------------------------
# Pure-Python helpers (no torch)
# --------------------------------------------------------------------------------------------------
def label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """label -> id and id -> label, in the given (sorted, labels.json) order."""
    l2i = {lab: i for i, lab in enumerate(labels)}
    return l2i, {i: lab for lab, i in l2i.items()}


def linear_warmup_decay(warmup_steps: int, total_steps: int):
    """LR multiplier: 0 -> 1 linearly over warmup_steps, then 1 -> 0 linearly until total_steps."""
    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return f


@dataclass
class EarlyStopping:
    """Stop when the validation loss has not improved by more than min_delta for `patience` evaluations."""
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    best_epoch: int = -1
    bad: int = 0
    history: list = field(default_factory=list)

    def step(self, val_loss: float, epoch: int) -> bool:
        """Record one evaluation; return True if it is a new best (save a checkpoint)."""
        self.history.append(val_loss)
        if val_loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = val_loss, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA). cuDNN deterministic mode trades a little speed for repeatability."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def no_decay(param_name: str) -> bool:
    """Biases and LayerNorm weights get no weight decay (standard BERT recipe)."""
    return param_name.endswith(".bias") or "LayerNorm" in param_name or "layer_norm" in param_name


# --------------------------------------------------------------------------------------------------
# torch parts (lazy imports)
# --------------------------------------------------------------------------------------------------
def encode(tokenizer, texts: list[str], max_len: int):
    """Tokenise to fixed-length tensors (padding to max_len keeps batches uniform on a small GPU)."""
    return tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")


def make_loader(enc, y, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415
    tensors = [enc["input_ids"], enc["attention_mask"]] + ([torch.as_tensor(y, dtype=torch.long)] if y is not None else [])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers)


def make_optimizer(model, lr: float, weight_decay: float):
    import torch  # noqa: PLC0415
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (nodecay if no_decay(n) else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": nodecay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import torch  # noqa: PLC0415
    return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup_steps, total_steps))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device: str, use_amp: bool,
                    max_grad_norm: float, on_step=None) -> float:
    """One pass over `loader` with AMP autocast (fp16, CUDA only), loss scaling and gradient clipping.
    Returns the mean training loss. `on_step(global_step_in_epoch, loss)` is called after every step."""
    import torch  # noqa: PLC0415
    model.train()
    total, n = 0.0, 0
    for i, (ids, mask, y) in enumerate(loader, 1):
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                            enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                      # clip the true (unscaled) gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)                          # skipped automatically if grads are inf/nan
        scaler.update()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
        if on_step:
            on_step(i, loss.item())
    return total / max(1, n)


def evaluate(model, loader, device: str, use_amp: bool) -> tuple[float, list[int]]:
    """Mean cross-entropy and argmax predictions on a labelled loader."""
    import torch  # noqa: PLC0415
    model.eval()
    total, n, preds = 0.0, 0, []
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
            total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            n += len(y)
            preds += logits.argmax(-1).tolist()
    return total / max(1, n), preds


def save_checkpoint(model, tokenizer, out_dir: Path, labels: list[str], meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)      # safetensors weights + config.json (id2label inside)
    tokenizer.save_pretrained(out_dir)
    (out_dir / LABEL_MAP_FILE).write_text(json.dumps({"labels": labels, **meta}, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# Inference wrapper (eval script, app, FastAPI service)
# --------------------------------------------------------------------------------------------------
class DistilBertIntentClassifier:
    """Loads a checkpoint written by save_checkpoint and classifies messages one at a time or in batches."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", max_len: int | None = None,
                 num_threads: int | None = None):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        self.model_dir = Path(model_dir)
        if not (self.model_dir / LABEL_MAP_FILE).exists():
            raise FileNotFoundError(f"No DistilBERT checkpoint in {self.model_dir}. "
                                    "Run: .venv\\Scripts\\python scripts\\train_distilbert.py")
        meta = json.loads((self.model_dir / LABEL_MAP_FILE).read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.max_len = int(max_len or meta.get("max_len", 64))
        if num_threads:
            torch.set_num_threads(int(num_threads))
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(device).eval()
        self._torch = torch

    def size_on_disk_mb(self) -> float:
        return sum(p.stat().st_size for p in self.model_dir.iterdir() if p.is_file()) / 1e6

    def predict(self, texts: list[str], top_k: int = 1) -> list[dict]:
        """[{label, confidence, top_k: [{label, prob}, ...]}] per text. No padding beyond the longest text."""
        torch = self._torch
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
        k = max(1, min(int(top_k), len(self.labels)))
        p, idx = probs.topk(k, dim=-1)
        out = []
        for pr, ix in zip(p.tolist(), idx.tolist()):
            out.append({"label": self.labels[ix[0]], "confidence": pr[0],
                        "top_k": [{"label": self.labels[i], "prob": v} for i, v in zip(ix, pr)]})
        return out

    def timed_predict_one(self, text: str) -> tuple[dict, float]:
        """One request, wall time including tokenisation; CUDA is synchronised so the time is real."""
        t0 = time.perf_counter()
        r = self.predict([text])[0]
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        return r, time.perf_counter() - t0
