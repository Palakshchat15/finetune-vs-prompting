r"""Generate notebooks/finetune_lora_colab.ipynb from the cell sources below.

The notebook embeds verbatim copies of FINETUNE_SYSTEM (src/prompts.py) and the split logic
(src/data.py); tests/test_notebook.py checks they stay identical. Re-run after editing either:
    .venv\Scripts\python scripts\build_notebook.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nbformat  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.prompts import FINETUNE_SYSTEM  # noqa: E402

cfg = load_config()
ft = cfg["finetune"]
data = cfg["data"]

INTRO = f"""# Banking77: LoRA fine-tune of Qwen2.5-1.5B-Instruct (Colab, free T4)

**What this does:** fine-tunes `{ft['base_model']}` with LoRA to output the exact Banking77 intent label,
checks accuracy on the validation split, and exports (1) the LoRA adapter, (2) a merged GGUF Q4_K_M model for Ollama,
(3) `training_log.json` (loss curve, validation accuracy, settings, time, GPU).

## Steps for the owner (plain English)
1. **Open in Colab:** go to https://colab.research.google.com -> File -> Upload notebook -> choose
   `notebooks/finetune_lora_colab.ipynb` from the project folder.
2. **Pick the GPU:** Runtime -> Change runtime type -> **T4 GPU** -> Save.
3. **Run everything:** Runtime -> Run all. Nothing needs editing. If Colab asks "Restart session?" after installing
   packages, click Restart, then Runtime -> Run all again.
4. **Download the outputs:** the last cell downloads `banking77_outputs.zip` automatically (allow the browser download).
   If it does not, open the Files panel (folder icon, left) and download `/content/banking77_outputs.zip`.
5. **Put them in `models/`:** unzip into `AI_Engineer_Portfolio/finetune_vs_prompting/models/` so that
   `models/{ft['gguf_file']}`, `models/lora_adapter/` and `models/training_log.json` exist.
   Then on the laptop run `scripts/import_finetuned.ps1` and `scripts/eval_finetuned.py` (see PROGRESS.md).

**Expected runtime on a free T4:** about 5 min installs, 20-35 min training (3 epochs max, early stopping),
2-3 min validation, 10-15 min merge + GGUF conversion. **Total about 45-60 min.** Keep the tab open.

**Settings (documented):** 4-bit NF4 loading (QLoRA), LoRA r={ft['lora_r']}, alpha={ft['lora_alpha']},
dropout={ft['lora_dropout']}, all attention + MLP projections, lr={ft['lr']}, batch {ft['batch_size']},
max {ft['epochs']} epochs, evaluation every half epoch on validation loss, early stopping patience
{ft['early_stopping_patience']}, best checkpoint kept. Loss is computed on the answer tokens only
(the prompt is masked). The prompt is the same ChatML format the local Ollama model uses.
"""

SETUP = """# Install pinned-ish libraries (Colab already has torch + CUDA).
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
# DRY_RUN=1 runs a tiny model for a few CPU steps to catch bugs (used on the build laptop). Leave 0 on Colab.
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.44,<5", "peft>=0.12", "accelerate>=0.33", "bitsandbytes>=0.43",
                    "datasets", "gguf", "sentencepiece", "protobuf"], check=True)
    # Colab ships an old torchao that newer peft rejects at merge time ("incompatible version of torchao").
    # Nothing here uses torchao, so remove it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
"""

CONFIG = f"""# Settings (mirrors config.yaml -> finetune). Edit here only if you know why.
BASE_MODEL = {ft['base_model']!r}
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"   # DRY_RUN only
LORA_R, LORA_ALPHA, LORA_DROPOUT = {ft['lora_r']}, {ft['lora_alpha']}, {ft['lora_dropout']}
MAX_SEQ_LEN = {ft['max_seq_len']}
EPOCHS, LR, BATCH, GRAD_ACCUM = {ft['epochs']}, {ft['lr']}, {ft['batch_size']}, {ft['grad_accum']}
PATIENCE = {ft['early_stopping_patience']}
SEED = {data['seed']}
VAL_FRAC = {data['val_frac']}
GGUF_NAME = {ft['gguf_file']!r}
TRAIN_URL = {data['train_url']!r}
TEST_URL = {data['test_url']!r}
EXPECTED_VAL_FINGERPRINT = "5143ce10cc79c0b7"   # sha256 of sorted val row ids, computed by src/data.py locally
OUT = "/content/outputs" if IN_COLAB else os.path.abspath("nb_outputs")
os.makedirs(OUT, exist_ok=True)
"""

PROMPT = f"""# Prompt format: a verbatim copy of src/prompts.py (tests check they match).
FINETUNE_SYSTEM = {FINETUNE_SYSTEM!r}


def finetune_messages(text, label=None):
    msgs = [{{"role": "system", "content": FINETUNE_SYSTEM}}, {{"role": "user", "content": text}}]
    if label is not None:
        msgs.append({{"role": "assistant", "content": label}})
    return msgs
"""

DATA = """# Data prep: download the original PolyAI CSVs and rebuild the SAME stratified split as src/data.py.
import hashlib, io, json, time
import pandas as pd
import requests

UA = {"User-Agent": "banking77-portfolio-downloader/1.0"}


def load_csv(url):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)).rename(columns={"category": "label"})
    df["text"] = df["text"].astype(str).str.strip()
    return df[["text", "label"]]


def stratified_split(df, frac, seed):
    held = []
    for _, g in df.groupby("label", sort=True):
        n = max(1, int(round(frac * len(g))))
        held.append(g.sample(n=n, random_state=seed))
    held_df = pd.concat(held).sort_index()
    rest = df.drop(index=held_df.index)
    return rest, held_df


train_full = load_csv(TRAIN_URL)
train_df, val_df = stratified_split(train_full, VAL_FRAC, SEED)
LABELS = sorted(train_full["label"].unique())
fp = hashlib.sha256(",".join(map(str, sorted(val_df.index))).encode()).hexdigest()[:16]
print(f"train {len(train_df)}  val {len(val_df)}  labels {len(LABELS)}  val fingerprint {fp}")
assert len(LABELS) == 77 and len(train_df) == 9000 and len(val_df) == 1003
assert fp == EXPECTED_VAL_FINGERPRINT, "validation split differs from the local one"
if DRY_RUN:
    train_df, val_df = train_df.sample(64, random_state=SEED), val_df.sample(16, random_state=SEED)
"""

MODEL = """# Load the base model (4-bit on GPU) and attach LoRA.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none (CPU)"
print("device:", DEVICE, GPU_NAME)
if IN_COLAB and DEVICE != "cuda" and not DRY_RUN:
    raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU, then Run all again.")
torch.manual_seed(SEED)
model_id = TINY_MODEL if DRY_RUN else BASE_MODEL
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if DEVICE == "cuda":
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
"""

TOKENIZE = """# Tokenize: prompt tokens are masked (-100); the loss is on the label + end-of-turn token only.
END = "<|im_end|>"


def encode(text, label):
    prompt = tok.apply_chat_template(finetune_messages(text), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(label + END + "\\n", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_SEQ_LEN]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class ListDS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.items = [encode(t, l) for t, l in zip(df["text"], df["label"])]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
    }


train_ds, val_ds = ListDS(train_df), ListDS(val_df)
lens = [len(x["input_ids"]) for x in train_ds.items]
print("example prompt:\\n" + tok.decode(train_ds.items[0]["input_ids"]))
print(f"sequence length max {max(lens)} mean {sum(lens)/len(lens):.1f} (truncated at {MAX_SEQ_LEN}: {sum(l >= MAX_SEQ_LEN for l in lens)})")
"""

TRAIN = """# Train with early stopping on validation loss (evaluated every half epoch); keep the best checkpoint.
import math
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

steps_per_epoch = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM))
eval_every = max(1, steps_per_epoch // 2)
args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=32, gradient_accumulation_steps=GRAD_ACCUM,
    lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
    eval_strategy="steps", eval_steps=eval_every, save_strategy="steps", save_steps=eval_every,
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=10, fp16=(DEVICE == "cuda"), report_to="none", seed=SEED,
    max_steps=3 if DRY_RUN else -1, remove_unused_columns=False, dataloader_num_workers=0,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
t0 = time.time()
train_result = trainer.train()
train_seconds = time.time() - t0
print(f"training took {train_seconds/60:.1f} min; best checkpoint: {trainer.state.best_model_checkpoint}")
"""

EVAL = """# Validation accuracy: greedy generation, exact-match against the 77 label strings.
model.eval()
tok.padding_side = "left"
label_set = set(LABELS)
preds, t0 = [], time.time()
texts = list(val_df["text"])
for i in range(0, len(texts), 32):
    chunk = texts[i:i + 32]
    prompts = [tok.apply_chat_template(finetune_messages(t), tokenize=False, add_generation_prompt=True) for t in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    for row in out[:, enc["input_ids"].shape[1]:]:
        preds.append(tok.decode(row, skip_special_tokens=True).strip())
val_eval_seconds = time.time() - t0
gold = list(val_df["label"])
val_acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
invalid = sum(p not in label_set for p in preds) / len(preds)
print(f"validation accuracy {val_acc:.4f}  invalid-label rate {invalid:.4f}  ({val_eval_seconds:.0f}s)")
pd.DataFrame({"text": texts, "label": gold, "pred": preds}).to_csv(f"{OUT}/val_predictions.csv", index=False)
tok.padding_side = "right"
"""

EXPORT = """# Save the LoRA adapter and training_log.json.
adapter_dir = f"{OUT}/lora_adapter"
model.save_pretrained(adapter_dir)
tok.save_pretrained(adapter_dir)
hist = trainer.state.log_history
log = {
    "base_model": model_id, "dry_run": DRY_RUN, "gpu": GPU_NAME, "device": DEVICE,
    "settings": {"quantization": "4-bit NF4 (QLoRA)" if DEVICE == "cuda" else "fp32 (CPU dry run)",
                 "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                 "target_modules": sorted(lora.target_modules), "lr": LR, "batch_size": BATCH,
                 "grad_accum": GRAD_ACCUM, "max_epochs": EPOCHS, "max_seq_len": MAX_SEQ_LEN,
                 "early_stopping_patience": PATIENCE, "eval_every_steps": eval_every, "seed": SEED,
                 "system_prompt": FINETUNE_SYSTEM},
    "n_train": len(train_ds), "n_val": len(val_ds),
    "train_seconds": train_seconds, "gpu_hours": train_seconds / 3600,
    "epochs_completed": trainer.state.epoch, "global_steps": trainer.state.global_step,
    "best_checkpoint": trainer.state.best_model_checkpoint, "best_val_loss": trainer.state.best_metric,
    "train_loss": [{"step": h["step"], "epoch": h.get("epoch"), "loss": h["loss"]} for h in hist if "loss" in h],
    "val_loss": [{"step": h["step"], "epoch": h.get("epoch"), "loss": h["eval_loss"]} for h in hist if "eval_loss" in h],
    "val_accuracy": val_acc, "val_invalid_rate": invalid, "val_eval_seconds": val_eval_seconds,
    "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else None,
}
with open(f"{OUT}/training_log.json", "w") as f:
    json.dump(log, f, indent=1)
print(json.dumps({k: v for k, v in log.items() if k not in ("train_loss", "val_loss")}, indent=1))
"""

GGUF = """# Merge LoRA into the fp16 base and convert to GGUF Q4_K_M with llama.cpp (skipped in DRY_RUN).
if not DRY_RUN:
    import gc, shutil
    from peft import PeftModel
    for _name in ("model", "trainer"):   # free GPU memory; safe to re-run this cell
        globals().pop(_name, None)
    gc.collect(); torch.cuda.empty_cache()
    adapter_dir = f"{OUT}/lora_adapter"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged_dir = "/content/merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(merged_dir)
    del merged, base; gc.collect()
    if not os.path.exists("/content/llama.cpp"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], check=True)
    # No `pip install -r requirements-convert...`: it can reinstall a CPU torch. The converter uses the repo's own
    # gguf-py; transformers, safetensors, sentencepiece and numpy are already installed.
    f16 = "/content/model-f16.gguf"
    subprocess.run([sys.executable, "/content/llama.cpp/convert_hf_to_gguf.py", merged_dir, "--outfile", f16, "--outtype", "f16"], check=True)
    subprocess.run("cd /content/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF && cmake --build build --target llama-quantize -j 4",
                   shell=True, check=True)
    subprocess.run(["/content/llama.cpp/build/bin/llama-quantize", f16, f"{OUT}/{GGUF_NAME}", "Q4_K_M"], check=True)
    print("GGUF size (MB):", os.path.getsize(f"{OUT}/{GGUF_NAME}") / 1e6)
"""

DOWNLOAD = """# Zip the outputs (adapter, GGUF, training_log.json, val_predictions.csv) and download.
import shutil
shutil.rmtree(f"{OUT}/checkpoints", ignore_errors=True)
zip_path = shutil.make_archive("/content/banking77_outputs" if IN_COLAB else f"{OUT}_zip", "zip", OUT)
print("wrote", zip_path, os.path.getsize(zip_path) / 1e6, "MB")
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
"""

cells = [nbformat.v4.new_markdown_cell(INTRO)]
for src in (SETUP, CONFIG, PROMPT, DATA, MODEL, TOKENIZE, TRAIN, EVAL, EXPORT, GGUF, DOWNLOAD):
    cells.append(nbformat.v4.new_code_cell(src))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
               "kernelspec": {"display_name": "Python 3", "name": "python3"},
               "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "finetune_lora_colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbformat.write(nb, out)
print("wrote", out)
