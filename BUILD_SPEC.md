# Build spec: fine-tuning vs prompting (LoRA on Colab)

Follow `../CONVENTIONS.md` for all shared rules.

## Goal
Show when fine-tuning a small open LLM beats prompting a larger one, on accuracy, latency, memory and cost, for a real classification task.

**Training runs on Google Colab's free GPU.** The owner runs the notebook, because this laptop has no GPU and nothing may run on the owner's Google account unattended. Build **everything else** unattended, and make the notebook one-click.

## Data
- **Banking77** (PolyAI): 13,083 customer-support banking messages, 77 intents. Train 10,003 / test 3,080. Hugging Face `PolyAI/banking77`, or the original CSVs from the PolyAI task-specific-datasets GitHub if HF has issues.
- **Check the licence** (expected CC BY 4.0) and attribute it.
- **Validation:** hold out a stratified 10% of train for tuning and early stopping. Report the full test set.

## Systems to compare
| System | What it is |
|---|---|
| (a) TF-IDF + logistic regression | classic baseline, CPU, seconds |
| (b) Zero-shot `qwen2.5:7b` | the 77 label names in the prompt; Ollama structured output with an enum of labels |
| (c) Few-shot `qwen2.5:7b` | e.g. 1-2 examples per label, or retrieved nearest examples (kNN few-shot); document which |
| (d) Base Qwen2.5-1.5B-Instruct zero-shot | shows the fine-tuning gain on the same model |
| (e) **LoRA fine-tuned Qwen2.5-1.5B-Instruct** | trained on Colab, exported to GGUF (Q4_K_M), served locally via `ollama create` |

- If CPU time for (b)-(d) on the full test set is prohibitive, measure it first. Then use a **stratified test subset** (e.g. 10 per class = 770) for the slow LLM systems, run the fast systems on both the subset and the full set, and state this clearly.
- Use Ollama prompt-prefix caching (a fixed system prompt) to speed things up.

## Colab notebook (`notebooks/finetune_lora_colab.ipynb`)
- Works on a free T4: Unsloth or HF PEFT/TRL, 4-bit loading, LoRA r=16 (document the settings).
- 1-3 epochs with early stopping on validation.
- Uses the **same prompt format** as local inference, and the target is the exact label string.
- Evaluates on validation in the notebook.
- Exports:
  - merged model to GGUF Q4_K_M;
  - the LoRA adapter;
  - `training_log.json` (loss curve, validation accuracy, settings, time, GPU).
- **Top of the notebook:** plain-English step-by-step instructions for the owner:
  1. Open in Colab.
  2. Runtime -> T4 GPU.
  3. Run all.
  4. Download the outputs.
  5. Put them in `models/`.

  Expected runtime included.
- **Validate it as far as possible without a GPU:**
  - Run the data-prep cells locally.
  - Syntax-check the whole notebook.
  - Dry-run the training loop on a tiny CPU model and a few steps, if feasible, to catch bugs.
- **Local after-steps as scripts:**
  - `scripts/import_finetuned.ps1`: writes the Ollama Modelfile and runs `ollama create`, using the same prompt template;
  - `scripts/eval_finetuned.py`.

## Metrics
- Accuracy and macro-F1 (full test, or the subset where noted); per-class F1; the confusion between the most-confused intents.
- Latency p50/p95 per request on this CPU; throughput; model RAM/VRAM footprint; model size on disk.
- **Cost per 1,000 requests,** estimated from measured CPU-seconds, and a cloud GPU price assumption clearly labelled as an assumption.
- Training cost: Colab GPU time from the training log.
- Invalid-label rate (outputs not among the 77 labels) for the prompted systems.

## Streamlit app
1. **Classify:** type a message and see the prediction and latency from every available system side by side.
2. **Evaluation:** the comparison table (accuracy, macro-F1, latency, cost, size), an accuracy-vs-latency chart, per-class F1 and confusions.
3. **Training:** the loss curve and settings from `training_log.json`, once available.
4. **Monitoring:** traces.

Before the fine-tuned model exists, the app and README show (e) as "pending: run the Colab notebook". Everything else is complete.

## PROGRESS.md
The final state must say clearly: "Waiting for owner: run `notebooks/finetune_lora_colab.ipynb` on Colab, then run `scripts/import_finetuned.ps1` and `scripts/eval_finetuned.py`", followed by the exact steps.
