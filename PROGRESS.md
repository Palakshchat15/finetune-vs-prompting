# PROGRESS: fine-tuning vs prompting (Banking77, LoRA)

## UPGRADE RUN (2026-10-18): done on the GPU laptop (logs: _work_in_progress/upgrades/)
- Every step finished with rc=0 (`run_ai.log`): requirements-nlp, `run_classic.py`, CUDA torch 2.14.1+cu126 +
  requirements-torch, DistilBERT smoke test, full training, `eval_distilbert.py`, summarize / error analysis /
  significance, requirements-service. pytest: 59 passed.
- Classic baselines, full test acc / macro-F1: (f) nb_tfidf 0.861 / 0.860, (g) svm_tfidf 0.887 / 0.887,
  (h) svm_char 0.902 / 0.902, (i) spacy_lemma_svm 0.898 / 0.898 (best on val without stop-word removal).
- (j) distilbert_pt: early stopping after epoch 8, best epoch 6 (val loss 0.316, val acc 0.9242), 4.0 min on the
  RTX 3050, peak GPU memory 1,388 MB. Full test 0.928 / 0.928 (subset 0.913 / 0.913); p50 5.1 ms GPU, 21.2 ms CPU.
- McNemar: DistilBERT vs TF-IDF p=2.6e-13; LoRA 1.5B vs DistilBERT p=0.83 (tie). README headline updated to say so.
- Kubernetes (Docker Desktop, v1.36.1, kind-based, node desktop-control-plane): image built, pod ready (model load
  8.1 s), /predict OK. Load test c=4, 500 requests: 21.0 req/s, p50 157 ms, p95 269 ms, 0 errors. HPA test c=8 for
  180 s: 23.9 req/s, p50 323 ms, p95 398 ms, 0 errors; CPU 366%/70%, scaled 1 -> 2 replicas at about 7.5 min after
  deploy; pod memory 675 MiB (limit 1280Mi). Deployment deleted afterwards.
- README "Upgrade" section filled in from the result files.
- **Still open:** retake the app screenshots with the new rows; Docker image size not recorded.

## UPGRADE (2026-10-17): code written (run on 2026-10-18, see above; details in UPGRADE_NOTES.md)
- Added (f)-(i) classic NLP baselines (`nb_tfidf`, `svm_tfidf`, `svm_char`, `spacy_lemma_svm`), (j) `distilbert_pt`
  (hand-written PyTorch loop), and a FastAPI service + Dockerfile + k8s/ manifests + load test. All are registered in
  config.yaml and picked up by summarize_results.py, error_analysis.py, significance.py and the app.
- Existing systems, caches and numbers are untouched. pytest: 54 passed, 5 skipped (torch/fastapi tests skip until installed).
- Pins checked on PyPI, not installed: spacy 3.8.16 + en_core_web_sm 3.8.0, torch 2.14.1 (cu126 / cpu index),
  transformers 5.20.1, fastapi 0.143.3.
- **Next (in order):** `pip install -r requirements-nlp.txt` -> `scripts\run_classic.py` -> CUDA torch +
  `requirements-torch.txt` -> `scripts\train_distilbert.py --epochs 1 --limit 512` (smoke) -> `scripts\train_distilbert.py`
  -> `scripts\eval_distilbert.py` -> `summarize_results.py`, `error_analysis.py`, `significance.py` -> pytest -> fill README
  "not run yet" cells and retake screenshots -> Docker build + Kubernetes deploy + load test (README section).

## COMPLETE (2026-10-02 ~14:45 IST)
- (e) trained on Colab T4 by the owner: 33.6 min, early-stopped at epoch 2.5 (best epoch 2.0), val acc 0.925.
  The merge cell first failed on Colab's old torchao vs newer peft; fixed with `pip uninstall torchao` (now in the
  notebook's setup cell; the merge cell is now re-runnable).
- import_finetuned.ps1 OK (smoke test card_arrival); eval_finetuned.py on subset + full test.
- (e) full test 0.927 / 0.929 macro-F1, p50 79 ms, 0.4% invalid labels. Exact McNemar vs TF-IDF p=7.7e-11,
  vs few-shot 7B p=5.5e-12; few-shot 7B vs TF-IDF p=0.73.
- README final (no placeholders), Training/Evaluation/Classify screenshots retaken, Training cost caption
  fixed ($ was rendered as LaTeX). pytest 27 passed. App RAM about 230 MB.

## Earlier state (2026-10-02 ~08:45 IST, GPU laptop: RTX 3050 Laptop 4 GB VRAM, 8 GB RAM)

Done on the GPU laptop (session 2):
- venv recreated on Python 3.14.0; pytest 27 passed. Models pulled: qwen2.5:7b, :3b, :1.5b.
- (b)-(d) run on test_subset (770) AND full test (3,080) via Task Scheduler. Speed (~1.8 s/item for 7B)
  made the full test affordable. Full test acc / macro-F1: zero_shot_7b 0.626 / 0.597, few_shot_7b
  0.888 / 0.888, zero_shot_3b 0.531 / 0.511, base_1p5b 0.412 / 0.380. Invalid labels 0%.
- Optional val_subset prompt tuning skipped (prompts unchanged; test never used for tuning).
- summarize_results.py + error_analysis.py run; README filled for (a)-(d); app re-screenshotted
  (Evaluation caption updated to GPU-laptop latencies). App RAM about 230 MB.
- Remaining after training: import, eval_finetuned.py (subset + full), README (e) row + training cost, Training page screenshot.

### Owner steps for (e)
1. Go to https://colab.research.google.com and choose File -> Upload notebook -> `notebooks/finetune_lora_colab.ipynb`.
2. Runtime -> Change runtime type -> **T4 GPU** -> Save.
3. Runtime -> **Run all**. It takes about 45-60 min in total (an estimate; not yet measured on a T4). If asked to restart after the installs, restart and Run all again.
4. The last cell downloads `banking77_outputs.zip`. If it doesn't, get it from the Files panel at `/content/banking77_outputs.zip`.
5. Unzip it into `AI_Engineer_Portfolio/finetune_vs_prompting/models/`. You should then have `models/banking77-qwen1.5b-lora.Q4_K_M.gguf`, `models/lora_adapter/`, `models/training_log.json` and `models/val_predictions.csv`.
6. With Ollama running:
   ```powershell
   cd AI_Engineer_Portfolio\finetune_vs_prompting
   powershell -ExecutionPolicy Bypass -File scripts\import_finetuned.ps1   # writes models\Modelfile, runs ollama create, smoke test
   .venv\Scripts\python scripts\eval_finetuned.py                         # subset + full test, then summarize + error analysis
   ```
   (Option from PLAN.md 5.4: on a GPU laptop with 8 GB or more of VRAM, you could train locally instead. The notebook runs outside Colab too, since `IN_COLAB` is False there, but it was only dry-run on CPU.)

## Done (session 1)
1. **venv** `.venv` (Python 3.13), pinned `requirements.txt` (pip freeze). No torch in the project venv; torch is needed only in Colab.
2. **Data:** `scripts/prepare_data.py`
   - Downloaded the original PolyAI CSVs: train 10,003 / test 3,080, 77 labels. The licence is **CC BY 4.0**, confirmed from the HF card saved to `data/raw/hf_dataset_card.md`.
   - Wrote `data/splits/` with seed 42:
     - train 9,000;
     - val 1,003 (stratified 10%);
     - test 3,080;
     - test_subset 770 (10 per class);
     - val_subset 154 (2 per class, for prompt tuning).
   - Checks: `split_summary.json` shows no train/val text overlap; 6 test texts also appear in train (a property of the dataset).
3. **(a) TF-IDF + LogReg:** `scripts/run_baseline.py` -> `results/baseline_tfidf.json`, `results/predictions/tfidf_*.csv`, `models/tfidf_logreg.joblib`.
   - C was chosen on val from {1, 3, 10, 30, 100, 300, 1000}: C=100 (val acc 0.8883 / macro-F1 0.8863). The grid was widened because 100 was the edge of the first grid.
   - **Full test: acc 0.8903, macro-F1 0.8905.** **Subset: acc 0.8753, macro-F1 0.8762.**
   - Latency p50 1.3 ms / p95 1.7 ms per request (build laptop CPU); model 14 MB; fit about 24 s.
4. **Error analysis:** `scripts/error_analysis.py` -> `results/error_buckets.json`. For (a) on the full test, 338 errors: 182 same-topic, 156 different-topic.
5. **Prompts:** `src/prompts.py`.
   - Zero-shot and few-shot share a fixed system prompt holding the 77 labels (about 530 tokens), so Ollama can cache the prefix.
   - They use structured output, a JSON enum of the labels.
   - Few-shot uses kNN: the 10 nearest train messages by TF-IDF cosine.
   - The fine-tune uses a short system prompt, ChatML, and the exact label as its target.
6. **Colab notebook:** `notebooks/finetune_lora_colab.ipynb`, generated by `scripts/build_notebook.py`. Re-run that script after changing prompts or config.
   - Setup: QLoRA NF4, r=16, early stopping on val loss.
   - It computes val accuracy and exports the adapter, a GGUF Q4_K_M (via llama.cpp) and `training_log.json`.
   - Validated without a GPU:
     - all cells parse;
     - the data-prep cells ran locally and reproduced the exact val split (fingerprint `5143ce10cc79c0b7`, asserted in the notebook);
     - a **full CPU dry run** (`DRY_RUN=1`, tiny Qwen2 model, 3 steps, torch 2.14 CPU / transformers 4.57.6 / peft 0.21.1) ran every cell except the GGUF cell (which is skipped in a dry run) and produced a `training_log.json` and a zip;
     - the tokenizer's chat template produced exactly the ChatML that `src/prompts.render_chatml` / the Ollama template produce.
     - Sequences: max 114, mean 57 tokens.
   - **Not validated:** 4-bit bitsandbytes on a T4, and the llama.cpp GGUF conversion and quantize step. Both need Colab.
7. **Eval scripts (not run):**
   - `scripts/eval_llm.py` (any system/split). It caches per item in `data/cache/*.jsonl`, resumes, holds `.llm_lock`, sends a warm-up request excluded from latency, and records the footprint from `/api/ps` and `/api/tags`.
   - `scripts/eval_finetuned.py`, `scripts/summarize_results.py` (already run: comparison with (a) done and the others "not run yet" / "pending") and `scripts/import_finetuned.ps1` (parse-checked; its config-reading line was tested).
8. **Tests:** `pytest`, 27 passing. They cover the splits, prompts/parsers, metrics, the lock (including waiting for another project and releasing on failure), cache resume after a crash, notebook/prompt consistency, and the systems.
9. **App:** `streamlit_app.py` with Classify / Evaluation / Training / Monitoring pages.
   - All 4 pages were screenshotted (`results/screenshots/`) and checked. The app uses about 215 MB RAM.
   - (d) shows "model not pulled"; (e) shows "pending: run the Colab notebook". No LLM call was made.
10. **README.md draft:** results are marked "not run yet" where applicable.
11. **Finding:** use `127.0.0.1`, not `localhost`, in config. Python's localhost lookup on Windows cost about 2 s per request (measured 2.4 s vs 0.6 s for a model-list call).

## Next (in order; the GPU laptop per PLAN.md)
1. **Setup:** recreate the venv (`python -m venv .venv; .venv\Scripts\pip install -r requirements.txt`), then `ollama pull qwen2.5:7b` and `ollama pull qwen2.5:1.5b`. (qwen2.5:1.5b is **not** pulled on the build laptop.) Delete any copied `.llm_lock`.
2. **Speed probe** (measure first, per BUILD_SPEC):
   `.venv\Scripts\python scripts\eval_llm.py --system zero_shot_7b --split test_subset --limit 20`
   The cache is reused later, so nothing is wasted. On a GPU the full test (3,080) may be affordable; decide from the s/item figure.
3. **Prompt tuning on dev only**, if wanted:
   `... eval_llm.py --system few_shot_7b --split val_subset` (and zero_shot_7b, base_1p5b). Adjust `src/prompts.py` or `few_shot.k` in config, then delete the val_subset caches and re-run. Never tune on test.
4. **Test runs:**
   `... eval_llm.py --system zero_shot_7b --split test_subset`, then `few_shot_7b`, then `base_1p5b` (add `--split test` too if affordable).
5. **Colab training** (owner steps above), then `import_finetuned.ps1` and `eval_finetuned.py`.
6. **Results:** `scripts\summarize_results.py` and `scripts\error_analysis.py`. Fill in the README tables, the interpretation, the LLM error buckets, and the footprints (RAM/VRAM, disk).
   - Cost: the measured s/request x the assumed price in config (label GPU-laptop timings as such).
   - Training cost from `training_log.json`.
   - Re-screenshot the app.
7. **Also run (b2) zero-shot qwen2.5:3b** (system `zero_shot_3b`, already in config.yaml; CONVENTIONS asks for 7B and 3B): `... eval_llm.py --system zero_shot_3b --split test_subset`.
