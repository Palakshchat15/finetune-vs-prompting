# Upgrade notes (2026-10-17): classic NLP, PyTorch DistilBERT, FastAPI + Kubernetes

**Run on 2026-10-18: done.** All steps succeeded (pytest 59 passed). Classic (f)-(i) 0.861-0.902 test accuracy;
DistilBERT (j) 0.928 (4.0 min training, best epoch 6), tied with the LoRA 1.5B (0.927, McNemar p=0.83) and above
TF-IDF (p=2.6e-13). Kubernetes: pod ready, 0 errors in both load tests (c=4: 21.0 req/s, p95 269 ms; c=8: 23.9 req/s,
p95 398 ms), HPA scaled 1 -> 2. Results are in the README "Upgrade" section. The status paragraph below is the
pre-run state.

**Status: code only, nothing run yet.** No training, evaluation, Docker or Kubernetes command has been run, and
no new package was installed. The existing six systems, their caches and every number in `results/` and the
README are unchanged. The new systems only add rows.

## What was added
1. **Classic NLP baselines** (`src/classic.py`, `scripts/run_classic.py`), registered in `config.yaml -> classic`:
   - `nb_tfidf`: (f) word TF-IDF + multinomial Naive Bayes;
   - `svm_tfidf`: (g) word TF-IDF + linear SVM;
   - `svm_char`: (h) char_wb 2-5-gram TF-IDF + linear SVM;
   - `spacy_lemma_svm`: (i) spaCy `en_core_web_sm` lemmas, with stop-word removal on/off chosen on val, + linear SVM.

   Hyperparameters are chosen on val only, then the model is refit on train and test is scored once. The outputs
   are the same as for (a): `results/classic_<key>.json`, `results/predictions/<key>_<split>.csv` and
   `models/<key>.joblib`.
2. **(j) DistilBERT with a hand-written PyTorch loop** (`src/distilbert.py`, `scripts/train_distilbert.py`,
   `scripts/eval_distilbert.py`), registered in `config.yaml -> distilbert`. The loop has:
   - AdamW, linear warm-up then decay;
   - fp16 AMP on CUDA, gradient clipping;
   - early stopping on val loss and seed control;
   - best checkpoint saved to `models/distilbert/` (gitignored by `models/*`);
   - max_len 64, batch 32.

   Evaluation writes per-item JSONL caches (`data/cache/distilbert_pt__distilbert-base-uncased__<split>.jsonl`),
   resumes after interruption and refuses to mix in records from another checkpoint. Its summary,
   `results/distilbert_pt.json`, has the GPU latency (every item) and the CPU latency (500 per split).
3. **FastAPI service + Kubernetes:**
   - `service/core.py` (framework-free logic) and `service/app.py`: POST /predict, GET /health, GET /ready;
   - `service/Dockerfile` (CPU-only torch) and `.dockerignore`;
   - `k8s/`: Namespace, ConfigMap, Deployment with startup/readiness/liveness probes and resource limits,
     a LoadBalancer Service and an HPA, plus `kustomization.yaml` and `extras/metrics-server-insecure-tls.json`;
   - `scripts/load_test.py`.
4. **Wiring:**
   - `src/systems.py` covers the new families in the registry and in classify(), for the app's Classify page;
   - `summarize_results.py`, `error_analysis.py` and `significance.py` add rows/pairs only when results exist;
   - the Streamlit app shows availability, the new comparison columns and a DistilBERT training section.

## Files
- **New:**
  - `src/classic.py`, `src/distilbert.py`;
  - `scripts/run_classic.py`, `scripts/train_distilbert.py`, `scripts/eval_distilbert.py`, `scripts/load_test.py`;
  - `service/__init__.py`, `service/core.py`, `service/app.py`, `service/Dockerfile`, `.dockerignore`;
  - `k8s/*.yaml`, `k8s/extras/metrics-server-insecure-tls.json`;
  - `requirements-nlp.txt`, `requirements-torch.txt`, `requirements-service.txt`;
  - `tests/test_classic.py`, `tests/test_distilbert.py`, `tests/test_service.py`;
  - `UPGRADE_NOTES.md`.
- **Changed (additions only):**
  - `config.yaml` (`classic`, `distilbert` blocks);
  - `src/systems.py`;
  - `scripts/summarize_results.py`, `scripts/error_analysis.py`, `scripts/significance.py`;
  - `streamlit_app.py`;
  - `README.md` (new "Upgrade" section, design decisions, 8 GB note);
  - `PROGRESS.md`;
  - `.gitignore`.
- **Untouched:** all existing results, caches, models, the notebook and the existing tests.

## How to run (in order)
From `AI_Engineer_Portfolio\finetune_vs_prompting`, with no other heavy job running. The scripts take
`AI_Engineer_Portfolio\.llm_lock`.
```powershell
# A. Classic baselines (CPU)
.venv\Scripts\pip install -r requirements-nlp.txt
.venv\Scripts\python scripts\run_classic.py
# B. DistilBERT (GPU)
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements-torch.txt
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python scripts\train_distilbert.py --epochs 1 --limit 512     # smoke test
.venv\Scripts\python scripts\train_distilbert.py
.venv\Scripts\python scripts\eval_distilbert.py
# C. Results
.venv\Scripts\python scripts\summarize_results.py
.venv\Scripts\python scripts\error_analysis.py
.venv\Scripts\python scripts\significance.py
.venv\Scripts\python -m pytest -q
.venv\Scripts\streamlit run streamlit_app.py
# D. Service + Kubernetes: see README "FastAPI service and Kubernetes" (build, metrics-server, apply -k, verify,
#    load test, delete -k)
```

## What to verify
- `pytest -q`: 54 passed and 5 skipped before the installs (verified 2026-10-17, no new packages). After step B
  the 3 torch tests run, and after `pip install -r requirements-service.txt` the 2 route tests run.
- Existing rows in `results/comparison.csv` are identical before and after `summarize_results.py`. Compare the
  (a)-(e) rows with the Oct 2 file. Only new rows and the new columns `latency_device`, `p50_ms_cpu` and
  `p95_ms_cpu` appear.
- `run_classic.py` prints the chosen hyperparameters. If a `NOTE: ... edge of the grid` appears, widen that grid in
  config and re-run, as was done for (a). The tuning stays on val only.
- `train_distilbert.py` prints the truncation count at max_len 64 (expected: very few). Val loss should fall for
  the first epochs, and early stopping should keep the best epoch. `peak_gpu_memory_mb` in
  `results/distilbert_training_log.json` must be well under 4 GB.
- Kubernetes:
  - `kubectl -n banking77 get pods` shows 1/1 READY only after the model has loaded;
  - `/ready` returns 200;
  - `/predict` returns a valid label;
  - during the `--tag hpa` load test, `kubectl get hpa` shows the CPU going up and the replicas going to 2;
  - `kubectl top pod` stays under the 1280Mi limit.
- Then fill in the README "not run yet" cells from `comparison.csv`, `distilbert_training_log.json`,
  `significance.json` and `load_test_*.json`. Retake the Evaluation and Training screenshots.

## Expected run time (estimates, not measured)
- **Classic baselines:** about 2-6 min in total. The spaCy lemmatisation of about 13k messages takes about
  30-60 s and is cached in `data/cache/spacy_lemmas__en_core_web_sm.jsonl`. The char n-gram SVM is the slowest fit.
- **DistilBERT training:** about 1-2 min per epoch on the RTX 3050 with fp16 (282 steps/epoch), so about 5-15 min
  with early stopping. A first run also downloads distilbert-base-uncased (about 270 MB).
- **eval_distilbert.py:** about 1-3 min on the GPU for 3,850 single requests, plus about 1-2 min of CPU latency
  sampling.
- **Kubernetes:**
  - the Docker build takes about 5-10 min the first time (the CPU torch wheel is about 200 MB);
  - the image is likely about 1-1.5 GB;
  - a pod is ready about 10-30 s after it starts;
  - the load tests take 3-4 min.

## Risks
- **Package pins** were checked on PyPI on 2026-10-17 (spacy 3.8.16 has a cp314 Windows wheel; torch 2.14.1 is on
  the cu126 and cpu indexes for cp314-win and cp312-linux), but they were not installed. transformers 5.x changed
  some APIs. The code uses only `AutoTokenizer`, `AutoModelForSequenceClassification`, `from_pretrained` and
  `save_pretrained` to limit exposure. The schedule is hand-written (`LambdaLR`), not `get_linear_schedule_with_warmup`.
- **Driver:** cu126 needs a recent NVIDIA driver. If `torch.cuda.is_available()` is False, update the driver or use
  the cu130 index.
- **Kubernetes on Docker Desktop:**
  - the HPA needs metrics-server (patched with `--kubelet-insecure-tls`, fine for a local cluster only);
  - with the *kind* provisioner, locally built images may not be visible;
  - memory requests and limits are estimates, so check them with `kubectl top pod` and adjust `deployment.yaml`.
- **Memory:** do not run DistilBERT training, Ollama evaluations and the Kubernetes cluster at the same time on the
  8 GB laptop. Stop the cluster (`kubectl delete -k k8s/`, or disable Kubernetes) before heavy jobs.
- **Lemma cache:** the spaCy cache is keyed by text and model name. Delete
  `data/cache/spacy_lemmas__en_core_web_sm.jsonl` if the spaCy or model version changes.
- **Stale DistilBERT caches:** after a retrain, `eval_distilbert.py` stops with a message to delete its old cache files.
  This is intended.
