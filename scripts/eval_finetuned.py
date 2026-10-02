r"""Evaluate (e), the LoRA fine-tuned model, after scripts\import_finetuned.ps1 has created it in Ollama.

  .venv\Scripts\python scripts\eval_finetuned.py                 # test_subset (770), then full test (3,080)
  .venv\Scripts\python scripts\eval_finetuned.py --subset-only

Resumable (per-item cache) and holds the shared lock, via scripts/eval_llm.py. Then rebuilds the comparison.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-only", action="store_true")
    a = ap.parse_args()
    splits = ["test_subset"] if a.subset_only else ["test_subset", "test"]
    for split in splits:
        subprocess.run([PY, str(ROOT / "scripts" / "eval_llm.py"), "--system", "finetuned_1p5b", "--split", split],
                       check=True)
    subprocess.run([PY, str(ROOT / "scripts" / "summarize_results.py")], check=True)
    subprocess.run([PY, str(ROOT / "scripts" / "error_analysis.py")], check=True)
