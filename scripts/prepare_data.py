r"""Download Banking77 and write the fixed-seed splits. Usage: .venv\Scripts\python scripts\prepare_data.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import download, make_splits  # noqa: E402

if __name__ == "__main__":
    print(download())
    print(json.dumps(make_splits(), indent=1))
