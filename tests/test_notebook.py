"""The Colab notebook must use the same prompt and split logic as local inference."""
import ast
import inspect
import json

import pytest

from src import data, prompts
from src.config import ROOT

NB = ROOT / "notebooks" / "finetune_lora_colab.ipynb"
pytestmark = pytest.mark.skipif(not NB.exists(), reason="run scripts/build_notebook.py")


def code_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


def test_all_cells_parse():
    for src in code_cells():
        ast.parse(src)


def test_same_system_prompt():
    g = {}
    prompt_cell = next(c for c in code_cells() if "FINETUNE_SYSTEM =" in c)
    exec(prompt_cell, g)
    assert g["FINETUNE_SYSTEM"] == prompts.FINETUNE_SYSTEM
    assert g["finetune_messages"]("x", "y") == prompts.finetune_messages("x", "y")


def test_same_split_function():
    data_cell = next(c for c in code_cells() if "def stratified_split" in c)
    assert inspect.getsource(data.stratified_split).split('"""')[-1].strip() in data_cell.replace("\r", "")
