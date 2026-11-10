"""Tests of task/prepare_data.py that need no network and no GPU.

What it does: checks the split sizes from TASK.md (1500 / 300 / 40), that the splits do not
overlap and fit in the dataset, and that write_jsonl keeps exactly the alpaca fields and the
Kazakh text.
Reads: task/prepare_data.py (imported by path).
Writes: only pytest's temporary folder.
Task part: infrastructure (tests); supports part 3 (data).
"""

import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "task" / "prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", PATH)
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


def test_split_sizes():
    sizes = {name: len(rows) for name, rows in prepare_data.SPLITS.items()}
    assert sizes == {"train.jsonl": 1500, "heldout.jsonl": 300, "manual_eval.jsonl": 40}


def test_splits_disjoint_and_in_range():
    seen = set()
    for rows in prepare_data.SPLITS.values():
        indices = set(rows)
        assert not indices & seen
        assert max(indices) < prepare_data.DATASET_ROWS
        seen |= indices


def test_write_jsonl_keeps_alpaca_fields(tmp_path):
    rows = [{"instruction": "Сәлем айт", "input": "", "output": "Сәлем!", "extra": 1}]
    prepare_data.write_jsonl(tmp_path / "x.jsonl", rows)
    lines = (tmp_path / "x.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"instruction": "Сәлем айт", "input": "", "output": "Сәлем!"}]
