"""Download the task data and write the three data files.

    python task/prepare_data.py     # -> data/train.jsonl, data/heldout.jsonl, data/manual_eval.jsonl

What it does: takes fixed, disjoint row ranges of a pinned revision of the dataset, so every run
sees the same rows.
Reads: AmanMussa/kazakh-instruction-v2 (Hugging Face, MIT license, 52,201 rows), Kazakh
instructions in the alpaca format (instruction, input, output).
Writes: data/train.jsonl (1500 rows), data/heldout.jsonl (300), data/manual_eval.jsonl (40).
Task part: data for all parts (training, proof of training, manual evaluation).

Adapted from justdoner/task/prepare_data.py: same dataset revision and row format; the splits
follow TASK.md.
"""

import argparse
import json
from pathlib import Path

DATASET = "AmanMussa/kazakh-instruction-v2"
REVISION = "c641407ee50723c84a43ea2ec867b464d3893c12"
DATASET_ROWS = 52201
SPLITS = {
    "train.jsonl": range(1500),
    "heldout.jsonl": range(40000, 40300),
    "manual_eval.jsonl": range(41000, 41040),
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "output": row["output"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "data")
    )
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", revision=REVISION)
    assert len(dataset) == DATASET_ROWS, (
        f"expected {DATASET_ROWS} rows, got {len(dataset)}"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in SPLITS.items():
        write_jsonl(out / name, (dataset[i] for i in rows))
        print(f"wrote {len(rows)} rows to {out / name}")


if __name__ == "__main__":
    main()
