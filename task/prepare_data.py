"""Download the task data, write the three data files and data/data_report.json.

    python task/prepare_data.py     # -> data/train.jsonl, heldout.jsonl, manual_eval.jsonl, data_report.json

What it does: takes fixed, disjoint row ranges of a pinned revision of the dataset, so every run
sees the same rows. The rows are written as they are: nothing is dropped or changed (the report
says so with zero counters by reason). It then describes the written files, numbers only:
- rows that WOULD be dropped / changed by common cleaning rules, by reason, not applied
  (empty instruction / output, whitespace at the edges, exact duplicates in the split, text also
  in train, longer than max_length 512 tokens);
- lengths in tokens with the Qwen2.5 chat template exactly as Soup renders an alpaca row
  (default system prompt + user turn + answer), of the prompt and of the answer: p50/p90/p95/p99/p100;
- fertility: tokens per whitespace word of the answer and of the instruction;
- round-trip: the file re-read equals the dataset rows; tokenizer decode(encode(text)) == text;
- sha256 of each file, dataset name, revision and row count, tokenizer and template sha256.
Reads: AmanMussa/kazakh-instruction-v2 (Hugging Face, MIT license, 52,201 rows), Kazakh
instructions in the alpaca format (instruction, input, output); the Qwen2.5 tokenizer.
Writes: data/train.jsonl (1500 rows), data/heldout.jsonl (300), data/manual_eval.jsonl (40),
data/data_report.json (memory/budget.py takes the padded batch length from tokens_full.train.p100).
Task part: data for all parts (training, proof of training, manual evaluation); memory budget.

Adapted from justdoner/task/prepare_data.py: same dataset revision and row format; the splits
follow TASK.md.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET = "AmanMussa/kazakh-instruction-v2"
REVISION = "c641407ee50723c84a43ea2ec867b464d3893c12"
DATASET_ROWS = 52201
SPLITS = {
    "train.jsonl": range(1500),
    "heldout.jsonl": range(40000, 40300),
    "manual_eval.jsonl": range(41000, 41040),
}
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
FIELDS = ("instruction", "input", "output")


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


def percentiles(values):
    """Nearest-rank percentiles and the mean."""
    v = sorted(values)
    if not v:
        return None
    pick = lambda q: v[min(len(v) - 1, max(0, -(-q * len(v) // 100) - 1))]  # noqa: E731
    return {"n": len(v), "mean": round(sum(v) / len(v), 2), "p50": pick(50), "p90": pick(90),
            "p95": pick(95), "p99": pick(99), "p100": v[-1]}


def data_report(out, source_rows):
    """Numbers about the written files; see the module docstring."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
    from common import user_turn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    report = {"dataset": DATASET, "revision": REVISION, "dataset_rows": DATASET_ROWS,
              "tokenizer": MODEL, "max_length": MAX_LENGTH,
              "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
              "policy": "rows written as they are: nothing dropped, nothing changed",
              "dropped": {"total": 0, "by_reason": {}}, "changed": {"total": 0, "by_reason": {}},
              "splits": {}, "would_drop_or_change_not_applied": {}, "tokens_full": {},
              "tokens_prompt": {}, "tokens_answer": {}, "fertility": {}, "round_trip": {},
              "sha256": {}}
    texts_train = None
    for name, indices in SPLITS.items():
        split = name.removesuffix(".jsonl")
        path = out / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected = [{k: r[k] for k in FIELDS} for r in source_rows[name]]
        report["splits"][split] = {"source_rows": [indices.start, indices.stop], "rows": len(rows)}
        report["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

        full, prompt, answer, words_out, words_ins, tok_out, tok_ins = [], [], [], 0, 0, 0, 0
        decode_ok = 0
        for r in rows:
            user = [{"role": "user", "content": user_turn(r)}]
            p_text = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
            f_text = tok.apply_chat_template(user + [{"role": "assistant", "content": r["output"]}],
                                             tokenize=False)
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
            f_ids = tok(f_text, add_special_tokens=False)["input_ids"]
            full.append(len(f_ids))
            prompt.append(len(p_ids))
            answer.append(len(f_ids) - len(p_ids))
            o_ids = tok(r["output"], add_special_tokens=False)["input_ids"]
            i_ids = tok(r["instruction"], add_special_tokens=False)["input_ids"]
            tok_out += len(o_ids)
            tok_ins += len(i_ids)
            words_out += len(r["output"].split())
            words_ins += len(r["instruction"].split())
            decode_ok += all(tok.decode(tok(t, add_special_tokens=False)["input_ids"]) == t
                             for t in (r["instruction"], r["input"], r["output"]))

        keys = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
        texts = {(r["instruction"], r["input"], r["output"]) for r in rows}
        report["would_drop_or_change_not_applied"][split] = {
            "empty_instruction": sum(not r["instruction"].strip() for r in rows),
            "empty_output": sum(not r["output"].strip() for r in rows),
            "whitespace_at_edges": sum(any(r[k] != r[k].strip() for k in FIELDS) for r in rows),
            "exact_duplicates_in_split": len(keys) - len(set(keys)),
            "also_in_train": None if texts_train is None else len(texts & texts_train),
            f"longer_than_{MAX_LENGTH}_tokens": sum(n > MAX_LENGTH for n in full),
        }
        if split == "train":
            texts_train = texts
        report["tokens_full"][split] = percentiles(full)
        report["tokens_prompt"][split] = percentiles(prompt)
        report["tokens_answer"][split] = percentiles(answer)
        report["fertility"][split] = {
            "answer_tokens_per_word": round(tok_out / max(words_out, 1), 4),
            "instruction_tokens_per_word": round(tok_ins / max(words_ins, 1), 4),
            "answer_words": words_out, "answer_tokens": tok_out}
        report["round_trip"][split] = {
            "file_equals_dataset_rows": rows == expected,
            "tokenizer_decode_encode_equal_rows": decode_ok, "rows": len(rows)}
        assert rows == expected, f"{name}: the written file differs from the dataset rows"
    return report


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
    source_rows = {}
    for name, rows in SPLITS.items():
        source_rows[name] = [dataset[i] for i in rows]
        write_jsonl(out / name, source_rows[name])
        print(f"wrote {len(rows)} rows to {out / name}")
    report = data_report(out, source_rows)
    (out / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"wrote {out / 'data_report.json'}: tokens_full {report['tokens_full']}")
    print(f"would drop/change (not applied): {report['would_drop_or_change_not_applied']}")


if __name__ == "__main__":
    main()
