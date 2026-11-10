"""Is the prompt masked out of the training labels? The batch the trainer really sees.

    python checks/tokenizer_template.py [--rows 4] [--soup-tools] [--run-id ...]

What it does: runs plain `soup train` on task/task.yaml in this process, up to the first
`training_step` call. That call gets the batch exactly as Soup's dataset formatting and TRL's
collator built it (input_ids, attention_mask, labels); the script records it and stops before
any weight changes. Nothing in Soup or TRL is replaced: only `training_step` of the trainer
instance is wrapped to read its argument.
For each of the first --rows rows of that batch it reports: tokens in total, tokens with
labels != -100, the decoded text of ONLY the tokens in the loss, whether the system prompt or
the user question is among them, and whether <|im_end|> is. The same counts are then taken
over the whole tokenized training set the trainer holds.
Prompt = every token before the last "<|im_start|>assistant\\n" of the row (Qwen's default
system prompt + the user turn). The assistant header itself is counted separately.
If the prompt is NOT masked (or with --soup-tools), it also runs Soup's own checks on the same
data and config and keeps their raw output: `soup data doctor`, `soup train --dry-run`,
`soup plan`, `soup ship`. It never changes the training.
Reads: task/task.yaml, data/train.jsonl, artifacts/adapter/ (for soup ship).
Writes: runs/checks/prompt_masking.json (+ prompt_masking_<run_id>.json),
logs/check_masking_<run_id>.yaml (the config), logs/soup_tools_*_<run_id>.log when they run.
Exits 1 when the prompt is not masked (a failed check stops the pipeline).
Task part: Part 3 (silent failures): the tokenizer / chat template / labels link.

Where the labels come from (Soup 0.77.0, TRL 0.29.1), found by reading the installed code:
- soup_cli/config/schema.py:419   data.train_on_responses_only defaults to True.
- soup_cli/data/sft_format.py:56,93-99  -> _build_assistant_only_format_row -> build_assistant_only_labels.
- soup_cli/data/loss_mask.py:115-150  preferred path: apply_chat_template(return_assistant_tokens_mask=True);
  Qwen2.5's template has no {% generation %}, the mask is all zero, so it returns None (this is the
  "[transformers] return_assistant_tokens_mask==True but chat template does not contain
  {% generation %}" line in the training log).
- soup_cli/data/loss_mask.py:335-361  fallback: render messages[:i] vs messages[:i+1]; the token
  delta of each assistant turn is unmasked ("<|im_start|>assistant\\n" + answer + "<|im_end|>\\n"),
  everything else is -100.
- soup_cli/trainer/sft.py:143-172 (_map_text_sft_rows) maps the rows to input_ids/labels/attention_mask;
  sft.py:175-186 (_ensure_assistant_masks) adds assistant_masks = (labels != -100).
- trl/trainer/sft_trainer.py:982,1007  the dataset has input_ids, so TRL does not tokenize again;
  sft_trainer.py:1176 truncate_dataset cuts every list column, labels included.
- trl/trainer/sft_trainer.py:829-832  completion_only_loss=None -> False (no prompt/completion columns);
  assistant_only_loss=False only means TRL does not compute its own mask.
- trl/trainer/sft_trainer.py:167-168  the collator takes example["labels"] when present (else input_ids);
  :223-227 it also sets labels to -100 wherever assistant_masks == 0.
So by the code the prompt should be masked; this script checks it on the real batch.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (ARTIFACTS, ENV, LOGS, MODEL, REPO, RUNS, log, new_run_id, rel,  # noqa: E402
                    run_logged, run_record, write_json)

OUT_DIR = RUNS / "checks"
CODE_REFS = [
    "soup_cli/config/schema.py:419 data.train_on_responses_only default True",
    "soup_cli/data/sft_format.py:56,93-99 -> build_assistant_only_labels",
    "soup_cli/data/loss_mask.py:115-150 preferred path needs {% generation %} (Qwen2.5 has none) -> None",
    "soup_cli/data/loss_mask.py:335-361 fallback: assistant turn delta unmasked, rest -100",
    "soup_cli/trainer/sft.py:143-186 rows -> input_ids/labels/attention_mask (+assistant_masks)",
    "trl/trainer/sft_trainer.py:982,1007 pre-tokenized dataset is not tokenized again",
    "trl/trainer/sft_trainer.py:829-832 completion_only_loss None -> False",
    "trl/trainer/sft_trainer.py:167-168,223-227 collator uses dataset labels, then assistant_masks",
]


class Captured(Exception):
    """Raised from the first training_step, after the batch has been recorded."""


def capture_first_batch(config_path):
    """Run `soup train` in this process up to the first training_step; return (batch,
    trainer). The batch is moved to the CPU."""
    import transformers

    got = {}
    original_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_step = self.training_step

        def training_step(model, inputs, *step_args, **step_kwargs):
            got["batch"] = {k: v.detach().cpu() for k, v in inputs.items() if hasattr(v, "detach")}
            got["trainer"] = self
            raise Captured()

        self.training_step = training_step
        got["original_step"] = original_step

    transformers.Trainer.__init__ = init
    sys.argv = ["soup", "train", "--config", str(config_path), "--yes"]
    try:
        from soup_cli.cli import run

        run()
    except (Captured, SystemExit) as exc:
        log(f"[masking] soup train stopped: {type(exc).__name__}")
    finally:
        transformers.Trainer.__init__ = original_init
    if "batch" not in got:
        sys.exit("soup train ended before the first training_step; see the output above")
    return got["batch"], got["trainer"]


def find_last(seq, pattern):
    for start in range(len(seq) - len(pattern), -1, -1):
        if seq[start:start + len(pattern)] == pattern:
            return start
    return None


def spans(positions):
    """Contiguous runs of positions: [[a, b), ...]."""
    runs = []
    for p in positions:
        if runs and runs[-1][1] == p:
            runs[-1][1] = p + 1
        else:
            runs.append([p, p + 1])
    return runs


def analyse_row(tok, ids, labels, header, im_end):
    """Split one row into prompt / assistant header / answer and count the loss tokens."""
    n = len(ids)
    trained = [i for i in range(n) if labels[i] != -100]
    h = find_last(ids, header)
    prompt_end = h if h is not None else 0
    answer_start = h + len(header) if h is not None else 0
    in_prompt = [i for i in trained if i < prompt_end]
    in_header = [i for i in trained if prompt_end <= i < answer_start]
    in_answer = [i for i in trained if i >= answer_start]
    return {
        "tokens": n,
        "tokens_in_loss": len(trained),
        "prompt_tokens": prompt_end,
        "prompt_tokens_in_loss": len(in_prompt),
        "header_tokens_in_loss": len(in_header),
        "answer_tokens": n - answer_start,
        "answer_tokens_in_loss": len(in_answer),
        "header_found": h is not None,
        "labels_equal_input_ids_where_trained": all(labels[i] == ids[i] for i in trained),
        "im_end_in_loss": any(ids[i] == im_end for i in in_answer),
        "trained": trained,
        "prompt_end": prompt_end,
    }


def row_evidence(tok, ids, row):
    trained = row.pop("trained")
    prompt_text = tok.decode(ids[:row["prompt_end"]], skip_special_tokens=False)
    pieces = [tok.decode(ids[a:b], skip_special_tokens=False) for a, b in spans(trained)]
    trained_text = " ⟦…⟧ ".join(pieces)
    system = prompt_text.split("<|im_start|>user", 1)[0]
    user = prompt_text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
    system_body = system.replace("<|im_start|>system\n", "").replace("<|im_end|>", "").strip()
    return {
        **{k: v for k, v in row.items() if k != "prompt_end"},
        "loss_spans": [[a, b] for a, b in spans(trained)],
        "loss_text": trained_text,
        "loss_text_last_tokens": [tok.decode([t]) for t in [ids[i] for i in trained][-3:]],
        "system_prompt_in_loss": bool(system_body) and system_body in trained_text,
        "user_question_in_loss": bool(user.strip()) and user.strip() in trained_text,
        "system_prompt": system_body,
        "user_question_head": user.strip()[:200],
    }


def run_soup_tools(config_path, run_id):
    """Soup's own data / pre-flight / ship checks on the same data and config; raw output kept."""
    soup = [sys.executable, "-c", "import sys; from soup_cli.cli import run; sys.argv[0]='soup'; run()"]
    tools = {
        "data_doctor": soup + ["data", "doctor", "data/train.jsonl", "--model", MODEL, "--format",
                               "alpaca", "--max-length", "512", "--show-mask", "1", "--output",
                               rel(OUT_DIR / f"data_doctor_{run_id}.json")],
        "train_dry_run": soup + ["train", "--config", rel(config_path), "--dry-run", "--yes"],
        "plan": soup + ["plan", "--config", rel(config_path), "--state",
                        rel(OUT_DIR / f"soup_plan_{run_id}.tfstate")],
        "ship": soup + ["ship", "--base", MODEL, "--adapter", rel(ARTIFACTS / "adapter")],
    }
    results = {}
    for name, cmd in tools.items():
        path = LOGS / f"soup_tools_{name}_{run_id}.log"
        log(f"[masking] soup tool {name}: {' '.join(cmd[3:])}")
        code, wall = run_logged(cmd, path, env=dict(ENV, SOUP_DB_PATH=str(OUT_DIR / f"soup_{run_id}.db")),
                                timeout=1800)
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [line for line in text.splitlines()
                if any(k in line.lower() for k in ("mask", "generation", "prompt", "label", "loss"))]
        results[name] = {"exit_code": code, "wall_s": wall, "log": rel(path),
                         "lines_mentioning_mask_or_loss": hits[:40]}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--soup-tools", action="store_true",
                        help="run Soup's own checks even when the prompt is masked")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    import yaml

    run_id = args.run_id or new_run_id()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    config["output"] = "./" + rel(OUT_DIR / f"capture_{run_id}")  # nothing is saved there
    config_path = LOGS / f"check_masking_{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    batch, trainer = capture_first_batch(config_path)
    tok = trainer.processing_class
    tok = getattr(tok, "tokenizer", tok)
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    rows = []
    for r in range(min(args.rows, batch["input_ids"].shape[0])):
        keep = batch["attention_mask"][r].bool()
        ids = batch["input_ids"][r][keep].tolist()
        labels = batch["labels"][r][keep].tolist()
        rows.append(row_evidence(tok, ids, analyse_row(tok, ids, labels, header, im_end)))
        rows[-1]["padding_tokens"] = int((~keep).sum())
        rows[-1]["padding_in_loss"] = int((batch["labels"][r][~keep] != -100).sum())

    # The whole tokenized training set the trainer holds (what every step draws from).
    totals = {"rows": 0, "tokens": 0, "tokens_in_loss": 0, "prompt_tokens": 0,
              "prompt_tokens_in_loss": 0, "header_tokens_in_loss": 0, "answer_tokens_in_loss": 0,
              "rows_without_header": 0, "rows_with_prompt_in_loss": 0, "rows_without_im_end_in_loss": 0}
    for example in trainer.train_dataset:
        ids, labels = list(example["input_ids"]), list(example["labels"])
        if "assistant_masks" in example:  # what the collator applies on top of labels
            labels = [lab if m else -100 for lab, m in zip(labels, example["assistant_masks"])]
        row = analyse_row(tok, ids, labels, header, im_end)
        totals["rows"] += 1
        for key in ("tokens", "tokens_in_loss", "prompt_tokens", "prompt_tokens_in_loss",
                    "header_tokens_in_loss", "answer_tokens_in_loss"):
            totals[key] += row[key]
        totals["rows_without_header"] += not row["header_found"]
        totals["rows_with_prompt_in_loss"] += row["prompt_tokens_in_loss"] > 0
        totals["rows_without_im_end_in_loss"] += not row["im_end_in_loss"]

    batch_prompt_in_loss = sum(r["prompt_tokens_in_loss"] for r in rows)
    masked = batch_prompt_in_loss == 0 and totals["prompt_tokens_in_loss"] == 0
    template = tok.chat_template or ""
    result = {
        "check": "prompt_masking", "run_id": run_id, "masked": masked,
        "prompt_share_in_loss": round(totals["prompt_tokens_in_loss"] / max(totals["tokens_in_loss"], 1), 6),
        "prompt_share_in_loss_first_batch": round(
            batch_prompt_in_loss / max(sum(r["tokens_in_loss"] for r in rows), 1), 6),
        "header_share_in_loss": round(totals["header_tokens_in_loss"] / max(totals["tokens_in_loss"], 1), 6),
        "evidence": rows[0] if rows else None,
        "first_batch_rows": rows,
        "first_batch_keys": sorted(batch),
        "train_dataset_columns": list(getattr(trainer.train_dataset, "column_names", []) or []),
        "train_dataset_totals": totals,
        "trainer_args": {k: str(getattr(trainer.args, k, None)) for k in
                         ("assistant_only_loss", "completion_only_loss", "packing", "padding_free",
                          "max_length", "per_device_train_batch_size")},
        "collator": type(trainer.data_collator).__name__,
        "collator_completion_only_loss": getattr(trainer.data_collator, "completion_only_loss", None),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "has_generation_markers": "{% generation %}" in template,
        "code_refs": CODE_REFS,
        "config": rel(config_path),
        **run_record(config["training"].get("seed")),
    }
    if not masked or args.soup_tools:
        result["soup_tools"] = run_soup_tools(config_path, run_id)

    write_json(OUT_DIR / f"prompt_masking_{run_id}.json", result)
    shutil.copyfile(OUT_DIR / f"prompt_masking_{run_id}.json", OUT_DIR / "prompt_masking.json")

    for i, r in enumerate(rows):
        print(f"\n--- row {i}: {r['tokens']} tokens, {r['tokens_in_loss']} in loss "
              f"(prompt {r['prompt_tokens_in_loss']}/{r['prompt_tokens']}, header {r['header_tokens_in_loss']}, "
              f"answer {r['answer_tokens_in_loss']}/{r['answer_tokens']}), padding {r['padding_tokens']} "
              f"(in loss {r['padding_in_loss']})")
        print(f"system prompt in loss: {r['system_prompt_in_loss']} | user question in loss: "
              f"{r['user_question_in_loss']} | <|im_end|> in loss: {r['im_end_in_loss']} | "
              f"last loss tokens: {r['loss_text_last_tokens']}")
        print(f"LOSS TEXT: {r['loss_text'][:800]}")
    log(f"[masking] masked={masked}, prompt share in loss {result['prompt_share_in_loss']} over "
        f"{totals['rows']} rows, header share {result['header_share_in_loss']} -> "
        f"{rel(OUT_DIR / 'prompt_masking.json')}")
    if not masked:
        sys.exit(1)


if __name__ == "__main__":
    main()
