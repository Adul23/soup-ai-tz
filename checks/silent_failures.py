"""Soup's built-in checks on the real data and artifacts, and on injected silent failures.

    python checks/silent_failures.py [--skip-ship] [--only a,pad_eos,max_length_64,...]

What it does, all numbers into runs/checks/silent_failures.json (no interpretation):
(a) Soup's own checks on the REAL inputs, full output in logs/:
    `soup data doctor` on data/train.jsonl (all rows, max_length 512, --show-mask 1),
    `soup train --dry-run` (pre-flight) on task/task.yaml,
    `soup ship --base <model> --adapter artifacts/adapter --task-eval <first 20 held-out rows>`
    (the CLI evaluates a LoRA adapter on top of its base; it takes no 4-bit model).
(b) Injected failures, each on a copy of the first 50 train rows (or of the real adapter /
    the merged model), no weight update anywhere (Soup's real batch is taken at the first
    training_step, as in checks/tokenizer_template.py):
    - pad_eos: the tokenizer's pad_token set to <|im_end|> (eos) inside `soup train`;
      is <|im_end|> still in the labels of the first batch and of the tokenized set?
    - max_length_64: data.max_length 64; share of rows whose <|im_end|> is cut;
    - dirty_data: 10 duplicated rows, 5 answers with "X және X", 5 with ".Буква" (no space);
    - renamed_adapter: lora keys in adapter_model.safetensors renamed (q_proj -> q_proj_renamed);
      harness/merge.py on it: are the merged weights equal to the zero-adapter merge (= base)?
      (and target_modules renamed in adapter_config.json, the loud variant);
    - template_no_system: inference chat template without Qwen's default system prompt; answer
      loss of merged and base on 50 held-out rows vs the training template.
    For every item: injected (what), soup_caught (per command: command, exit code, output log,
    report / warning lines, and whether it differs from the same command on the clean copy),
    our_check (what our code measures and its numbers, and whether it flags the item).
A Soup command that refuses an injected input with an error (e.g. "no causal-loss target
remains" at max_length 64, ship's "Target modules ... not found") is recorded as soup_caught,
loud: it is a result, not a crash. For (a) the doctor's counts are copied out as numbers
(rows_truncated, rows counted by eos_in_labels).
Exit codes: 0 done; 1 a Soup check failed on the REAL inputs (doctor MAJOR or dry-run non-zero);
3 one of our checks did not flag an injected failure that had an effect (1 and 3 are findings);
4 our own code failed unexpectedly (a capture without a Soup error, a merge that did not finish,
an exception in this script).
Disk: `df -h` of the repository disk and /tmp is printed before every item. The large
throw-away copies (injected adapters, their merged models) are written under /tmp and deleted
right after their comparison; their sha256 stays in the JSON.
Reads: data/*.jsonl, task/task.yaml, artifacts/adapter/, artifacts/merged/, artifacts/zero_merged/,
runs/smoke.json (base_weights of the merge).
Writes: runs/checks/silent_failures.json (+ _<run_id>.json), runs/checks/silent_<run_id>/ (the small
injected copies: data, tokenizers, configs; Soup reports), logs/silent_*_<run_id>.log,
/tmp/soup_silent_<run_id>/ (removed at the end).
Task part: Part 3 (silent failures).
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verify"))
from common import (ARTIFACTS, DATA, ENV, LOGS, MODEL, REPO, RUNS, dir_sha256, log,  # noqa: E402
                    new_run_id, read_json, read_jsonl, rel, run_logged, run_record, user_turn,
                    write_json)

OUT_DIR = RUNS / "checks"
ROWS = 50
SOUP = [sys.executable, "-c", "import sys; from soup_cli.cli import run; sys.argv[0]='soup'; run()"]
WARN = re.compile(r"warn|truncat|exceed|duplicat|missing|unexpected|pad|eos|mismatch|error", re.I)
NO_SYSTEM_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content']"
    " + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}{% endif %}")
TMP_ROOT = "/tmp"
ITEMS = ("a", "pad_eos", "max_length_64", "dirty_data", "renamed_adapter", "template_no_system")


class Ctx:
    def __init__(self, run_id, skip_ship):
        self.run_id, self.skip_ship = run_id, skip_ship
        self.dir = OUT_DIR / f"silent_{run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        # Large throw-away copies (merged models, injected adapters) go to /tmp: /kaggle/working
        # is nearly full after the main chain. Each is deleted right after it has been compared.
        self.tmp = Path(TMP_ROOT) / f"soup_silent_{run_id}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.crashed = []


def disk(where):
    """Print and return `df -h` of the repository's disk and /tmp before a step."""
    import subprocess

    paths = [str(REPO)] + [p for p in ("/kaggle/working", TMP_ROOT) if Path(p).exists()]
    text = subprocess.run(["df", "-h", *dict.fromkeys(paths)], capture_output=True, text=True).stdout
    log(f"[silent] disk before {where}:\n{text.rstrip()}")
    return text.splitlines()


# ---------------------------------------------------------------- running Soup

def strip_ts(line):
    return line.split(" ", 1)[1] if " " in line else line


def soup(ctx, name, args, timeout=1800):
    """Run a Soup CLI command, keep its full output in logs/, return the raw facts."""
    path = LOGS / f"silent_{name}_{ctx.run_id}.log"
    cmd = SOUP + [str(a) for a in args]
    log(f"[silent] soup {' '.join(map(str, args))}")
    code, wall = run_logged(cmd, path, env=dict(ENV, SOUP_DB_PATH=str(ctx.dir / "soup.db")),
                            timeout=timeout)
    lines = not_ours([strip_ts(l) for l in
                      path.read_text(encoding="utf-8", errors="replace").splitlines()[1:-1]])
    return {"command": "soup " + " ".join(map(str, args)), "exit_code": code, "wall_s": wall,
            "log": rel(path), "warning_lines": [l for l in lines if WARN.search(l)][:60],
            "output_tail": lines[-25:]}


def doctor(ctx, name, data, model=MODEL, max_length=512, sample=ROWS):
    report = ctx.dir / f"doctor_{name}.json"
    out = soup(ctx, f"doctor_{name}", ["data", "doctor", rel(data), "--model", model, "--format",
                                       "alpaca", "--max-length", max_length, "--sample", sample,
                                       "--show-mask", 1, "--output", rel(report)])
    rep = read_json(report, {})
    out["report"] = rel(report)
    out["overall"] = rep.get("overall")
    out["checks"] = {c["name"]: {"verdict": c["verdict"], "message": c["message"],
                                 "evidence": c["evidence"]} for c in rep.get("checks", [])}
    return out


def dry_run(ctx, name, config):
    out = soup(ctx, f"dry_run_{name}", ["train", "--config", rel(config), "--dry-run", "--yes"])
    text = " ".join(out["output_tail"])
    m = re.search(r"Data OK:\s*(\d+) train samples", text)
    out["train_samples"] = int(m.group(1)) if m else None
    return out


def ship(ctx, name, adapter):
    if ctx.skip_ship:
        return {"command": None, "not_run": "--skip-ship"}
    task_eval = ctx.dir / "ship_task_eval.jsonl"
    if not task_eval.exists():  # Soup's custom-eval format; 'contains' is its whole-answer scorer
        with open(task_eval, "w", encoding="utf-8") as handle:
            for row in read_jsonl(DATA / "heldout.jsonl")[:20]:
                handle.write(json.dumps({"prompt": user_turn(row), "expected": row["output"],
                                         "scoring": "contains"}, ensure_ascii=False) + "\n")
    verdict = ctx.dir / f"ship_{name}.json"
    out = soup(ctx, f"ship_{name}", ["ship", "--base", MODEL, "--adapter", rel(adapter),
                                     "--task-eval", rel(task_eval), "--device", "cuda",
                                     "--output", rel(verdict)], timeout=2700)
    out["verdict_json"] = read_json(verdict)
    out["verdict"] = {0: "SHIP", 2: "DON'T SHIP"}.get(out["exit_code"], f"exit {out['exit_code']}")
    out["task_eval"] = rel(task_eval)
    return out


def compare_doctor(clean, dirty):
    worse = {k: [clean["checks"].get(k, {}).get("verdict"), v["verdict"]]
             for k, v in dirty["checks"].items()
             if v["verdict"] != clean["checks"].get(k, {}).get("verdict")}
    caught = dirty["exit_code"] != clean["exit_code"] or dirty["overall"] != clean["overall"] or bool(worse)
    return {"caught": caught, "checks_changed_vs_clean": worse,
            "overall_clean": clean["overall"], "overall_injected": dirty["overall"]}


def train_log_new(clean_cap, cap, tag):
    """Warning-like lines of the injected soup train output that the clean run did not print."""
    return new_lines({"warning_lines": clean_cap.get("soup_train_warning_lines", [])},
                     {"warning_lines": cap.get("soup_train_warning_lines", [])}, (tag, "clean"))


def new_lines(clean, dirty, replace=()):
    def norm(lines):
        out = set()
        for line in lines:
            for a in replace:
                line = line.replace(a, "")
            out.add(re.sub(r"\d+(\.\d+)?", "#", line))
        return out
    return sorted(norm(dirty["warning_lines"]) - norm(clean["warning_lines"]))


def not_ours(lines):
    """Drop the lines this script prints itself ([silent] ...): they are not a signal from Soup."""
    return [l for l in lines if not l.startswith("[silent]")]


def soup_error(lines):
    """Soup's own refusal: the 'Error: ...' line and the lines that continue it."""
    for i, line in enumerate(lines):
        if line.startswith("Error:"):
            block = [line]
            for nxt in lines[i + 1:i + 6]:
                if not nxt.strip() or nxt.startswith(("[", "Traceback", "20")):
                    break
                block.append(nxt)
            return " ".join(b.strip() for b in block)
    return None


# ---------------------------------------------------------------- data copies and configs

def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def config(ctx, name, data, max_length=None):
    import yaml

    cfg = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    cfg["data"]["train"] = "./" + rel(data)
    if max_length:
        cfg["data"]["max_length"] = max_length
    cfg["output"] = "./" + rel(ctx.dir / f"capture_{name}")  # nothing is saved there
    path = ctx.dir / f"config_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def capture(ctx, name, cfg, pad_eos=False):
    """Soup's real first batch + tokenized set, in a fresh process (see inner_capture)."""
    out = ctx.dir / f"capture_{name}.json"
    path = LOGS / f"silent_capture_{name}_{ctx.run_id}.log"
    code, wall = run_logged([sys.executable, "checks/silent_failures.py", "--inner-capture",
                             rel(cfg), rel(out), *(["--pad-eos"] if pad_eos else [])], path)
    lines = [strip_ts(l) for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    result = read_json(out, {})
    refusal = soup_error(lines) if code != 0 else None
    if (code != 0 or not result) and not refusal:
        ctx.crashed.append(f"capture_{name}")  # our code failed, not a Soup refusal
    return {"exit_code": code, "wall_s": wall, "log": rel(path), "json": rel(out) if result else None,
            "soup_refused": refusal,
            "soup_train_warning_lines": not_ours([l for l in lines if WARN.search(l)])[:60], **result}


def inner_capture(cfg, out, pad_eos):
    """Runs inside its own process: `soup train` up to the first training_step."""
    import transformers
    from tokenizer_template import analyse_row, capture_first_batch

    if pad_eos:
        original = transformers.AutoTokenizer.from_pretrained

        def patched(*args, **kwargs):
            tok = original(*args, **kwargs)
            tok.pad_token = "<|im_end|>"
            print(f"[silent] INJECTED pad_token = {tok.pad_token!r} (id {tok.pad_token_id})", flush=True)
            return tok
        transformers.AutoTokenizer.from_pretrained = patched

    batch, trainer = capture_first_batch(cfg)
    tok = trainer.processing_class
    tok = getattr(tok, "tokenizer", tok)
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def count(rows_iter):
        t = {"rows": 0, "rows_im_end_in_loss": 0, "rows_im_end_present_but_masked": 0,
             "rows_im_end_cut": 0, "rows_without_loss_tokens": 0, "rows_prompt_in_loss": 0,
             "tokens": 0, "tokens_in_loss": 0}
        for ids, labels in rows_iter:
            row = analyse_row(tok, ids, labels, header, im_end)
            start = row["prompt_end"] + len(header) if row["header_found"] else len(ids)
            present = any(ids[i] == im_end for i in range(start, len(ids)))
            t["rows"] += 1
            t["rows_im_end_in_loss"] += row["im_end_in_loss"]
            t["rows_im_end_present_but_masked"] += present and not row["im_end_in_loss"]
            t["rows_im_end_cut"] += not present
            t["rows_without_loss_tokens"] += row["tokens_in_loss"] == 0
            t["rows_prompt_in_loss"] += row["prompt_tokens_in_loss"] > 0
            t["tokens"] += row["tokens"]
            t["tokens_in_loss"] += row["tokens_in_loss"]
        return t

    def batch_rows():
        for r in range(batch["input_ids"].shape[0]):
            keep = batch["attention_mask"][r].bool()
            yield batch["input_ids"][r][keep].tolist(), batch["labels"][r][keep].tolist()

    def dataset_rows():
        for ex in trainer.train_dataset:
            ids, labels = list(ex["input_ids"]), list(ex["labels"])
            if "assistant_masks" in ex:
                labels = [lab if m else -100 for lab, m in zip(labels, ex["assistant_masks"])]
            yield ids, labels

    pads = ~batch["attention_mask"].bool()
    result = {
        "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
        "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id, "im_end_id": im_end,
        "trainer_max_length": str(getattr(trainer.args, "max_length", None)),
        "first_batch": count(batch_rows()),
        "first_batch_padding_tokens": int(pads.sum()),
        "first_batch_padding_in_loss": int((batch["labels"][pads] != -100).sum()),
        "first_batch_im_end_tokens_in_loss": int(((batch["input_ids"] == im_end)
                                                  & (batch["labels"] != -100)).sum()),
        "train_dataset": count(dataset_rows()),
    }
    write_json(out, result)
    print(f"[silent] capture: {json.dumps(result, ensure_ascii=False)}", flush=True)


def share(n, d):
    return n / d if d else None


# ---------------------------------------------------------------- (a) real inputs

def item_a(ctx, train_rows):
    out = {"injected": "nothing: Soup's own checks on the real data/train.jsonl, task/task.yaml "
                       "and artifacts/adapter",
           "data_doctor": doctor(ctx, "real", DATA / "train.jsonl", sample=len(train_rows)),
           "train_dry_run": dry_run(ctx, "real", REPO / "task" / "task.yaml"),
           "ship": ship(ctx, "real", ARTIFACTS / "adapter")}
    checks = out["data_doctor"]["checks"]

    def fraction(check):
        m = re.search(r"(\d+)/(\d+) rows", (checks.get(check) or {}).get("message", ""))
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)
    truncated, scanned = fraction("truncation_risk")
    no_eos, eos_rows = fraction("eos_in_labels")
    out["real_data_numbers"] = {
        "source": out["data_doctor"]["report"],
        "rows_total": len(train_rows),
        "rows_truncated": truncated, "truncation_rows_scanned": scanned,
        "eos_in_labels_rows_without_eos": no_eos, "eos_in_labels_rows_counted": eos_rows,
        "rows_not_counted_by_eos_in_labels": None if eos_rows is None else len(train_rows) - eos_rows,
    }
    out["failed_on_real_inputs"] = (out["data_doctor"]["exit_code"] != 0
                                    or out["data_doctor"]["overall"] == "MAJOR"
                                    or out["train_dry_run"]["exit_code"] != 0)
    return out


# ---------------------------------------------------------------- (b) injected

def item_pad_eos(ctx, clean_data, clean_cap, clean_doc):
    tok_dir = ctx.dir / "tokenizer_pad_eos"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = "<|im_end|>"
    tok.save_pretrained(str(tok_dir))
    cap = capture(ctx, "pad_eos", config(ctx, "pad_eos", clean_data), pad_eos=True)
    doc = doctor(ctx, "pad_eos", clean_data, model=rel(tok_dir))
    effect = cap.get("train_dataset", {}).get("rows_im_end_in_loss") != \
        clean_cap.get("train_dataset", {}).get("rows_im_end_in_loss")
    ds = cap.get("train_dataset", {})
    return {
        "injected": "tokenizer.pad_token = '<|im_end|>' (the eos) inside soup train (patched "
                    "AutoTokenizer.from_pretrained); the same tokenizer saved for data doctor",
        "soup_caught": {
            "any": compare_doctor(clean_doc, doc)["caught"]
                   or bool(train_log_new(clean_cap, cap, "pad_eos")),
            "data_doctor": {**doc, **compare_doctor(clean_doc, doc)},
            "soup_train_log": {"log": cap.get("log"),
                               "new_warning_lines_vs_clean": train_log_new(clean_cap, cap, "pad_eos")},
            "train_dry_run": {"not_run": "the pad token is not a soup.yaml field; --dry-run "
                                         "loads no tokenizer"},
            "ship": {"not_run": "ship evaluates a trained adapter; this fault is in training"}},
        "our_check": {
            "what": "Soup's real first batch and tokenized set: rows whose answer <|im_end|> "
                    "is in the loss (labels != -100), vs the clean copy",
            "clean": {"pad_token": clean_cap.get("pad_token"), **clean_cap.get("train_dataset", {}),
                      "first_batch_im_end_tokens_in_loss": clean_cap.get("first_batch_im_end_tokens_in_loss")},
            "injected": {"pad_token": cap.get("pad_token"), **ds,
                         "first_batch_im_end_tokens_in_loss": cap.get("first_batch_im_end_tokens_in_loss"),
                         "first_batch_padding_in_loss": cap.get("first_batch_padding_in_loss")},
            "injection_applied": cap.get("pad_token") == "<|im_end|>",
            "im_end_masked_share": share(ds.get("rows_im_end_present_but_masked", 0), ds.get("rows")),
            "effect_present": effect,
            "flags": bool(ds.get("rows_im_end_present_but_masked")),
        },
        "capture": cap.get("json"),
    }


def item_max_length(ctx, clean_data, clean_cap, clean_doc, clean_dry):
    cfg = config(ctx, "max_length_64", clean_data, max_length=64)
    cap = capture(ctx, "max_length_64", cfg)
    doc = doctor(ctx, "max_length_64", clean_data, max_length=64)
    dry = dry_run(ctx, "max_length_64", cfg)
    ds = cap.get("train_dataset", {})
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    lengths = []
    for row in read_jsonl(clean_data):
        user = [{"role": "user", "content": user_turn(row)}]
        text = tok.apply_chat_template(user + [{"role": "assistant", "content": row["output"]}],
                                       tokenize=False)
        lengths.append(len(tok(text.rstrip("\n"), add_special_tokens=False)["input_ids"]))
    cut = sum(n > 64 for n in lengths)  # the last token of the row is the answer's <|im_end|>
    return {
        "injected": "data.max_length 64 (task.yaml has 512) on the 50-row copy",
        "soup_caught": {
            "any": bool(cap.get("soup_refused")) or compare_doctor(clean_doc, doc)["caught"]
                   or dry["exit_code"] != clean_dry["exit_code"]
                   or bool(train_log_new(clean_cap, cap, "max_length_64")),
            "loud": bool(cap.get("soup_refused")),
            "soup_train": {"log": cap.get("log"), "exit_code": cap.get("exit_code"),
                           "refused": cap.get("soup_refused")},
            "data_doctor": {**doc, **compare_doctor(clean_doc, doc)},
            "train_dry_run": {**dry, "caught": dry["exit_code"] != clean_dry["exit_code"],
                              "new_warning_lines_vs_clean": new_lines(clean_dry, dry, ("max_length_64", "clean"))},
            "soup_train_log": {"log": cap.get("log"),
                               "new_warning_lines_vs_clean": train_log_new(clean_cap, cap, "max_length_64")},
            "ship": {"not_run": "ship evaluates a trained adapter; this fault is in training"}},
        "our_check": {
            "what": "rows of the copy whose full chat-template rendering (up to the answer's "
                    "<|im_end|>) is longer than 64 tokens, i.e. whose <|im_end|> is cut; counted with "
                    "the tokenizer, so it does not need Soup's tokenized set (Soup refused to build it)",
            "rows": len(lengths), "rows_im_end_cut": cut,
            "im_end_cut_share": share(cut, len(lengths)),
            "length_p50_p100": [sorted(lengths)[len(lengths) // 2], max(lengths)] if lengths else None,
            "soup_tokenized_set": ds or None,
            "effect_present": cut > 0,
            "flags": cut > 0,
        },
        "capture": cap.get("json"),
    }


def item_dirty_data(ctx, rows, clean_data, clean_doc, clean_dry):
    from language import NO_SPACE_AFTER_PERIOD, REPEAT_PAIR

    dirty = [dict(r) for r in rows[:ROWS - 10]] + [dict(r) for r in rows[:10]]  # 10 duplicates
    changed = {"repeat_pair": [], "no_space_after_period": []}
    for i in range(10, 15):  # "X және X": the first word of 3+ letters doubled
        m = re.search(r"\w{3,}", dirty[i]["output"])
        if m:
            w = m.group(0)
            dirty[i]["output"] = dirty[i]["output"][:m.start()] + f"{w} және {w}" + dirty[i]["output"][m.end():]
            changed["repeat_pair"].append(i)
    for i in range(15, 20):  # ". Буква" -> ".Буква"
        new = re.sub(r"\.\s+(?=[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ])", ".", dirty[i]["output"])
        if new == dirty[i]["output"]:  # one-sentence answer: append a second sentence
            new = dirty[i]["output"].rstrip(".") + ".Бұл сөйлем."
        dirty[i]["output"] = new
        changed["no_space_after_period"].append(i)
    path = write_rows(ctx.dir / "train50_dirty.jsonl", dirty)
    cfg = config(ctx, "dirty_data", path)
    doc = doctor(ctx, "dirty_data", path)
    dry = dry_run(ctx, "dirty_data", cfg)

    def ours(data_rows):
        keys = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in data_rows]
        outs = [r["output"] for r in data_rows]
        return {"rows": len(data_rows), "exact_duplicate_rows": len(keys) - len(set(keys)),
                "rows_repeat_pair": sum(bool(REPEAT_PAIR.search(t)) for t in outs),
                "rows_no_space_after_period": sum(bool(NO_SPACE_AFTER_PERIOD.search(t)) for t in outs)}
    clean_ours, dirty_ours = ours(rows[:ROWS]), ours(dirty)
    return {
        "injected": {"duplicate_rows": 10, "repeat_pair_rows": changed["repeat_pair"],
                     "no_space_after_period_rows": changed["no_space_after_period"],
                     "file": rel(path)},
        "soup_caught": {
            "any": compare_doctor(clean_doc, doc)["caught"] or dry["exit_code"] != clean_dry["exit_code"]
                   or dry.get("train_samples") != clean_dry.get("train_samples"),
            "data_doctor": {**doc, **compare_doctor(clean_doc, doc)},
            "train_dry_run": {**dry, "caught": dry["exit_code"] != clean_dry["exit_code"]
                              or dry.get("train_samples") != clean_dry.get("train_samples"),
                              "train_samples_clean": clean_dry.get("train_samples")},
            "ship": {"not_run": "ship evaluates a trained adapter; this fault is in the data"}},
        "our_check": {
            "what": "exact duplicate rows (instruction+input+output), checks/language.py patterns "
                    "on the outputs",
            "clean": clean_ours, "injected": dirty_ours, "effect_present": True,
            "flags": dirty_ours["exact_duplicate_rows"] > clean_ours["exact_duplicate_rows"]
                     and dirty_ours["rows_repeat_pair"] > clean_ours["rows_repeat_pair"]
                     and dirty_ours["rows_no_space_after_period"] > clean_ours["rows_no_space_after_period"],
        },
    }


def tensor_diff(a_dir, b_dir, targets):
    """Per tensor: equal or not, between two saved fp16 models; grouped into LoRA targets / rest."""
    import torch
    from safetensors import safe_open

    def index(folder):
        out = {}
        for p in sorted(Path(folder).glob("*.safetensors")):
            with safe_open(str(p), framework="pt", device="cpu") as h:
                for k in h.keys():
                    out[k] = p
        return out
    ia, ib = index(a_dir), index(b_dir)
    res = {"targets": {"tensors": 0, "changed": 0, "max_abs_diff": 0.0},
           "other": {"tensors": 0, "changed": 0, "max_abs_diff": 0.0},
           "names_only_in_one": sorted(set(ia) ^ set(ib))[:10]}
    for key in sorted(set(ia) & set(ib)):
        with safe_open(str(ia[key]), framework="pt", device="cpu") as h:
            ta = h.get_tensor(key)
        with safe_open(str(ib[key]), framework="pt", device="cpu") as h:
            tb = h.get_tensor(key)
        group = res["targets" if any(f".{t}." in key for t in targets) else "other"]
        group["tensors"] += 1
        if ta.shape != tb.shape or not torch.equal(ta, tb):
            group["changed"] += 1
            if ta.shape == tb.shape:
                group["max_abs_diff"] = max(group["max_abs_diff"],
                                            float((ta.float() - tb.float()).abs().max()))
    return res


def item_renamed_adapter(ctx, base_weights):
    from safetensors.torch import load_file, save_file

    reference = ARTIFACTS / "zero_merged"
    assert reference.is_dir(), "artifacts/zero_merged missing: run verify first (build_controls.py)"
    src = ARTIFACTS / "adapter"
    targets = read_json(src / "adapter_config.json", {}).get("target_modules") or []
    targets = [targets] if isinstance(targets, str) else list(targets)
    out = {"injected": {}, "soup_caught": {}, "our_check": {}}

    disk("renamed_adapter")
    keys_dir = ctx.tmp / "adapter_renamed_keys"
    shutil.copytree(src, keys_dir)
    tensors = load_file(str(src / "adapter_model.safetensors"))
    renamed = {}
    for k, v in tensors.items():
        for t in targets:
            k = k.replace(f".{t}.", f".{t}_renamed.")
        renamed[k] = v.contiguous()
    save_file(renamed, str(keys_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg_dir = ctx.tmp / "adapter_renamed_config"
    shutil.copytree(src, cfg_dir)
    acfg = read_json(cfg_dir / "adapter_config.json")
    acfg["target_modules"] = [f"{t}_renamed" for t in targets]
    write_json(cfg_dir / "adapter_config.json", acfg)
    out["injected"] = {
        "keys": f"adapter_model.safetensors keys .{targets} -> ._renamed (adapter_config unchanged)",
        "config": f"adapter_config.json target_modules -> {acfg['target_modules']} (weights unchanged)",
        "copies": {"keys": {"path": str(keys_dir), "sha256": dir_sha256(keys_dir)},
                   "config": {"path": str(cfg_dir), "sha256": dir_sha256(cfg_dir)},
                   "deleted_after_check": True}}

    ref_diff = tensor_diff(ARTIFACTS / "merged", reference, targets)
    for variant, adapter in (("keys", keys_dir), ("config", cfg_dir)):
        merged = ctx.tmp / f"merged_{variant}"
        disk(f"merge of the {variant}-renamed adapter")
        path = LOGS / f"silent_merge_{variant}_{ctx.run_id}.log"
        code, wall = run_logged([sys.executable, "harness/merge.py", "--base-weights", base_weights,
                                 "--adapter", rel(adapter), "--out", rel(merged),
                                 "--run-id", f"silent_{variant}_{ctx.run_id}"], path)
        lines = [strip_ts(l) for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        step = read_json(RUNS / "steps" / f"merge_silent_{variant}_{ctx.run_id}.json", {})
        diff = tensor_diff(merged, reference, targets) if (code == 0 and merged.is_dir()) else None
        sha = dir_sha256(merged)
        if merged.is_dir():
            shutil.rmtree(merged)
        peft_warning = next((l for l in lines if "missing adapter keys" in l), None)
        if code != 0 and not any("rror" in l and "peft" in l.lower() or "Target modules" in l
                                 for l in lines):
            ctx.crashed.append(f"merge_{variant}")  # e.g. disk full: our step failed, nothing measured
        out["our_check"][variant] = {
            "merge_exit_code": code, "merge_wall_s": wall, "merge_log": rel(path),
            "merge_error": next((l for l in reversed(lines) if "Error" in l), None) if code else None,
            "merge_warning_lines": [l for l in lines if WARN.search(l)][:30],
            "identical": None if diff is None else (diff["targets"]["changed"] == 0
                                                    and diff["other"]["changed"] == 0
                                                    and not diff["names_only_in_one"]),
            "peft_missing_keys_warning": peft_warning[:4000] if peft_warning else None,
            "only_signal": ("the peft UserWarning above; merge exit 0 and the model saved"
                            if code == 0 and peft_warning else None),
            "lora_modules_merged": step.get("lora_modules_merged"),
            "vs_zero_merged": diff, "merged_sha256": sha, "merged_deleted_after_check": True,
            "merged_equals_base": (diff is not None and diff["targets"]["changed"] == 0
                                   and diff["other"]["changed"] == 0),
            "effect_present": code == 0 and diff is not None and diff["targets"]["changed"] == 0,
            "flags": code != 0 or (diff is not None and diff["targets"]["changed"] == 0)}
        out["soup_caught"][f"ship_{variant}"] = ship(ctx, f"renamed_{variant}", adapter)
        shutil.rmtree(adapter, ignore_errors=True)
    out["our_check"]["what"] = ("harness/merge.py on the injected adapter, then every tensor compared "
                                "with artifacts/zero_merged (B = 0 merge = base); flags when no "
                                "LoRA-target tensor changed (or the merge failed)")
    out["our_check"]["reference_real_merged_vs_zero_merged"] = ref_diff
    out["our_check"]["base_weights"] = base_weights
    out["our_check"]["flags"] = all(out["our_check"][v]["flags"] for v in ("keys", "config"))
    out["our_check"]["effect_present"] = any(out["our_check"][v]["effect_present"] for v in ("keys", "config"))
    out["soup_caught"]["data_doctor"] = {"not_run": "data doctor checks data, not an adapter"}
    out["soup_caught"]["train_dry_run"] = {"not_run": "pre-flight checks a training config, not an adapter"}
    for s in out["soup_caught"].values():  # ship: exit 1 = refused to run (loud), 2 = a verdict
        if isinstance(s, dict) and s.get("command"):
            s["caught"] = s.get("exit_code") not in (0, 2)
            s["refused"] = soup_error(s.get("output_tail", [])) if s["caught"] else None
    out["soup_caught"]["any"] = any(s.get("caught") for s in out["soup_caught"].values()
                                    if isinstance(s, dict))
    out["soup_caught"]["by_variant"] = {v: bool(out["soup_caught"].get(f"ship_{v}", {}).get("caught"))
                                        for v in ("keys", "config")}
    out["soup_caught"]["note"] = ("ship exit 2 = DON'T SHIP is a verdict, not a detected fault; "
                                  "compare its verdict_json with item a's ship on the real adapter")
    return out


def item_template(ctx, clean_data, clean_doc):
    import torch
    from merge_verify import paired_bootstrap
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verify_artifacts import MAX_LENGTH, score

    tok = AutoTokenizer.from_pretrained(MODEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = read_jsonl(DATA / "heldout.jsonl")[:ROWS]

    def examples(template):
        exs = []
        for i, row in enumerate(rows):
            user = [{"role": "user", "content": user_turn(row)}]
            prompt = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False,
                                             chat_template=template)
            full = tok.apply_chat_template(user + [{"role": "assistant", "content": row["output"]}],
                                           tokenize=False, chat_template=template)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            assert f_ids[:len(p_ids)] == p_ids, f"row {i}: prompt is not a prefix"
            last = max(j for j, t in enumerate(f_ids) if t == im_end)
            answer = [j for j in range(len(p_ids), last + 1) if j < MAX_LENGTH]
            exs.append({"row": i, "ids": f_ids[:MAX_LENGTH], "answer": answer, "prompt": prompt})
        return exs

    default, no_system = examples(None), examples(NO_SYSTEM_TEMPLATE)
    assert all(d["prompt"].endswith(n["prompt"]) for d, n in zip(default, no_system)), \
        "the no-system prompt should be the training prompt without its system block"
    losses = {}
    for name, path in (("merged", str(ARTIFACTS / "merged")), ("base", MODEL)):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map={"": 0}).eval()
        for tname, exs in (("training_template", default), ("no_system_template", no_system)):
            losses[f"{name}_{tname}"] = score(model, tok, exs, torch)
        del model
        torch.cuda.empty_cache()
    boot = {f"{m}: no_system - training": paired_bootstrap(losses[f"{m}_no_system_template"]["per_row"],
                                                           losses[f"{m}_training_template"]["per_row"])
            for m in ("merged", "base")}

    tok_dir = ctx.dir / "tokenizer_no_system"
    tok2 = AutoTokenizer.from_pretrained(MODEL)
    tok2.chat_template = NO_SYSTEM_TEMPLATE
    tok2.save_pretrained(str(tok_dir))
    doc = doctor(ctx, "template_no_system", clean_data, model=rel(tok_dir))
    masking = read_json(OUT_DIR / "prompt_masking.json", {})
    return {
        "injected": "inference chat template without a system prompt (Qwen's template adds its "
                    "default system prompt, as in training): " + NO_SYSTEM_TEMPLATE,
        "soup_caught": {
            "any": compare_doctor(clean_doc, doc)["caught"],
            "data_doctor": {**doc, **compare_doctor(clean_doc, doc),
                            "note": "the changed template given to data doctor as the tokenizer's"},
            "train_dry_run": {"not_run": "the fault is at inference; pre-flight sees the training config"},
            "ship": {"not_run": "soup ship renders prompts with the model's own template; it has no "
                                "option for a different template (only --chat-template-kwargs)"}},
        "our_check": {
            "what": "answer-token loss (verify_artifacts.score) on 50 held-out rows with the "
                    "training template vs without the system prompt; template sha256 vs training",
            "rows": len(rows),
            "loss": {k: {"loss": v["loss"], "perplexity": v["perplexity"], "answer_tokens": v["answer_tokens"]}
                     for k, v in losses.items()},
            "paired_bootstrap": boot,
            "training_template_sha256": masking.get("chat_template_sha256"),
            "inference_template_sha256": hashlib.sha256(NO_SYSTEM_TEMPLATE.encode()).hexdigest(),
            "effect_present": True,
            "flags": masking.get("chat_template_sha256") is None
                     or hashlib.sha256(NO_SYSTEM_TEMPLATE.encode()).hexdigest() != masking.get("chat_template_sha256"),
        },
    }


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-ship", action="store_true", help="do not run soup ship (slow)")
    parser.add_argument("--only", default=",".join(ITEMS), help=f"comma-separated from {','.join(ITEMS)}")
    parser.add_argument("--inner-capture", nargs=2, metavar=("CONFIG", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--pad-eos", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner_capture:
        inner_capture(*args.inner_capture, args.pad_eos)
        return

    only = [i for i in args.only.split(",") if i]
    assert not set(only) - set(ITEMS), f"unknown items {set(only) - set(ITEMS)}"
    ctx = Ctx(new_run_id(), args.skip_ship)
    train_rows = read_jsonl(DATA / "train.jsonl")
    clean_data = write_rows(ctx.dir / "train50_clean.jsonl", train_rows[:ROWS])
    base_weights = read_json(RUNS / "smoke.json", {}).get("base_weights")
    if "renamed_adapter" in only:
        assert base_weights, "runs/smoke.json has no base_weights: run harness/smoke.py first"
    result = {"check": "silent_failures", "run_id": ctx.run_id, "rows_in_copies": ROWS,
              "work_dir": rel(ctx.dir), "items": {}}
    out_path = OUT_DIR / f"silent_failures_{ctx.run_id}.json"

    def save():
        result["crashed"] = ctx.crashed
        result.update(run_record(read_json(RUNS / "smoke.json", {}).get("seed")))
        write_json(out_path, result)
        shutil.copyfile(out_path, OUT_DIR / "silent_failures.json")

    needs_clean = {"pad_eos", "max_length_64", "dirty_data", "template_no_system"} & set(only)
    if needs_clean:  # the same commands on the clean 50-row copy: what "caught" is compared with
        clean_cfg = config(ctx, "clean", clean_data)
        result["clean_copy"] = {
            "data": rel(clean_data),
            "data_doctor": doctor(ctx, "clean", clean_data),
            "train_dry_run": dry_run(ctx, "clean", clean_cfg),
            "capture": capture(ctx, "clean", clean_cfg)
            if {"pad_eos", "max_length_64"} & set(only) else None}
        save()
    clean = result.get("clean_copy", {})

    steps = {
        "a": lambda: item_a(ctx, train_rows),
        "pad_eos": lambda: item_pad_eos(ctx, clean_data, clean["capture"], clean["data_doctor"]),
        "max_length_64": lambda: item_max_length(ctx, clean_data, clean["capture"],
                                                 clean["data_doctor"], clean["train_dry_run"]),
        "dirty_data": lambda: item_dirty_data(ctx, train_rows, clean_data, clean["data_doctor"],
                                              clean["train_dry_run"]),
        "renamed_adapter": lambda: item_renamed_adapter(ctx, base_weights),
        "template_no_system": lambda: item_template(ctx, clean_data, clean["data_doctor"]),
    }
    for name in only:
        log(f"[silent] ==== {name}")
        disk(name)
        try:
            result["items"][name] = steps[name]()
        except Exception as exc:  # recorded, then the run goes on; exit 4 at the end
            import traceback

            traceback.print_exc()
            ctx.crashed.append(name)
            result["items"][name] = {"crashed": f"{type(exc).__name__}: {exc}"}
        save()

    items = result["items"]
    result["summary"] = {
        name: {"soup_caught": (it.get("soup_caught") or {}).get("any"),
               "our_check_flags": (it.get("our_check") or {}).get("flags"),
               "effect_present": (it.get("our_check") or {}).get("effect_present")}
        for name, it in items.items() if name != "a" and "crashed" not in it}
    if "a" in items and "crashed" not in items["a"]:
        result["summary"]["a"] = {
            "data_doctor_overall": items["a"]["data_doctor"]["overall"],
            "data_doctor_exit_code": items["a"]["data_doctor"]["exit_code"],
            "train_dry_run_exit_code": items["a"]["train_dry_run"]["exit_code"],
            "ship_verdict": items["a"]["ship"].get("verdict"),
            "failed_on_real_inputs": items["a"]["failed_on_real_inputs"]}
    shutil.rmtree(ctx.tmp, ignore_errors=True)
    missed = [n for n, s in result["summary"].items()
              if n != "a" and s["effect_present"] and not s["our_check_flags"]]
    result["our_check_missed"] = missed
    save()

    print(f"\n{'item':20s} {'soup_caught':>12s} {'our_flags':>10s} {'effect':>7s}")
    for name, s in result["summary"].items():
        if name == "a":
            print(f"{'a (real inputs)':20s} doctor {s['data_doctor_overall']} (exit {s['data_doctor_exit_code']}), "
                  f"dry-run exit {s['train_dry_run_exit_code']}, ship {s['ship_verdict']}")
        else:
            print(f"{name:20s} {str(s['soup_caught']):>12s} {str(s['our_check_flags']):>10s} "
                  f"{str(s['effect_present']):>7s}")
    log(f"[silent] crashed {ctx.crashed}, our check missed {missed} -> {rel(OUT_DIR / 'silent_failures.json')}")
    if ctx.crashed:
        sys.exit(4)
    if result["summary"].get("a", {}).get("failed_on_real_inputs"):
        sys.exit(1)
    if missed:
        sys.exit(3)


if __name__ == "__main__":
    main()
