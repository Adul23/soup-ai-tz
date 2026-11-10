"""Smoke generation: base, merged and 4-bit answer the same 5 prompts.

    python harness/smoke_generate.py [--rows 5] [--run-id ...]

What it does: takes the first N rows of data/manual_eval.jsonl, builds the user turn exactly as
Soup does for an alpaca row, applies the base tokenizer's chat template (the same prompt ids
for all three models) and decodes greedily. Loads one model at a time. Records whether the
merged answers differ from the base ones at all, and whether the merged tokenizer's chat
template is the base one.
Reads: data/manual_eval.jsonl, the base model, artifacts/merged/, artifacts/merged_4bit/.
Writes: logs/smoke_generations.jsonl (appended; every line carries the run id),
logs/nvidia_smi_generate_<run_id>.csv, runs/steps/generate_<run_id>.json.
Task part: end-to-end smoke test (task 1); first look for part 2.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARTIFACTS, DATA, LOGS, MODEL, STEPS, SmiLogger, log, new_run_id, now_iso,  # noqa: E402
                    read_jsonl, rel, run_record, torch_peaks, user_turn, write_json)

MAX_NEW_TOKENS = 256
SEED = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = args.run_id or new_run_id()
    torch.manual_seed(SEED)
    rows = read_jsonl(DATA / "manual_eval.jsonl")[:args.rows]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_turn(row)}],
                                       add_generation_prompt=True, tokenize=False) for row in rows]
    models = {"base": MODEL, "merged": str(ARTIFACTS / "merged"),
              "4bit": str(ARTIFACTS / "merged_4bit")}
    template_same = {name: AutoTokenizer.from_pretrained(path).chat_template == tok.chat_template
                     for name, path in models.items() if name != "base"}

    out_path = LOGS / "smoke_generations.jsonl"
    answers, per_model = {}, {}
    started = time.perf_counter()
    with SmiLogger(LOGS / f"nvidia_smi_generate_{run_id}.csv") as smi:
        for name, path in models.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loading = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                         device_map={"": 0}).eval()
            answers[name] = []
            for index, (row, prompt) in enumerate(zip(rows, prompts)):
                enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                         pad_token_id=tok.pad_token_id)
                new = gen[0, enc["input_ids"].shape[1]:].tolist()
                text = tok.decode(new, skip_special_tokens=True)
                answers[name].append(text)
                with open(out_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "run_id": run_id, "time_utc": now_iso(), "model": name, "path": path,
                        "row": index, "prompt": prompt, "reference": row["output"],
                        "generation": text, "new_tokens": len(new),
                        "ended_with_eos": bool(new) and new[-1] in (tok.eos_token_id, tok.pad_token_id),
                    }, ensure_ascii=False) + "\n")
            per_model[name] = {"wall_s": round(time.perf_counter() - loading, 2),
                               **torch_peaks(torch)}
            log(f"[generate] {name}: {per_model[name]}")
            del model
    wall = round(time.perf_counter() - started, 2)

    n = len(rows)
    result = {"step": "generate", "run_id": run_id, "exit_code": 0, "wall_s": wall,
              "rows": n, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
              "chat_template_same_as_base": template_same, "per_model": per_model,
              "identical_merged_vs_base": sum(a == b for a, b in zip(answers["merged"], answers["base"])),
              "identical_4bit_vs_merged": sum(a == b for a, b in zip(answers["4bit"], answers["merged"])),
              "identical_4bit_vs_base": sum(a == b for a, b in zip(answers["4bit"], answers["base"])),
              "generations": rel(out_path), **smi.summary(), **run_record(SEED)}
    write_json(STEPS / f"generate_{run_id}.json", result)
    log(f"[generate] identical answers out of {n}: merged=base {result['identical_merged_vs_base']}, "
        f"4bit=merged {result['identical_4bit_vs_merged']}, 4bit=base {result['identical_4bit_vs_base']}")


if __name__ == "__main__":
    main()
