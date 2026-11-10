"""Measure some or all of the five variants on held-out answers, on one GPU. Numbers only.

    python verify/verify_artifacts.py --run-id ID --device 0 --variants base,zero_adapter,merged \\
        --skip-gen-for zero_adapter [--limit 20] [--gen-batch 16]

Variants: base (Qwen/Qwen2.5-1.5B-Instruct, fp16), zero_adapter (artifacts/zero_merged: the
adapter with B = 0 merged by harness/merge.py), merged (artifacts/merged), base_4bit
(artifacts/base_4bit: the base quantized by harness/quantize.py), merged_4bit
(artifacts/merged_4bit). verify/build_controls.py builds the two controls. Two processes
(e.g. GPU 0 and GPU 1) can each take part of the variants; verify/merge_verify.py puts their
part files together into runs/verify.json and computes everything that compares variants
(expectations, paired bootstrap, NF4 codes). verify/run_parallel.py starts both and the merge.

For every variant given, on data/heldout.jsonl (300 rows, or the first --limit), with the chat
template the training used:
- loss and perplexity on the ANSWER tokens only: the answer text and its closing <|im_end|>;
  system prompt, user turn and the "<|im_start|>assistant\\n" header are -100. Loss = sum of
  token losses / number of tokens (not a mean over rows). Truncated at 512 tokens like training.
  The per-row sums are kept for the paired bootstrap.
- bits per character: the same summed loss / ln 2 / characters of the decoded scored answer
  text (the <|im_end|> token's loss is included, it adds no characters).
- greedy generation (do_sample=False, repetition_penalty=1.0, max_new_tokens=256), batched
  (--gen-batch, left padding, rows sorted by prompt length): share that ends with EOS, mean
  length in tokens, share with a loop (the same 4-word n-gram 3+ times). Not for the variants
  in --skip-gen-for. With --limit, every variant also generates with batch 1 and the number of
  rows whose answer differs between batch 1 and the batch size is recorded.
If base and zero_adapter run in the same process and their loss differs by more than the fp16
tolerance, the process writes what it has and stops (exit 2), as TASK.md asks.
Reads: data/heldout.jsonl, the model folders of the variants, runs/checks/prompt_masking.json.
Writes: runs/verify_parts/verify_<run_id>_gpu<device>.json,
logs/verify_generations_<run_id>_gpu<device>.jsonl, logs/nvidia_smi_verify_<run_id>_gpu<device>.csv.
Task part: Part 2 (proof of training).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (ARTIFACTS, DATA, LOGS, MODEL, RUNS, SmiLogger, log, now_iso,  # noqa: E402
                    read_json, read_jsonl, rel, run_record, torch_peaks, user_turn, write_json)

MAX_LENGTH = 512
SEED = 0
ZERO_REL_TOL = 1e-3  # fp16 has ~1e-3 relative precision
PARTS = RUNS / "verify_parts"
VARIANTS = {
    "base": MODEL,
    "zero_adapter": str(ARTIFACTS / "zero_merged"),
    "merged": str(ARTIFACTS / "merged"),
    "base_4bit": str(ARTIFACTS / "base_4bit"),
    "merged_4bit": str(ARTIFACTS / "merged_4bit"),
    # Memory experiment only (harness/kernels_experiment.py), never part of the verdict:
    "merged_kernels": str(ARTIFACTS / "merged_kernels"),
}


# ---------------------------------------------------------------- examples

def build_examples(tok, rows):
    """Prompt ids (system + user + assistant header), full ids, and the scored answer positions."""
    from soup_cli.data.loss_mask import build_assistant_only_labels

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    examples, cross = [], Counter()
    for index, row in enumerate(rows):
        user = [{"role": "user", "content": user_turn(row)}]
        full_msgs = user + [{"role": "assistant", "content": row["output"]}]
        prompt_text = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        full_text = tok.apply_chat_template(full_msgs, tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        assert full_ids[:len(prompt_ids)] == prompt_ids, f"row {index}: prompt is not a prefix"
        last_end = max(i for i, t in enumerate(full_ids) if t == im_end)
        answer = [i for i in range(len(prompt_ids), last_end + 1) if i < MAX_LENGTH]
        ids = full_ids[:MAX_LENGTH]

        # Cross-check with the labels Soup trained on (same tokenizer, same template).
        soup = build_assistant_only_labels(full_msgs, tok, max_length=MAX_LENGTH)
        soup_trained = {i for i, lab in enumerate(soup["labels"]) if lab != -100}
        no_header = tok(tok.apply_chat_template(user, add_generation_prompt=False, tokenize=False),
                        add_special_tokens=False)["input_ids"]
        header = set(range(len(no_header), len(prompt_ids)))
        trailing = set(range(last_end + 1, len(ids)))
        cross["rows"] += 1
        cross["input_ids_differ_from_soup"] += soup["input_ids"] != ids
        cross["answer_not_in_soup_loss"] += not set(answer) <= soup_trained
        cross["soup_loss_outside_header_answer_trailing"] += bool(
            soup_trained - set(answer) - header - trailing)
        if not answer:
            cross["rows_without_answer_after_truncation"] += 1
            continue
        examples.append({"row": index, "ids": ids, "answer": answer, "prompt_text": prompt_text,
                         "prompt_len": len(prompt_ids)})
    return examples, dict(cross)


# ---------------------------------------------------------------- loss

def score(model, tok, examples, torch):
    """Summed answer-token NLL / tokens, bits per character, and the per-row sums."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    per_row = {"row": [], "nll": [], "tokens": [], "chars": []}
    for ex in examples:
        ids = torch.tensor([ex["ids"]], device="cuda")
        labels = torch.full_like(ids, -100)
        labels[0, ex["answer"]] = ids[0, ex["answer"]]
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="sum")
        text_ids = [ex["ids"][i] for i in ex["answer"] if ex["ids"][i] != im_end]
        per_row["row"].append(ex["row"])
        per_row["nll"].append(float(nll))
        per_row["tokens"].append(len(ex["answer"]))
        per_row["chars"].append(len(tok.decode(text_ids, skip_special_tokens=True)))
    nll, tokens, chars = sum(per_row["nll"]), sum(per_row["tokens"]), sum(per_row["chars"])
    loss = nll / tokens
    return {"loss": loss, "perplexity": math.exp(loss), "bits_per_char": nll / math.log(2) / chars,
            "answer_tokens": tokens, "answer_chars": chars, "nll_sum": nll, "per_row": per_row}


# ---------------------------------------------------------------- generation

def has_loop(text, n=4, times=3):
    words = text.split()
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return any(count >= times for count in grams.values())


def set_greedy(model, tok, max_new_tokens):
    """Replace the model's generation_config (Qwen ships do_sample=True and
    repetition_penalty=1.05): passing a config to generate() would let the model's defaults
    back in for every value left at the global default."""
    from transformers import GenerationConfig

    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    model.generation_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        eos_token_id=eos, pad_token_id=tok.pad_token_id)
    return eos


def generate(model, tok, examples, torch, batch, eos):
    """Greedy answers {row: (new token ids, ended with EOS)}; rows sorted by prompt length so
    a batch holds prompts of similar length; left padding."""
    tok.padding_side = "left"
    order = sorted(examples, key=lambda ex: ex["prompt_len"])
    answers = {}
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        enc = tok([ex["prompt_text"] for ex in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False)
        for ex, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
            stop = next((i for i, t in enumerate(seq) if t in eos), None)
            answers[ex["row"]] = (seq[:stop + 1] if stop is not None else seq, stop is not None)
    return answers


def generation_stats(tok, answers, variant, batch, run_id, out_path):
    stats = {"rows": 0, "with_eos": 0, "tokens": 0, "with_loop": 0}
    with open(out_path, "a", encoding="utf-8") as handle:
        for row, (new, ended) in sorted(answers.items()):
            text = tok.decode(new, skip_special_tokens=True)
            loop = has_loop(text)
            stats["rows"] += 1
            stats["with_eos"] += ended
            stats["tokens"] += len(new)
            stats["with_loop"] += loop
            handle.write(json.dumps({"run_id": run_id, "time_utc": now_iso(), "variant": variant,
                                     "batch": batch, "row": row, "generation": text,
                                     "new_tokens": len(new), "ended_with_eos": ended, "loop": loop},
                                    ensure_ascii=False) + "\n")
    return {"eos_share": stats["with_eos"] / stats["rows"],
            "mean_new_tokens": stats["tokens"] / stats["rows"],
            "loop_share": stats["with_loop"] / stats["rows"], "generated_rows": stats["rows"]}


# ---------------------------------------------------------------- NF4 codes (used by merge_verify.py)

def nf4_code_diff(a_dir, b_dir, targets):
    """Changed 4-bit codes between two bitsandbytes NF4 checkpoints, by module type."""
    import torch
    from safetensors import safe_open

    def tensors(folder):
        out = {}
        for path in sorted(Path(folder).glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    out[key] = handle.get_tensor(key)
        return out

    a, b = tensors(a_dir), tensors(b_dir)
    assert set(a) == set(b), f"different tensor names: {sorted(set(a) ^ set(b))[:5]}"
    per_type, other_changed = {}, {}
    for key in sorted(a):
        ta, tb = a[key], b[key]
        if key.endswith(".weight") and ta.dtype == torch.uint8 and f"{key}.absmax" in a:
            kind = key.split(".")[-2]
            x = ta ^ tb
            changed = int(((x & 0x0F) != 0).sum()) + int(((x >> 4) != 0).sum())
            entry = per_type.setdefault(kind, {"codes": 0, "changed": 0, "absmax_blocks": 0,
                                               "absmax_changed": 0, "modules": 0})
            entry["codes"] += 2 * ta.numel()
            entry["changed"] += changed
            entry["modules"] += 1
            entry["absmax_blocks"] += a[f"{key}.absmax"].numel()
            entry["absmax_changed"] += int((a[f"{key}.absmax"] != b[f"{key}.absmax"]).sum())
        elif ".weight." not in key:  # quant-state tensors follow the codes / absmax
            diff = int((ta != tb).sum()) if ta.shape == tb.shape else ta.numel()
            if diff:
                other_changed[key] = diff
    for entry in per_type.values():
        entry["changed_share"] = entry["changed"] / entry["codes"]
        entry["absmax_changed_share"] = entry["absmax_changed"] / max(entry["absmax_blocks"], 1)
    target_types = [k for k in per_type if k in targets]
    other_types = [k for k in per_type if k not in targets]

    def pooled(kinds):
        codes = sum(per_type[k]["codes"] for k in kinds)
        changed = sum(per_type[k]["changed"] for k in kinds)
        return {"codes": codes, "changed": changed, "changed_share": changed / codes if codes else None}

    return {"per_module_type": per_type, "lora_targets": target_types,
            "targets": pooled(target_types), "non_targets": pooled(other_types),
            "overall": pooled(list(per_type)), "non_quantized_tensors_changed": other_changed}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="shared by the parallel parts")
    parser.add_argument("--device", type=int, default=0, help="physical GPU index (nvidia-smi)")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"comma-separated, from {','.join(VARIANTS)}")
    parser.add_argument("--skip-gen-for", default="", help="comma-separated variants: loss only")
    parser.add_argument("--gen-batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N held-out rows; also compares batch 1 vs --gen-batch")
    args = parser.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    skip_gen = {v for v in args.skip_gen_for.split(",") if v}
    unknown = set(variants + list(skip_gen)) - set(VARIANTS)
    assert not unknown, f"unknown variants {unknown}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # before torch: cuda:0 is that GPU

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    missing = [VARIANTS[v] for v in variants if v != "base" and not Path(VARIANTS[v]).is_dir()]
    assert not missing, f"missing artifacts {missing}; run harness/smoke.py and verify/build_controls.py"

    tag = f"{args.run_id}_gpu{args.device}"
    out_json = PARTS / f"verify_{tag}.json"
    gen_path = LOGS / f"verify_generations_{tag}.jsonl"
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = read_jsonl(DATA / "heldout.jsonl")[:args.limit]
    examples, cross = build_examples(tok, rows)
    log(f"[verify gpu{args.device}] {variants}: {len(examples)}/{len(rows)} rows; Soup label cross-check {cross}")

    part = {"run_id": args.run_id, "device": args.device,
            "cuda_device": torch.cuda.get_device_name(0), "variants_requested": variants,
            "skip_gen_for": sorted(skip_gen), "limit": args.limit, "heldout_rows": len(rows),
            "scored_rows": len(examples), "max_length": MAX_LENGTH,
            "prompt_construction_vs_soup_labels": cross,
            "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
            "chat_template_sha256_per_variant": {
                v: hashlib.sha256((AutoTokenizer.from_pretrained(VARIANTS[v]).chat_template or "")
                                  .encode()).hexdigest() for v in variants},
            "chat_template_sha256_task_a": read_json(RUNS / "checks" / "prompt_masking.json", {})
            .get("chat_template_sha256"),
            "generation": {"decoding": "greedy", "do_sample": False, "repetition_penalty": 1.0,
                           "max_new_tokens": args.max_new_tokens, "batch": args.gen_batch,
                           "padding_side": "left", "order": "sorted by prompt length"},
            "generations": rel(gen_path), "variants": {}}

    def save():
        part.update(run_record(SEED))
        write_json(out_json, part)

    with SmiLogger(LOGS / f"nvidia_smi_verify_{tag}.csv", gpu=args.device) as smi:
        for name in variants:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(VARIANTS[name], dtype=torch.float16,
                                                         device_map={"": 0}).eval()
            entry = {"path": VARIANTS[name] if name == "base" else rel(VARIANTS[name]),
                     "memory_footprint_mib": round(model.get_memory_footprint() / 2**20, 1)}
            entry.update(score(model, tok, examples, torch))
            entry["loss_wall_s"] = round(time.perf_counter() - started, 2)
            log(f"[verify gpu{args.device}] {name}: loss {entry['loss']:.5f}, ppl "
                f"{entry['perplexity']:.3f}, bpc {entry['bits_per_char']:.4f}")
            part["variants"][name] = entry

            if name == "zero_adapter" and "base" in part["variants"]:
                base = part["variants"]["base"]["loss"]
                rel_diff = abs(entry["loss"] - base) / base
                entry["loss_relative_difference_to_base"] = rel_diff
                if rel_diff > ZERO_REL_TOL:
                    part["stopped"] = (f"zero_adapter loss {entry['loss']:.6f} vs base {base:.6f}: "
                                       f"relative difference {rel_diff:.2e} > {ZERO_REL_TOL}")
                    log(f"[verify gpu{args.device}] STOP: {part['stopped']}")
                    save()
                    sys.exit(2)

            if name in skip_gen:
                entry["generation"] = "skipped (--skip-gen-for)"
            else:
                eos = set_greedy(model, tok, args.max_new_tokens)
                gen_started = time.perf_counter()
                answers = generate(model, tok, examples, torch, args.gen_batch, eos)
                entry["generation_wall_s"] = round(time.perf_counter() - gen_started, 2)
                entry.update(generation_stats(tok, answers, name, args.gen_batch, args.run_id, gen_path))
                if args.limit is not None and args.gen_batch != 1:
                    single = generate(model, tok, examples, torch, 1, eos)
                    generation_stats(tok, single, name, 1, args.run_id, gen_path)
                    differ = [r for r in answers if answers[r][0] != single[r][0]]
                    entry["batch1_vs_batch"] = {
                        "batch": args.gen_batch, "rows": len(answers), "rows_differing": len(differ),
                        "differing_rows": differ,
                        "first_differing_token": {r: next((i for i, (x, y) in enumerate(
                            zip(answers[r][0], single[r][0])) if x != y),
                            min(len(answers[r][0]), len(single[r][0]))) for r in differ}}
                    log(f"[verify gpu{args.device}] {name}: batch 1 vs {args.gen_batch}: "
                        f"{len(differ)}/{len(answers)} answers differ")
            entry.update(torch_peaks(torch))
            entry["wall_s"] = round(time.perf_counter() - started, 2)
            save()
            del model
    part["nvidia_smi"] = smi.summary()
    save()
    log(f"[verify gpu{args.device}] -> {rel(out_json)}")


if __name__ == "__main__":
    main()
