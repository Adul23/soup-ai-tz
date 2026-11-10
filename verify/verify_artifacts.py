"""Check the three artifacts and two controls on held-out answers. Numbers only.

    python verify/verify_artifacts.py [--gen-batch 32] [--max-new-tokens 256] [--run-id ...]

Variants: base (Qwen/Qwen2.5-1.5B-Instruct, fp16), zero_adapter (artifacts/zero_merged: the
adapter with B = 0 merged by harness/merge.py), merged (artifacts/merged), base_4bit
(artifacts/base_4bit: the base quantized by harness/quantize.py), merged_4bit
(artifacts/merged_4bit). verify/build_controls.py builds the two controls.

For every variant, on data/heldout.jsonl (300 rows), with the chat template the training used:
- loss and perplexity on the ANSWER tokens only: the answer text and its closing <|im_end|>;
  system prompt, user turn and the "<|im_start|>assistant\\n" header are -100. Loss = sum of
  token losses / number of tokens (not a mean over rows). Truncated at 512 tokens like training.
- bits per character: the same summed loss / ln 2 / characters of the decoded scored answer
  text (the <|im_end|> token's loss is included, it adds no characters).
- greedy generation for all rows (do_sample=False, repetition_penalty=1.0, max_new_tokens=256,
  batched with left padding): share that ends with EOS, mean length in tokens, share with a loop
  (the same 4-word n-gram 3+ times).
Also:
- the share of NF4 codes that differ between base_4bit and merged_4bit, per LoRA target module
  type (from adapter_config.json), per other module type, and overall; plus non-quantized
  tensors that differ (embedding, norms).
- the prompt construction is checked against Soup's own labels (soup_cli build_assistant_only_labels)
  and the chat template against runs/checks/prompt_masking.json (task A).
Expectations are asserted but do not stop the script; a broken one is listed in
"expectation_violated". The exception: zero_adapter must match base within fp16 tolerance;
if it does not, the script writes what it has and stops (exit 2), as TASK.md asks.
Reads: data/heldout.jsonl, the five model folders, artifacts/adapter/adapter_config.json,
runs/checks/prompt_masking.json (if present).
Writes: runs/verify.json (+ runs/verify_<run_id>.json), logs/verify_generations_<run_id>.jsonl,
logs/nvidia_smi_verify_<run_id>.csv; a table on the console.
Task part: Part 2 (proof of training).
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (ARTIFACTS, DATA, LOGS, MODEL, RUNS, SmiLogger, log,  # noqa: E402
                    new_run_id, now_iso, read_json, read_jsonl, rel, run_record, torch_peaks,
                    user_turn, write_json)

MAX_LENGTH = 512
SEED = 0
ZERO_REL_TOL = 1e-3  # fp16 has ~1e-3 relative precision
VARIANTS = {
    "base": MODEL,
    "zero_adapter": str(ARTIFACTS / "zero_merged"),
    "merged": str(ARTIFACTS / "merged"),
    "base_4bit": str(ARTIFACTS / "base_4bit"),
    "merged_4bit": str(ARTIFACTS / "merged_4bit"),
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
        header = set(range(len(tok(tok.apply_chat_template(user, add_generation_prompt=False,
                                                           tokenize=False),
                                   add_special_tokens=False)["input_ids"]), len(prompt_ids)))
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
                         "reference": row["output"]})
    return examples, dict(cross)


# ---------------------------------------------------------------- metrics

def score(model, tok, examples, torch):
    """Summed answer-token NLL / tokens, and bits per character."""
    total_nll, total_tokens, total_chars = 0.0, 0, 0
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for ex in examples:
        ids = torch.tensor([ex["ids"]], device="cuda")
        labels = torch.full_like(ids, -100)
        labels[0, ex["answer"]] = ids[0, ex["answer"]]
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_tokens += len(ex["answer"])
        text_ids = [ex["ids"][i] for i in ex["answer"] if ex["ids"][i] != im_end]
        total_chars += len(tok.decode(text_ids, skip_special_tokens=True))
    loss = total_nll / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss),
            "bits_per_char": total_nll / math.log(2) / total_chars,
            "answer_tokens": total_tokens, "answer_chars": total_chars, "nll_sum": total_nll}


def has_loop(text, n=4, times=3):
    words = text.split()
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return any(count >= times for count in grams.values())


def generate_all(model, tok, examples, torch, batch, max_new_tokens, variant, run_id, out_path):
    from transformers import GenerationConfig

    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    # Replaces the model's own generation_config (Qwen ships do_sample=True and
    # repetition_penalty=1.05): passing a config to generate() would let the model's
    # defaults back in for every value left at the global default.
    model.generation_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        eos_token_id=eos, pad_token_id=tok.pad_token_id)
    tok.padding_side = "left"
    stats = {"rows": 0, "with_eos": 0, "tokens": 0, "with_loop": 0}
    with open(out_path, "a", encoding="utf-8") as handle:
        for start in range(0, len(examples), batch):
            chunk = examples[start:start + batch]
            enc = tok([ex["prompt_text"] for ex in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc)
            for ex, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                stop = next((i for i, t in enumerate(seq) if t in eos), None)
                new = seq[:stop + 1] if stop is not None else seq
                text = tok.decode(new, skip_special_tokens=True)
                loop = has_loop(text)
                stats["rows"] += 1
                stats["with_eos"] += stop is not None
                stats["tokens"] += len(new)
                stats["with_loop"] += loop
                handle.write(json.dumps({"run_id": run_id, "time_utc": now_iso(), "variant": variant,
                                         "row": ex["row"], "generation": text, "new_tokens": len(new),
                                         "ended_with_eos": stop is not None, "loop": loop},
                                        ensure_ascii=False) + "\n")
    return {"eos_share": stats["with_eos"] / stats["rows"],
            "mean_new_tokens": stats["tokens"] / stats["rows"],
            "loop_share": stats["with_loop"] / stats["rows"], "generated_rows": stats["rows"],
            "generation_config": {"do_sample": False, "repetition_penalty": 1.0,
                                  "max_new_tokens": max_new_tokens, "batch": batch,
                                  "padding_side": "left", "eos_token_id": eos,
                                  "effective": {k: str(v) for k, v in
                                                model.generation_config.to_diff_dict().items()}}}


# ---------------------------------------------------------------- NF4 codes

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
        elif ".weight." not in key:  # skip quant-state tensors, they follow the codes/absmax
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
            "overall": pooled(list(per_type)),
            "non_quantized_tensors_changed": other_changed}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N held-out rows (a quick run); default all 300")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = args.run_id or new_run_id()
    torch.manual_seed(SEED)
    missing = [p for n, p in VARIANTS.items() if n != "base" and not Path(p).is_dir()]
    assert not missing, f"missing artifacts {missing}; run harness/smoke.py and verify/build_controls.py"

    tok = AutoTokenizer.from_pretrained(MODEL)
    template_sha = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
    task_a = read_json(RUNS / "checks" / "prompt_masking.json", {})
    templates = {name: hashlib.sha256((AutoTokenizer.from_pretrained(path).chat_template or "")
                                      .encode()).hexdigest() for name, path in VARIANTS.items()}
    rows = read_jsonl(DATA / "heldout.jsonl")[:args.limit]
    examples, cross = build_examples(tok, rows)
    log(f"[verify] {len(examples)}/{len(rows)} held-out rows scored; cross-check with Soup labels {cross}")

    result = {"run_id": run_id, "limit": args.limit, "heldout_rows": len(rows),
              "scored_rows": len(examples),
              "max_length": MAX_LENGTH, "prompt_construction_vs_soup_labels": cross,
              "chat_template_sha256": template_sha, "chat_template_sha256_per_variant": templates,
              "chat_template_sha256_task_a": task_a.get("chat_template_sha256"),
              "variants": {}, "expectations": {}, "expectation_violated": []}
    controls = sorted(Path(RUNS / "steps").glob("controls_*.json"))
    result["merge_base_weights"] = read_json(controls[-1], {}).get("base_weights") if controls else None
    out_json = RUNS / f"verify_{run_id}.json"
    gen_path = LOGS / f"verify_generations_{run_id}.jsonl"

    def expect(name, condition, message, **values):
        try:
            assert condition, message
            result["expectations"][name] = {"ok": True, **values}
        except AssertionError as exc:
            result["expectations"][name] = {"ok": False, "message": str(exc), **values}
            result["expectation_violated"].append(name)
            log(f"[verify] EXPECTATION VIOLATED {name}: {exc}")

    def save():
        result.update(run_record(SEED))
        write_json(out_json, result)
        shutil.copyfile(out_json, RUNS / "verify.json")

    expect("template_matches_training",
           (not task_a or task_a.get("chat_template_sha256") == template_sha)
           and len(set(templates.values())) == 1
           and cross.get("input_ids_differ_from_soup", 0) == 0
           and cross.get("answer_not_in_soup_loss", 0) == 0,
           "chat template or tokenization differs from training (task A / Soup labels)",
           task_a_present=bool(task_a))

    with SmiLogger(LOGS / f"nvidia_smi_verify_{run_id}.csv") as smi:
        for name, path in VARIANTS.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                         device_map={"": 0}).eval()
            entry = {"path": path if name == "base" else rel(path),
                     "memory_footprint_mib": round(model.get_memory_footprint() / 2**20, 1)}
            entry.update(score(model, tok, examples, torch))
            log(f"[verify] {name}: loss {entry['loss']:.5f}, ppl {entry['perplexity']:.3f}, "
                f"bpc {entry['bits_per_char']:.4f}")
            result["variants"][name] = entry

            if name == "zero_adapter":
                base = result["variants"]["base"]["loss"]
                rel_diff = abs(entry["loss"] - base) / base
                expect("zero_adapter_equals_base", rel_diff <= ZERO_REL_TOL,
                       f"zero adapter loss {entry['loss']:.6f} vs base {base:.6f}: relative "
                       f"difference {rel_diff:.2e} > {ZERO_REL_TOL}",
                       relative_difference=rel_diff, tolerance=ZERO_REL_TOL)
                if rel_diff > ZERO_REL_TOL:
                    result["stopped"] = "zero_adapter does not match base; stopped before the rest"
                    save()
                    sys.exit(2)

            entry.update(generate_all(model, tok, examples, torch, args.gen_batch,
                                      args.max_new_tokens, name, run_id, gen_path))
            entry.update(torch_peaks(torch))
            entry["wall_s"] = round(time.perf_counter() - started, 2)
            save()
            del model
    result["nvidia_smi"] = smi.summary()
    result["generations"] = rel(gen_path)

    v = {name: e["loss"] for name, e in result["variants"].items()}
    expect("merged_loss_below_base", v["merged"] < v["base"],
           f"loss(merged) {v['merged']:.5f} is not below loss(base) {v['base']:.5f}")
    low, high = sorted((v["merged"], v["base_4bit"]))
    expect("merged_4bit_between_merged_and_base_4bit", low <= v["merged_4bit"] <= high,
           f"loss(merged_4bit) {v['merged_4bit']:.5f} is outside [merged {v['merged']:.5f}, "
           f"base_4bit {v['base_4bit']:.5f}]")

    adapter_cfg = read_json(ARTIFACTS / "adapter" / "adapter_config.json", {})
    targets = adapter_cfg.get("target_modules") or []
    targets = [targets] if isinstance(targets, str) else list(targets)
    codes = nf4_code_diff(ARTIFACTS / "base_4bit", ARTIFACTS / "merged_4bit", targets)
    result["nf4_codes_base_4bit_vs_merged_4bit"] = codes
    expect("nf4_non_target_modules_unchanged",
           codes["non_targets"]["changed"] == 0 and not codes["non_quantized_tensors_changed"],
           f"{codes['non_targets']['changed']} NF4 codes changed outside the LoRA targets {targets}; "
           f"non-quantized tensors changed: {list(codes['non_quantized_tensors_changed'])[:5]}")
    save()

    print(f"\n{'variant':14s} {'loss':>9s} {'ppl':>9s} {'bpc':>8s} {'tokens':>7s} {'eos':>6s} "
          f"{'len':>7s} {'loop':>6s} {'alloc MiB':>10s}")
    for name, e in result["variants"].items():
        print(f"{name:14s} {e['loss']:9.5f} {e['perplexity']:9.3f} {e['bits_per_char']:8.4f} "
              f"{e['answer_tokens']:7d} {e.get('eos_share', float('nan')):6.3f} "
              f"{e.get('mean_new_tokens', float('nan')):7.1f} {e.get('loop_share', float('nan')):6.3f} "
              f"{e.get('torch_max_allocated_mib') or float('nan'):10.1f}")
    print("\nNF4 codes changed, base_4bit vs merged_4bit:")
    for kind, e in codes["per_module_type"].items():
        mark = "target" if kind in targets else "other "
        print(f"  {mark} {kind:10s} {e['changed_share']:.6f} of {e['codes']} codes "
              f"(absmax blocks changed {e['absmax_changed_share']:.6f})")
    print(f"  targets {codes['targets']['changed_share']}, non-targets "
          f"{codes['non_targets']['changed_share']}, overall {codes['overall']['changed_share']}")
    print(f"\nexpectation_violated: {result['expectation_violated']}")
    log(f"[verify] -> {rel(out_json)} (copied to runs/verify.json)")


if __name__ == "__main__":
    main()
