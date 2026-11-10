"""Put the verify_artifacts.py part files of one run together into runs/verify.json.

    python verify/merge_verify.py --run-id ID

What it does: reads runs/verify_parts/verify_<run_id>_gpu*.json, checks that the parts used the
same rows, data, template and commit, and computes what compares variants across parts:
- expectations (asserted, never stopping the script; broken ones are listed in
  "expectation_violated"): template matches training; loss(zero_adapter) = loss(base) within
  fp16 tolerance; loss(merged) < loss(base); loss(merged_4bit) between merged and base_4bit;
  no NF4 code changed outside the LoRA target modules;
- a paired bootstrap over rows (1000 resamples, seed 0) of the loss difference
  merged - base, merged_4bit - merged, merged_4bit - base_4bit: 95% interval;
- the share of NF4 codes that differ between artifacts/base_4bit and artifacts/merged_4bit,
  per LoRA target module type, per other module type and overall.
No interpretation: numbers, and a mark where an expectation is broken.
Reads: runs/verify_parts/verify_<run_id>_gpu*.json, artifacts/base_4bit/, artifacts/merged_4bit/,
artifacts/adapter/adapter_config.json.
Writes: runs/verify_<run_id>.json, runs/verify.json (a copy of the latest); a table on the console.
Exits 2 if a part stopped (zero_adapter did not match base).
Task part: Part 2 (proof of training).
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACTS, RUNS, log, read_json, rel, run_record, write_json  # noqa: E402
from verify_artifacts import PARTS, ZERO_REL_TOL, nf4_code_diff  # noqa: E402

BOOTSTRAP = 1000
SEED = 0
PAIRS = (("merged", "base"), ("merged_4bit", "merged"), ("merged_4bit", "base_4bit"))
SAME = ("limit", "heldout_rows", "scored_rows", "max_length", "chat_template_sha256",
        "chat_template_sha256_task_a")


def paired_bootstrap(a, b):
    """95% interval of loss(a) - loss(b), rows resampled with replacement, the same rows for both."""
    assert a["row"] == b["row"], "the two variants were scored on different rows"
    n, rng, diffs = len(a["row"]), random.Random(SEED), []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a["nll"][i] for i in idx) / sum(a["tokens"][i] for i in idx)
                     - sum(b["nll"][i] for i in idx) / sum(b["tokens"][i] for i in idx))
    diffs.sort()
    point = sum(a["nll"]) / sum(a["tokens"]) - sum(b["nll"]) / sum(b["tokens"])
    return {"difference": point, "ci95": [diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]],
            "share_of_resamples_below_0": sum(d < 0 for d in diffs) / BOOTSTRAP,
            "resamples": BOOTSTRAP, "rows": n, "seed": SEED}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    paths = sorted(PARTS.glob(f"verify_{args.run_id}_gpu*.json"))
    assert paths, f"no part files runs/verify_parts/verify_{args.run_id}_gpu*.json"
    parts = [read_json(p, {}) for p in paths]
    result = {"run_id": args.run_id, "parts": [rel(p) for p in paths], "variants": {},
              "expectations": {}, "expectation_violated": [], "part_mismatches": {}}
    for key in SAME:
        values = {str(p.get(key)) for p in parts}
        result[key] = parts[0].get(key)
        if len(values) > 1:
            result["part_mismatches"][key] = sorted(values)
    for key in ("git", "data_sha256"):
        if len({str(p.get(key)) for p in parts}) > 1:
            result["part_mismatches"][key] = [p.get(key) for p in parts]
    result["generation"] = parts[0].get("generation")
    result["stopped"] = [p["stopped"] for p in parts if p.get("stopped")]
    templates, cross = {}, {}
    for p in parts:
        templates.update(p.get("chat_template_sha256_per_variant", {}))
        for k, v in (p.get("prompt_construction_vs_soup_labels") or {}).items():
            cross[k] = max(cross.get(k, 0), v)
        for name, entry in p["variants"].items():
            assert name not in result["variants"], f"variant {name} is in two parts"
            result["variants"][name] = {**entry, "device": p["device"], "cuda_device": p.get("cuda_device")}
    result["prompt_construction_vs_soup_labels"] = cross
    result["chat_template_sha256_per_variant"] = templates
    result["nvidia_smi_per_part"] = {f"gpu{p['device']}": p.get("nvidia_smi") for p in parts}
    v = result["variants"]

    def expect(name, needed, check):
        """check() returns (condition, message); not evaluated when a variant is missing."""
        absent = [n for n in needed if n not in v]
        if absent:
            result["expectations"][name] = {"ok": None, "message": f"not evaluated: missing {absent}"}
            return
        condition, message, values = check()
        try:
            assert condition, message
            result["expectations"][name] = {"ok": True, **values}
        except AssertionError as exc:
            result["expectations"][name] = {"ok": False, "message": str(exc), **values}
            result["expectation_violated"].append(name)
            log(f"[merge_verify] EXPECTATION VIOLATED {name}: {exc}")

    task_a = result.get("chat_template_sha256_task_a")
    expect("template_matches_training", [], lambda: (
        (task_a is None or task_a == result["chat_template_sha256"])
        and len(set(templates.values()) | {result["chat_template_sha256"]}) == 1
        and cross.get("input_ids_differ_from_soup", 0) == 0 and cross.get("answer_not_in_soup_loss", 0) == 0
        and not result["part_mismatches"],
        "chat template / tokenization differs from training (task A, Soup labels) or the parts differ",
        {"task_a_present": task_a is not None}))

    def zero_check():
        diff = abs(v["zero_adapter"]["loss"] - v["base"]["loss"]) / v["base"]["loss"]
        return (diff <= ZERO_REL_TOL,
                f"loss(zero_adapter) {v['zero_adapter']['loss']:.6f} vs loss(base) {v['base']['loss']:.6f}: "
                f"relative difference {diff:.2e} > {ZERO_REL_TOL}",
                {"relative_difference": diff, "tolerance": ZERO_REL_TOL})
    expect("zero_adapter_equals_base", ["zero_adapter", "base"], zero_check)
    zero = v.get("zero_adapter")
    if zero and zero.get("generation") == "skipped (--skip-gen-for)" \
            and result["expectations"]["zero_adapter_equals_base"]["ok"]:
        zero["generation"] = "generation skipped: identical loss to base"

    expect("merged_loss_below_base", ["merged", "base"], lambda: (
        v["merged"]["loss"] < v["base"]["loss"],
        f"loss(merged) {v['merged']['loss']:.5f} is not below loss(base) {v['base']['loss']:.5f}", {}))

    def between():
        low, high = sorted((v["merged"]["loss"], v["base_4bit"]["loss"]))
        return (low <= v["merged_4bit"]["loss"] <= high,
                f"loss(merged_4bit) {v['merged_4bit']['loss']:.5f} is outside [merged "
                f"{v['merged']['loss']:.5f}, base_4bit {v['base_4bit']['loss']:.5f}]", {})
    expect("merged_4bit_between_merged_and_base_4bit", ["merged_4bit", "merged", "base_4bit"], between)

    result["paired_bootstrap"] = {f"{a}-{b}": paired_bootstrap(v[a]["per_row"], v[b]["per_row"])
                                  for a, b in PAIRS if a in v and b in v}

    if (ARTIFACTS / "base_4bit").is_dir() and (ARTIFACTS / "merged_4bit").is_dir():
        targets = read_json(ARTIFACTS / "adapter" / "adapter_config.json", {}).get("target_modules") or []
        targets = [targets] if isinstance(targets, str) else list(targets)
        codes = nf4_code_diff(ARTIFACTS / "base_4bit", ARTIFACTS / "merged_4bit", targets)
        result["nf4_codes_base_4bit_vs_merged_4bit"] = codes
        expect("nf4_non_target_modules_unchanged", [], lambda: (
            codes["non_targets"]["changed"] == 0 and not codes["non_quantized_tensors_changed"],
            f"{codes['non_targets']['changed']} NF4 codes changed outside the LoRA targets {targets}; "
            f"non-quantized tensors changed: {list(codes['non_quantized_tensors_changed'])[:5]}", {}))
    else:
        codes = None
        result["expectations"]["nf4_non_target_modules_unchanged"] = {
            "ok": None, "message": "not evaluated: artifacts/base_4bit or merged_4bit missing"}

    result.update(run_record(SEED))
    out = RUNS / f"verify_{args.run_id}.json"
    write_json(out, result)
    shutil.copyfile(out, RUNS / "verify.json")

    nan = float("nan")
    print(f"\n{'variant':14s} {'gpu':>3s} {'loss':>9s} {'ppl':>9s} {'bpc':>8s} {'tokens':>7s} "
          f"{'eos':>6s} {'len':>7s} {'loop':>6s} {'b1≠b':>6s} {'alloc MiB':>10s}")
    for name, e in v.items():
        b1 = e.get("batch1_vs_batch")
        print(f"{name:14s} {e['device']:3d} {e['loss']:9.5f} {e['perplexity']:9.3f} "
              f"{e['bits_per_char']:8.4f} {e['answer_tokens']:7d} {e.get('eos_share', nan):6.3f} "
              f"{e.get('mean_new_tokens', nan):7.1f} {e.get('loop_share', nan):6.3f} "
              f"{(str(b1['rows_differing']) + '/' + str(b1['rows'])) if b1 else '-':>6s} "
              f"{e.get('torch_max_allocated_mib') or nan:10.1f}")
    print("\npaired bootstrap, loss difference [95% interval]:")
    for pair, bs in result["paired_bootstrap"].items():
        print(f"  {pair:24s} {bs['difference']:+.5f}  [{bs['ci95'][0]:+.5f}, {bs['ci95'][1]:+.5f}]")
    if codes:
        print("\nNF4 codes changed, base_4bit vs merged_4bit:")
        for kind, e in codes["per_module_type"].items():
            mark = "target" if kind in codes["lora_targets"] else "other "
            print(f"  {mark} {kind:10s} {e['changed_share']:.6f} of {e['codes']} codes "
                  f"(absmax blocks changed {e['absmax_changed_share']:.6f})")
        print(f"  targets {codes['targets']['changed_share']}, non-targets "
              f"{codes['non_targets']['changed_share']}, overall {codes['overall']['changed_share']}")
    if result["part_mismatches"]:
        print(f"\nPART MISMATCHES: {result['part_mismatches']}")
    print(f"\nexpectation_violated: {result['expectation_violated']}")
    log(f"[merge_verify] -> {rel(out)} (copied to runs/verify.json)")
    if result["stopped"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
