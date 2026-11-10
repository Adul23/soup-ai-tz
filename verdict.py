"""SHIP / DON'T SHIP from the numbers the other scripts wrote. Nothing is measured here.

    python verdict.py        # -> runs/verdict.json

RULES (fixed; thresholds change only with the author's consent):
  trained       = CI95(loss merged - loss base) lies entirely below 0
                  (verify.json paired_bootstrap["merged-base"].ci95[1] < 0)
  survives_4bit = CI95(merged_4bit - base_4bit) lies entirely below 0
                  AND CI95(merged_4bit - merged) high < +0.02 (in the worst case 4-bit loses at
                  most 0.02 nats per answer token against merged)
  usable        = loop_share(merged_4bit) <= 0.10 AND eos_share(merged_4bit) >= 0.90
                  (verify.json variants.merged_4bit: greedy generation on the held-out rows; loop
                  = the same 4-word n-gram 3+ times. This is the only source: the 3-gram loop share
                  of checks/language.py is copied as context and is not in the rule)
  SHIP only if all three are true; otherwise DON'T SHIP with every failed rule and its numbers.
CI95 = the paired bootstrap interval of verify/merge_verify.py (1000 resamples over rows).

Inputs: runs/verify.json, runs/checks/language.json, runs/checks/silent_failures.json,
runs/memory_budget.json. The rules use verify.json only; the other three files are required,
and their headline numbers and sha256 are copied into the verdict so it names exactly what it
was decided on. Refuses (exit 1) when an input is missing or verify.json is a --limit run.
Writes: runs/verdict.json (+ verdict_<run_id>.json).
Task part: verdict.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
from common import RUNS, file_sha256, log, new_run_id, read_json, rel, run_record, write_json  # noqa: E402

LOOP_MAX = 0.10
EOS_MIN = 0.90
DEGRADATION_MAX = 0.02
INPUTS = {
    "verify": RUNS / "verify.json",
    "language": RUNS / "checks" / "language.json",
    "silent_failures": RUNS / "checks" / "silent_failures.json",
    "memory_budget": RUNS / "memory_budget.json",
}


def ci(verify, pair):
    bs = verify["paired_bootstrap"][pair]
    return {"pair": pair, "difference": bs["difference"], "ci95": bs["ci95"]}


def main():
    run_id = new_run_id()
    data = {name: read_json(path) for name, path in INPUTS.items()}
    missing = [rel(INPUTS[n]) for n, d in data.items() if d is None]
    if missing:
        sys.exit(f"missing inputs {missing}: no verdict")
    verify = data["verify"]
    if verify.get("limit") is not None:
        sys.exit(f"runs/verify.json is a --limit {verify['limit']} run: no verdict on a subset")

    rules = {}
    t = ci(verify, "merged-base")
    rules["trained"] = {"ok": t["ci95"][1] < 0, "rule": "CI95(merged - base) high < 0", **t}

    a = ci(verify, "merged_4bit-base_4bit")
    b = ci(verify, "merged_4bit-merged")
    small_loss = b["ci95"][1] < DEGRADATION_MAX
    rules["survives_4bit"] = {
        "ok": a["ci95"][1] < 0 and small_loss,
        "rule": f"CI95(merged_4bit - base_4bit) high < 0 AND CI95(merged_4bit - merged) high < "
                f"+{DEGRADATION_MAX}",
        "merged_4bit-base_4bit": a, "merged_4bit-merged": b,
        "parts": {"merged_4bit_below_base_4bit": a["ci95"][1] < 0,
                  f"merged_4bit_vs_merged_high_below_{DEGRADATION_MAX}": small_loss}}

    m4 = verify["variants"]["merged_4bit"]
    loop, eos = m4.get("loop_share"), m4.get("eos_share")
    rules["usable"] = {
        "ok": loop is not None and eos is not None and loop <= LOOP_MAX and eos >= EOS_MIN,
        "rule": f"merged_4bit loop_share <= {LOOP_MAX} AND eos_share >= {EOS_MIN} (verify.json)",
        "loop_share": loop, "eos_share": eos, "generated_rows": m4.get("generated_rows"),
        "loop_definition": "verify_artifacts.py: the same 4-word n-gram 3+ times"}

    failed = [name for name, r in rules.items() if not r["ok"]]
    lang = data["language"]["groups"].get("generations_merged_4bit", {})
    silent = data["silent_failures"]
    budget = data["memory_budget"]
    result = {
        "run_id": run_id,
        "verdict": "SHIP" if not failed else "DON'T SHIP",
        "failed_rules": {name: rules[name] for name in failed},
        "rules": rules,
        "thresholds": {"loop_share_max": LOOP_MAX, "eos_share_min": EOS_MIN,
                       "merged_4bit_minus_merged_ci_high_max": DEGRADATION_MAX},
        "inputs": {name: {"path": rel(path), "sha256": file_sha256(path),
                          "run_id": data[name].get("run_id")} for name, path in INPUTS.items()},
        "context_not_in_rules": {
            "verify_scored_rows": verify.get("scored_rows"),
            "verify_expectation_violated": verify.get("expectation_violated"),
            "language_merged_4bit": {k: lang.get(k) for k in
                                     ("loop_share", "repeat_pair_share", "kyrgyz_per_100_words_total",
                                      "no_space_after_period_share")},
            "silent_failures_summary": silent.get("summary"),
            "silent_failures_crashed": silent.get("crashed"),
            "silent_failures_our_check_missed": silent.get("our_check_missed"),
            "memory_budget_train_plain_mib": (budget.get("train_plain") or {}).get("torch_allocated_peak_mib"),
            "memory_budget_compare": budget.get("compare"),
        },
    }
    result.update(run_record())
    out = RUNS / f"verdict_{run_id}.json"
    write_json(out, result)
    shutil.copyfile(out, RUNS / "verdict.json")

    print(f"\nVERDICT: {result['verdict']}")
    for name, r in rules.items():
        print(f"  {'ok  ' if r['ok'] else 'FAIL'} {name:14s} {r['rule']}")
    print(f"    trained: merged-base {t['difference']:+.5f} [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}]")
    print(f"    survives_4bit: merged_4bit-base_4bit {a['difference']:+.5f} [{a['ci95'][0]:+.5f}, "
          f"{a['ci95'][1]:+.5f}]; merged_4bit-merged {b['difference']:+.5f} [{b['ci95'][0]:+.5f}, "
          f"{b['ci95'][1]:+.5f}]")
    print(f"    usable: loop_share {loop}, eos_share {eos}")
    log(f"[verdict] {result['verdict']} -> {rel(out)} (copied to runs/verdict.json)")


if __name__ == "__main__":
    main()
