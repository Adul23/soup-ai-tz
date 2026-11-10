"""Memory experiment with the ported kernels: run it and compare with the plain Soup run. Numbers only.

    python harness/kernels_experiment.py

What it does, stopping at the first failure (the JSON is written either way):
1. harness/run_train_kernels.py (unless artifacts/adapter_kernels exists).
2. harness/merge.py artifacts/adapter_kernels -> artifacts/merged_kernels, with the same
   --base-weights as artifacts/merged (from its merge step JSON).
3. verify/verify_artifacts.py --variants merged_kernels (loss / perplexity / bits per character on
   all held-out rows; no generation).
4. runs/kernels_experiment.json:
   - nvidia-smi and torch peaks: plain Soup (latest successful runs/steps/train_<id>.json) vs kernels;
   - the loss curve per logged step: max and mean relative difference |kernels - plain| / plain;
   - held-out loss of the kernels adapter, next to the plain merged loss from runs/verify.json
     (with the row count each was computed on);
   - the kernels' counters; training wall time and train_runtime of both runs.
The main chain (verify, verdict) never uses these artifacts.
Reads: runs/steps/train_*.json, soup_train_*.json, train_kernels_*.json, merge_*.json,
runs/verify.json, runs/verify_parts/.
Writes: artifacts/adapter_kernels/, artifacts/merged_kernels/, logs/kernels_experiment_*_<run_id>.log,
runs/kernels_experiment_<run_id>.json, runs/kernels_experiment.json (a copy of the latest).
Task part: separate memory experiment (part 1 context).
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARTIFACTS, LOGS, RUNS, STEPS, log, new_run_id, read_json, rel,  # noqa: E402
                    run_logged, run_record, write_json)

PEAKS = ("nvidia_smi_peak_mib", "nvidia_smi_baseline_mib", "torch_max_allocated_mib",
         "torch_max_reserved_mib", "wall_s")


def latest(pattern, **match):
    found = [read_json(p, {}) for p in sorted(STEPS.glob(pattern))]
    found = [s for s in found if all(s.get(k) == v for k, v in match.items())]
    return found[-1] if found else None


def curve(history):
    return {e["step"]: e["loss"] for e in history or [] if "loss" in e}


def runtime(history):
    return next((e.get("train_runtime") for e in reversed(history or []) if "train_runtime" in e), None)


def main():
    run_id = new_run_id()
    py = sys.executable
    out = {"experiment": "kernels", "run_id": run_id, "failed_step": None}

    def finish():
        out.update(run_record())
        write_json(RUNS / f"kernels_experiment_{run_id}.json", out)
        shutil.copyfile(RUNS / f"kernels_experiment_{run_id}.json", RUNS / "kernels_experiment.json")
        log(f"[kernels_experiment] -> runs/kernels_experiment_{run_id}.json (copied to runs/kernels_experiment.json)")

    def step(name, cmd):
        code, wall = run_logged(cmd, LOGS / f"kernels_experiment_{name}_{run_id}.log")
        out.setdefault("steps", {})[name] = {"exit_code": code, "wall_s": wall,
                                             "log": rel(LOGS / f"kernels_experiment_{name}_{run_id}.log")}
        if code != 0:
            out["failed_step"] = name
            finish()
            sys.exit(f"{name} failed (exit {code})")

    if not (ARTIFACTS / "adapter_kernels").is_dir():
        step("train", [py, "harness/run_train_kernels.py", "--run-id", run_id])
    merge = latest("merge_*.json", out="artifacts/merged", exit_code=0)
    assert merge, "no merge step JSON for artifacts/merged: run the plain chain first"
    if not (ARTIFACTS / "merged_kernels").is_dir():
        step("merge", [py, "harness/merge.py", "--base-weights", merge["base_weights"], "--adapter",
                       "artifacts/adapter_kernels", "--out", "artifacts/merged_kernels",
                       "--run-id", f"{run_id}_kernels"])
    step("verify", [py, "verify/verify_artifacts.py", "--run-id", f"{run_id}_kernels", "--device", "0",
                    "--variants", "merged_kernels", "--skip-gen-for", "merged_kernels"])

    plain = latest("train_2*.json", step="train", exit_code=0)
    kern = latest("train_kernels_2*.json", step="train_kernels", exit_code=0)
    assert plain and kern, "missing the plain or the kernels training step JSON"
    plain_hist = read_json(Path(plain["inproc_json"]) if Path(plain["inproc_json"]).is_absolute()
                           else STEPS.parents[1] / plain["inproc_json"], {}).get("log_history")
    kern_hist = read_json(STEPS.parents[1] / kern["inner_json"], {}).get("log_history")
    a, b = curve(plain_hist), curve(kern_hist)
    common = sorted(set(a) & set(b))
    rel_diff = [abs(b[s] - a[s]) / abs(a[s]) for s in common if a[s]]
    part = read_json(RUNS / "verify_parts" / f"verify_{run_id}_kernels_gpu0.json", {})
    plain_verify = read_json(RUNS / "verify.json", {})
    kv = part.get("variants", {}).get("merged_kernels", {})
    pv = plain_verify.get("variants", {}).get("merged", {})

    out.update({
        "plain_train_step": f"runs/steps/train_{plain['run_id']}.json",
        "kernels_train_step": f"runs/steps/train_kernels_{kern['run_id']}.json",
        "peaks": {"plain": {k: plain.get(k) for k in PEAKS}, "kernels": {k: kern.get(k) for k in PEAKS}},
        "train_runtime_s": {"plain": runtime(plain_hist), "kernels": runtime(kern_hist)},
        "loss_curve": {"steps_plain": len(a), "steps_kernels": len(b), "steps_compared": len(common),
                       "max_relative_difference": max(rel_diff) if rel_diff else None,
                       "mean_relative_difference": sum(rel_diff) / len(rel_diff) if rel_diff else None,
                       "first_step_relative_difference": rel_diff[0] if rel_diff else None},
        "heldout": {"kernels_merged": {k: kv.get(k) for k in ("loss", "perplexity", "bits_per_char",
                                                               "answer_tokens")},
                    "kernels_scored_rows": part.get("scored_rows"),
                    "plain_merged": {k: pv.get(k) for k in ("loss", "perplexity", "bits_per_char",
                                                             "answer_tokens")},
                    "plain_scored_rows": plain_verify.get("scored_rows"),
                    "plain_verify": "runs/verify.json", "kernels_verify": rel(
                        RUNS / "verify_parts" / f"verify_{run_id}_kernels_gpu0.json")},
        "counters": kern.get("counters"), "kernels_engaged": kern.get("kernels_engaged"),
        "merge_base_weights": merge["base_weights"],
    })
    finish()
    print(f"\n{'':10s} {'smi peak':>9s} {'torch alloc':>12s} {'torch resv':>11s} {'wall s':>8s}")
    for side in ("plain", "kernels"):
        p = out["peaks"][side]
        print(f"{side:10s} {p['nvidia_smi_peak_mib']!s:>9s} {p['torch_max_allocated_mib']!s:>12s} "
              f"{p['torch_max_reserved_mib']!s:>11s} {p['wall_s']!s:>8s}")
    print(f"loss curve: {out['loss_curve']}")
    print(f"held-out: {out['heldout']['kernels_merged']} vs plain {out['heldout']['plain_merged']}")
    print(f"counters: {out['counters']}")


if __name__ == "__main__":
    main()
