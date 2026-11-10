"""Task 1: run the whole chain once, as simply as possible, and collect runs/smoke.json.

    python harness/smoke.py --base-weights fp16|nf4-dequant

What it does: runs, each in its own process and with its console output in a timestamped log,
run_train.py (prepare data + soup train), merge.py, quantize.py, smoke_generate.py; stops at
the first failure. Then puts the step JSONs together: wall time of every step, nvidia-smi and
torch peaks, git commit, versions, seed, data sha256. The summary is written even when a step
failed, with that step's exit code.
Reads: runs/steps/<step>_<run_id>.json written by the steps.
Writes: logs/<step>_console_<run_id>.log, runs/smoke_<run_id>.json, runs/smoke.json (a copy of
the latest), plus everything the steps write.
Task part: task 1 (end-to-end chain); the numbers feed parts 1 and 2.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LOGS, RUNS, STEPS, log, new_run_id, read_json, rel, run_logged,  # noqa: E402
                    run_record, write_json)

KEEP = ("exit_code", "wall_s", "nvidia_smi_peak_mib", "nvidia_smi_baseline_mib",
        "torch_max_allocated_mib", "torch_max_reserved_mib")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-weights", required=True, choices=("fp16", "nf4-dequant"),
                        help="what merge.py merges into; no default on purpose")
    args = parser.parse_args()

    run_id = new_run_id()
    py = sys.executable
    steps = [
        ("train", [py, "harness/run_train.py", "--run-id", run_id]),
        ("merge", [py, "harness/merge.py", "--base-weights", args.base_weights, "--run-id", run_id]),
        ("quantize", [py, "harness/quantize.py", "--run-id", run_id]),
        ("generate", [py, "harness/smoke_generate.py", "--run-id", run_id]),
    ]
    summary = {"run_id": run_id, "base_weights": args.base_weights, "steps": {}, "failed_step": None}
    for name, cmd in steps:
        console = LOGS / f"{name}_console_{run_id}.log"
        log(f"[smoke] {name}: {' '.join(cmd[1:])}")
        code, wall = run_logged(cmd, console)
        step = read_json(STEPS / f"{name}_{run_id}.json", {})
        summary["steps"][name] = {**{k: step.get(k) for k in KEEP}, "exit_code": code,
                                  "process_wall_s": wall, "console_log": rel(console),
                                  "step_json": rel(STEPS / f"{name}_{run_id}.json")}
        if name == "train":
            for key in ("max_steps", "logged_steps", "first_loss", "last_loss", "soup_compute_dtype",
                        "cuda_device", "device_count", "trainer", "adapter_size_mib"):
                summary["steps"][name][key] = step.get(key)
        if name == "merge":
            summary["steps"][name]["lora_modules_merged"] = step.get("lora_modules_merged")
        if name == "quantize":
            for key in ("leaf_module_types", "memory_footprint_mib", "out_size_mib"):
                summary["steps"][name][key] = step.get(key)
        if name == "generate":
            for key in ("identical_merged_vs_base", "identical_4bit_vs_merged",
                        "identical_4bit_vs_base", "chat_template_same_as_base", "per_model", "rows"):
                summary["steps"][name][key] = step.get(key)
        if code != 0:
            summary["failed_step"] = name
            break

    summary.update(run_record(seed=read_json(STEPS / f"train_{run_id}.json", {}).get("seed")))
    write_json(RUNS / f"smoke_{run_id}.json", summary)
    shutil.copyfile(RUNS / f"smoke_{run_id}.json", RUNS / "smoke.json")
    log(f"[smoke] summary -> {rel(RUNS / f'smoke_{run_id}.json')} (copied to runs/smoke.json)")
    if summary["failed_step"]:
        sys.exit(f"step {summary['failed_step']} failed; see {summary['steps'][summary['failed_step']]['console_log']}")


if __name__ == "__main__":
    main()
