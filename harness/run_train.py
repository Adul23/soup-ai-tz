"""Prepare the data and run plain `soup train` on task/task.yaml, with nvidia-smi in the background.

    python harness/run_train.py [--run-id 20261110T093000Z] [--skip-prepare]

What it does:
1. python task/prepare_data.py (unless --skip-prepare).
2. Writes the run's config: task/task.yaml with `output` set to the adapter folder.
3. Runs `soup train` through soup_train_inproc.py (same process as Soup, for torch peaks) while
   nvidia_smi_logger.py samples the GPU every 200 ms.
4. Writes the step JSON: exit code, wall time, nvidia-smi and torch peaks, trainer settings,
   loss log, adapter sha256, run record (git commit, versions, seed, data sha256).
Exits non-zero when any of it fails; the logs stay.
Reads: task/task.yaml, data/train.jsonl.
Writes: artifacts/adapter/, logs/prepare_data_<run_id>.log, logs/train_<run_id>.log,
logs/train_<run_id>.yaml, logs/nvidia_smi_train_<run_id>.csv, runs/steps/train_<run_id>.json.
Task part: Part 1 (real training peak) and part 2 (the trained adapter and its loss curve).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARTIFACTS, LOGS, REPO, STEPS, SmiLogger, dir_sha256, dir_size_mib, log,  # noqa: E402
                    must_not_exist, new_run_id, read_json, rel, run_logged, run_record, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--adapter", default=str(ARTIFACTS / "adapter"))
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    import yaml

    run_id = args.run_id or new_run_id()
    adapter = Path(args.adapter)
    train_log = LOGS / f"train_{run_id}.log"
    out_json = STEPS / f"train_{run_id}.json"
    must_not_exist(adapter)
    must_not_exist(train_log)

    result = {"step": "train", "run_id": run_id}
    if not args.skip_prepare:
        code, wall = run_logged([sys.executable, "task/prepare_data.py"],
                                LOGS / f"prepare_data_{run_id}.log")
        result["prepare_data"] = {"exit_code": code, "wall_s": wall,
                                  "log": rel(LOGS / f"prepare_data_{run_id}.log")}
        if code != 0:
            write_json(out_json, {**result, "exit_code": code, **run_record()})
            sys.exit(f"prepare_data failed (exit {code})")

    config = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    config["output"] = "./" + rel(adapter)
    config_path = LOGS / f"train_{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                           encoding="utf-8")
    seed = config["training"].get("seed")
    log(f"[run_train] {run_id}: soup train, config {rel(config_path)}, adapter -> {rel(adapter)}")

    inproc_json = STEPS / f"soup_train_{run_id}.json"
    with SmiLogger(LOGS / f"nvidia_smi_train_{run_id}.csv") as smi:
        code, wall = run_logged([sys.executable, "harness/soup_train_inproc.py",
                                 "--config", rel(config_path), "--out", rel(inproc_json)], train_log)
    inner = read_json(inproc_json, {})
    losses = [entry["loss"] for entry in inner.get("log_history", []) if "loss" in entry]
    result.update({
        "exit_code": code, "wall_s": wall, "log": rel(train_log), "config": rel(config_path),
        "config_used": config, **smi.summary(),
        "torch_max_allocated_mib": inner.get("torch_max_allocated_mib"),
        "torch_max_reserved_mib": inner.get("torch_max_reserved_mib"),
        "cuda_device": inner.get("cuda_device"), "device_count": inner.get("device_count"),
        "soup_compute_dtype": inner.get("soup_compute_dtype"),
        "trainer": inner.get("trainer"), "max_steps": inner.get("max_steps"),
        "logged_steps": len(losses), "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "adapter": rel(adapter), "adapter_size_mib": dir_size_mib(adapter),
        "adapter_sha256": dir_sha256(adapter), "inproc_json": rel(inproc_json),
        **run_record(seed),
    })
    write_json(out_json, result)
    log(f"[run_train] exit {code}, {wall} s, nvidia-smi peak {smi.peak_mib} MiB, torch allocated "
        f"{result['torch_max_allocated_mib']} MiB, loss {result['first_loss']} -> {result['last_loss']}")
    if code != 0 or not (adapter / "adapter_config.json").exists():
        sys.exit(f"training failed (exit {code}); see {rel(train_log)}")


if __name__ == "__main__":
    main()
