"""Memory experiment: `soup train` with the ported kernels (kernels/), NOT the main chain.

    python harness/run_train_kernels.py [--run-id ...]

What it does (outer process): the same task/task.yaml, seed and data/train.jsonl as the plain run,
with `output` set to artifacts/adapter_kernels; starts nvidia_smi_logger.py on GPU 0 only
(`nvidia-smi -i 0`) and runs this file again with --inner, its output in a timestamped log.
Inner process (as justdoner/harness/_train.py): starts the CUDA context, calls kernels.apply(),
then Soup's own CLI entry point `soup train --config ... --yes` in the same process; a read-only
TrainerCallback records the trainer settings and the per-step logs; afterwards it records
torch.cuda.max_memory_allocated / max_memory_reserved and the kernels' counters (fused-head
training steps, fp16-MLP calls, SwiGLU launches).
A run fails (non-zero exit, the JSON says why) if soup train fails or if the fused head or the
fp16 MLP never engaged (a patch silently fell back to the original code).
Reads: task/task.yaml, data/train.jsonl (not regenerated: the plain run's data).
Writes: artifacts/adapter_kernels/, logs/train_kernels_<run_id>.log,
logs/train_kernels_<run_id>.yaml, logs/nvidia_smi_train_kernels_<run_id>.csv,
runs/steps/train_kernels_<run_id>.json (+ train_kernels_inner_<run_id>.json).
Task part: separate memory experiment (part 1 context); its adapter is never used for the verdict.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ARTIFACTS,
    DATA,
    LOGS,
    REPO,
    STEPS,
    SmiLogger,
    dir_sha256,
    dir_size_mib,
    log,
    must_not_exist,
    new_run_id,
    read_json,
    rel,
    run_logged,
    run_record,
    torch_peaks,
    write_json,
)

TRAINER_ARGS = (
    "fp16",
    "bf16",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "seed",
    "optim",
    "lr_scheduler_type",
    "warmup_steps",
    "gradient_checkpointing",
    "max_length",
    "packing",
    "completion_only_loss",
    "assistant_only_loss",
)


def inner(config, out):
    sys.argv = ["soup", "train", "--config", config, "--yes"]
    import torch

    torch.empty(1, device="cuda")  # CUDA context before any patch
    import transformers

    sys.path.insert(0, str(REPO))
    import kernels

    kernels.apply()
    seen = {"trainer": None, "log_history": [], "max_steps": None}

    class Record(transformers.TrainerCallback):
        def on_train_begin(self, train_args, state, control, **kwargs):
            seen["trainer"] = {
                k: str(getattr(train_args, k, None)) for k in TRAINER_ARGS
            }
            seen["max_steps"] = state.max_steps

        def on_log(self, train_args, state, control, logs=None, **kwargs):
            seen["log_history"].append({"step": state.global_step, **(logs or {})})

    original_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Record())

    transformers.Trainer.__init__ = init
    torch.cuda.reset_peak_memory_stats()
    code = 0
    try:
        from soup_cli.cli import run

        run()
    except SystemExit as exc:
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
    counters = kernels.stats()
    engaged, not_engaged = kernels.engaged(counters)
    write_json(
        out,
        {
            "exit_code": code,
            "kernels_engaged": engaged,
            "not_engaged": not_engaged,
            "counters": counters,
            **torch_peaks(torch),
            "cuda_device": torch.cuda.get_device_name(0),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "trainer": seen["trainer"],
            "max_steps": seen["max_steps"],
            "log_history": seen["log_history"],
        },
    )
    print(
        f"[kernels] counters {counters}; engaged={engaged} {not_engaged or ''}",
        flush=True,
    )
    sys.exit(code or (0 if engaged else 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--adapter", default=str(ARTIFACTS / "adapter_kernels"))
    parser.add_argument(
        "--inner", nargs=2, metavar=("CONFIG", "OUT_JSON"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.inner:
        inner(*args.inner)
        return

    import yaml

    run_id = args.run_id or new_run_id()
    adapter = Path(args.adapter)
    train_log = LOGS / f"train_kernels_{run_id}.log"
    must_not_exist(adapter)
    assert (DATA / "train.jsonl").exists(), (
        "data/train.jsonl missing: run the plain chain first"
    )

    config = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
    config["output"] = "./" + rel(adapter)
    config_path = LOGS / f"train_kernels_{run_id}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    inner_json = STEPS / f"train_kernels_inner_{run_id}.json"
    log(
        f"[run_train_kernels] {run_id}: soup train + kernels.apply(), adapter -> {rel(adapter)}"
    )

    with SmiLogger(LOGS / f"nvidia_smi_train_kernels_{run_id}.csv", gpu=0) as smi:
        code, wall = run_logged(
            [
                sys.executable,
                "harness/run_train_kernels.py",
                "--inner",
                rel(config_path),
                rel(inner_json),
            ],
            train_log,
        )
    inner_result = read_json(inner_json, {})
    losses = [e["loss"] for e in inner_result.get("log_history", []) if "loss" in e]
    result = {
        "step": "train_kernels",
        "run_id": run_id,
        "exit_code": code,
        "wall_s": wall,
        "log": rel(train_log),
        "config": rel(config_path),
        "config_used": config,
        **smi.summary(),
        **{
            k: inner_result.get(k)
            for k in (
                "torch_max_allocated_mib",
                "torch_max_reserved_mib",
                "kernels_engaged",
                "not_engaged",
                "counters",
                "cuda_device",
                "trainer",
                "max_steps",
            )
        },
        "logged_steps": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "inner_json": rel(inner_json),
        "adapter": rel(adapter),
        "adapter_size_mib": dir_size_mib(adapter),
        "adapter_sha256": dir_sha256(adapter),
        **run_record(config["training"].get("seed")),
    }
    write_json(STEPS / f"train_kernels_{run_id}.json", result)
    log(
        f"[run_train_kernels] exit {code}, {wall} s, nvidia-smi peak {smi.peak_mib} MiB, torch "
        f"allocated {result['torch_max_allocated_mib']} MiB, counters {result['counters']}"
    )
    if code != 0:
        sys.exit(
            f"kernels run failed (exit {code}; 3 = a patch never engaged); see {rel(train_log)}"
        )


if __name__ == "__main__":
    main()
