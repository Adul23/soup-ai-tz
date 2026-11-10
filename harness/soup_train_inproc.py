"""Run plain `soup train` in this process and record PyTorch's peak memory. Started by
run_train.py.

    python harness/soup_train_inproc.py --config logs/train_<run_id>.yaml --out runs/steps/soup_train_<run_id>.json

What it does: calls Soup's own CLI entry point (`soup train --config ... --yes`) inside this
process, so that torch.cuda.max_memory_allocated / max_memory_reserved can be read after it
finishes; nvidia-smi alone cannot tell PyTorch's memory from the CUDA context and the cache.
Nothing in Soup, transformers or the model is patched: a read-only TrainerCallback records the
trainer settings (fp16 / bf16, steps) and the per-step logs.
Reads: the config given by --config (task/task.yaml with the output path filled in).
Writes: the adapter at the config's `output`, and the --out JSON.
Task part: Part 1 (torch peak of training) and part 2 (loss curve of the training).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import torch_peaks, write_json  # noqa: E402

TRAINER_ARGS = ("fp16", "bf16", "per_device_train_batch_size", "gradient_accumulation_steps",
                "learning_rate", "num_train_epochs", "max_steps", "seed", "optim",
                "lr_scheduler_type", "warmup_steps", "warmup_ratio", "gradient_checkpointing",
                "max_length", "packing", "completion_only_loss", "assistant_only_loss")


def plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(getattr(value, "value", value))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    import transformers

    seen = {"trainer": None, "log_history": [], "max_steps": None}

    class Record(transformers.TrainerCallback):
        def on_train_begin(self, train_args, state, control, **kwargs):
            seen["trainer"] = {name: plain(getattr(train_args, name, None)) for name in TRAINER_ARGS}
            seen["max_steps"] = state.max_steps

        def on_log(self, train_args, state, control, logs=None, **kwargs):
            seen["log_history"].append({"step": state.global_step, **(logs or {})})

    original_init = transformers.Trainer.__init__

    def init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.add_callback(Record())

    transformers.Trainer.__init__ = init

    torch.cuda.reset_peak_memory_stats()
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    code = 0
    try:
        from soup_cli.cli import run

        run()
    except SystemExit as exc:
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)

    from soup_cli.utils.gpu import cuda_supports_bf16, get_compute_dtype

    write_json(args.out, {
        "exit_code": code,
        **torch_peaks(torch),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "soup_cuda_supports_bf16": cuda_supports_bf16(),
        "soup_compute_dtype": str(get_compute_dtype()),
        "trainer": seen["trainer"],
        "max_steps": seen["max_steps"],
        "log_history": seen["log_history"],
    })
    sys.exit(code)


if __name__ == "__main__":
    main()
