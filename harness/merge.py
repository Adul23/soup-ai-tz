"""Merge the LoRA adapter into the base model, in fp16.

    python harness/merge.py --base-weights fp16|nf4-dequant [--run-id ...]

What it does: loads the base model, applies the adapter with peft, merges it
(merge_and_unload) and saves a standalone fp16 model with the tokenizer.
--base-weights has no default on purpose, and the choice is written to the log and the JSON:
- fp16: the original fp16 weights of Qwen2.5-1.5B-Instruct (what `soup merge` does).
- nf4-dequant: the base loaded in NF4 exactly as Soup trained it, dequantized to fp16, then
  merged: the weights the adapter was actually trained against.
Reads: artifacts/adapter/, the base model from Hugging Face.
Writes: artifacts/merged/ (*.safetensors, config, tokenizer), logs/nvidia_smi_merge_<run_id>.csv,
runs/steps/merge_<run_id>.json. Its console output is logged by the caller (smoke.py).
Task part: produces the `merged` artifact checked in part 2; merge peak for part 1.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARTIFACTS, LOGS, MODEL, STEPS, SmiLogger, bnb_nf4_config, dir_sha256,  # noqa: E402
                    dir_size_mib, log, must_not_exist, new_run_id, read_json, rel, run_record,
                    torch_peaks, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-weights", required=True, choices=("fp16", "nf4-dequant"))
    parser.add_argument("--adapter", default=str(ARTIFACTS / "adapter"))
    parser.add_argument("--out", default=str(ARTIFACTS / "merged"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or new_run_id()
    adapter, out = Path(args.adapter), Path(args.out)
    must_not_exist(out)
    log(f"[merge] MERGE TARGET: base weights = {args.base_weights} "
        + ("(original fp16 checkpoint)" if args.base_weights == "fp16"
           else "(NF4 as in training, dequantized to fp16)"))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with SmiLogger(LOGS / f"nvidia_smi_merge_{run_id}.csv") as smi:
        if args.base_weights == "fp16":
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                         device_map={"": 0})
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                         quantization_config=bnb_nf4_config(torch),
                                                         device_map={"": 0})
            model = model.dequantize() or model
            assert getattr(model.config, "quantization_config", None) is None, \
                "config still says quantized after dequantize(); the saved model would be wrong"
            leftover = [n for n, m in model.named_modules() if "4bit" in type(m).__name__]
            assert not leftover, f"still quantized after dequantize(): {leftover[:5]}"
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        log(f"[merge] base loaded, parameter dtypes {dtypes}")
        model = PeftModel.from_pretrained(model, str(adapter))
        lora_modules = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
        model = model.merge_and_unload()
        log(f"[merge] merged {lora_modules} LoRA modules")
        out.mkdir(parents=True)
        model.save_pretrained(str(out))
        tokenizer_from = adapter if (adapter / "tokenizer_config.json").exists() else MODEL
        AutoTokenizer.from_pretrained(str(tokenizer_from)).save_pretrained(str(out))
        torch.cuda.synchronize()
    wall = round(time.perf_counter() - started, 2)

    result = {"step": "merge", "run_id": run_id, "exit_code": 0, "wall_s": wall,
              "base_weights": args.base_weights, "base_model": MODEL, "adapter": rel(adapter),
              "adapter_config": read_json(adapter / "adapter_config.json"),
              "lora_modules_merged": lora_modules, "base_param_dtypes": dtypes,
              "tokenizer_from": str(tokenizer_from) if tokenizer_from == MODEL else rel(tokenizer_from),
              "out": rel(out), "out_size_mib": dir_size_mib(out), "out_sha256": dir_sha256(out),
              **smi.summary(), **torch_peaks(torch), **run_record()}
    write_json(STEPS / f"merge_{run_id}.json", result)
    log(f"[merge] done in {wall} s -> {rel(out)} ({result['out_size_mib']} MiB), nvidia-smi peak "
        f"{smi.peak_mib} MiB, torch allocated {result['torch_max_allocated_mib']} MiB")


if __name__ == "__main__":
    main()
