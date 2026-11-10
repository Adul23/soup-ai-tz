"""Quantize the merged model to 4 bits (bitsandbytes NF4).

    python harness/quantize.py [--run-id ...]

What it does: loads artifacts/merged/ with the same NF4 config Soup trains with (NF4, double
quantization, fp16 compute) and saves the quantized model; counts which layers became 4-bit
and which stayed fp16 (embedding, norms, the tied lm_head).
Reads: artifacts/merged/.
Writes: artifacts/merged_4bit/, logs/nvidia_smi_quantize_<run_id>.csv,
runs/steps/quantize_<run_id>.json. Its console output is logged by the caller (smoke.py).
Task part: produces the `4-bit` artifact checked in part 2; quantize peak for part 1.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARTIFACTS, LOGS, STEPS, SmiLogger, bnb_nf4_config, dir_sha256, dir_size_mib,  # noqa: E402
                    log, must_not_exist, new_run_id, rel, run_record, torch_peaks, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", default=str(ARTIFACTS / "merged"))
    parser.add_argument("--out", default=str(ARTIFACTS / "merged_4bit"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or new_run_id()
    merged, out = Path(args.merged), Path(args.out)
    must_not_exist(out)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = bnb_nf4_config(torch)
    log(f"[quantize] {rel(merged)} -> {rel(out)} with {config.to_dict()}")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with SmiLogger(LOGS / f"nvidia_smi_quantize_{run_id}.csv") as smi:
        model = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.float16,
                                                     quantization_config=config, device_map={"": 0})
        kinds = {}
        for _name, module in model.named_modules():
            if not list(module.children()):
                kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
        out.mkdir(parents=True)
        model.save_pretrained(str(out))
        AutoTokenizer.from_pretrained(str(merged)).save_pretrained(str(out))
        torch.cuda.synchronize()
    wall = round(time.perf_counter() - started, 2)

    result = {"step": "quantize", "run_id": run_id, "exit_code": 0, "wall_s": wall,
              "merged": rel(merged), "out": rel(out), "bnb_config": config.to_dict(),
              "leaf_module_types": kinds,
              "memory_footprint_mib": round(model.get_memory_footprint() / 2**20, 1),
              "out_size_mib": dir_size_mib(out), "out_sha256": dir_sha256(out),
              **smi.summary(), **torch_peaks(torch), **run_record()}
    result["bnb_config"] = {k: str(v) for k, v in result["bnb_config"].items()}
    write_json(STEPS / f"quantize_{run_id}.json", result)
    log(f"[quantize] done in {wall} s, layers {kinds}, {result['out_size_mib']} MiB on disk, "
        f"nvidia-smi peak {smi.peak_mib} MiB, torch allocated {result['torch_max_allocated_mib']} MiB")


if __name__ == "__main__":
    main()
