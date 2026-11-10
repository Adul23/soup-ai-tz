"""Build the two control artifacts for verify_artifacts.py.

    python verify/build_controls.py [--run-id ...]

What it does:
- base_4bit: the base model quantized by harness/quantize.py, the same script and config that
  made artifacts/merged_4bit (NF4, double quantization, fp16 compute).
- zero_adapter: a copy of artifacts/adapter with every lora_B tensor set to zero (so B @ A = 0),
  merged by harness/merge.py with the same --base-weights that artifacts/merged was merged with
  (read from that merge's step JSON, never guessed).
Reads: artifacts/adapter/, runs/steps/merge_*.json (the one that wrote artifacts/merged).
Writes: artifacts/base_4bit/, artifacts/zero_adapter/, artifacts/zero_merged/,
logs/build_controls_*_<run_id>.log, runs/steps/controls_<run_id>.json (plus the step JSONs of
quantize.py and merge.py).
Task part: Part 2 (proof of training): the controls of the three-artifact comparison.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (ARTIFACTS, LOGS, MODEL, STEPS, dir_sha256, log, must_not_exist,  # noqa: E402
                    new_run_id, read_json, rel, run_logged, run_record, write_json)


def merged_base_weights():
    """--base-weights of the merge that produced artifacts/merged (latest such step JSON)."""
    found = [read_json(p, {}) for p in sorted(STEPS.glob("merge_*.json"))]
    found = [s for s in found if s.get("out") == "artifacts/merged" and s.get("exit_code") == 0]
    if not found:
        sys.exit("no runs/steps/merge_*.json for artifacts/merged: cannot tell which base weights "
                 "it was merged into, so the zero adapter cannot be merged the same way")
    return found[-1]["base_weights"], found[-1]["run_id"]


def make_zero_adapter(src, dst):
    import torch
    from safetensors.torch import load_file, save_file

    must_not_exist(dst)
    shutil.copytree(src, dst)
    path = dst / "adapter_model.safetensors"
    tensors = load_file(str(path))
    zeroed = [k for k in tensors if "lora_B" in k]
    kept = [k for k in tensors if "lora_A" in k]
    assert zeroed and kept, f"unexpected adapter keys: {list(tensors)[:5]}"
    for key in zeroed:
        tensors[key] = torch.zeros_like(tensors[key])
    save_file(tensors, str(path), metadata={"format": "pt"})
    check = load_file(str(path))
    assert all(not check[k].any() for k in zeroed), "lora_B not zero after saving"
    original = load_file(str(src / "adapter_model.safetensors"))
    assert all(torch.equal(check[k], original[k]) for k in kept), "lora_A changed"
    return {"lora_B_tensors_zeroed": len(zeroed), "lora_A_tensors_kept": len(kept)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or new_run_id()
    py = sys.executable
    result = {"step": "controls", "run_id": run_id}

    base_weights, merge_run = merged_base_weights()
    log(f"[controls] artifacts/merged was merged into base weights = {base_weights} "
        f"(merge run {merge_run}); zero adapter is merged the same way")
    result["zero_adapter"] = make_zero_adapter(ARTIFACTS / "adapter", ARTIFACTS / "zero_adapter")
    result["zero_adapter"]["sha256"] = dir_sha256(ARTIFACTS / "zero_adapter")

    steps = {
        "zero_merged": [py, "harness/merge.py", "--base-weights", base_weights, "--adapter",
                        "artifacts/zero_adapter", "--out", "artifacts/zero_merged", "--run-id",
                        f"{run_id}_zero"],
        "base_4bit": [py, "harness/quantize.py", "--merged", MODEL, "--out", "artifacts/base_4bit",
                      "--run-id", f"{run_id}_base4bit"],
    }
    for name, cmd in steps.items():
        code, wall = run_logged(cmd, LOGS / f"build_controls_{name}_{run_id}.log")
        result[name] = {"exit_code": code, "wall_s": wall,
                        "log": rel(LOGS / f"build_controls_{name}_{run_id}.log")}
        if code != 0:
            write_json(STEPS / f"controls_{run_id}.json", {**result, **run_record()})
            sys.exit(f"{name} failed (exit {code})")
    result.update({"base_weights": base_weights, "merged_from_merge_run": merge_run, **run_record()})
    write_json(STEPS / f"controls_{run_id}.json", result)
    log(f"[controls] done -> artifacts/zero_merged, artifacts/base_4bit")


if __name__ == "__main__":
    main()
