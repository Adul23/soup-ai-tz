"""Run verification on two GPUs at once (Kaggle T4 x2), wait for both, then merge.

    python verify/run_parallel.py [--limit 20] [--gen-batch 16]

What it does:
1. Builds the controls with verify/build_controls.py if artifacts/zero_merged or
   artifacts/base_4bit is missing.
2. Starts two verify_artifacts.py processes with the same run id:
   GPU 0: base, zero_adapter (loss only), merged;  GPU 1: base_4bit, merged_4bit.
   Each one's output goes to its own timestamped log and to the console with a [gpuN] prefix.
3. Waits for both, then runs verify/merge_verify.py (also when a part failed, so the failure is
   in runs/verify.json), and exits non-zero if anything failed.
Reads/writes: see build_controls.py, verify_artifacts.py, merge_verify.py; plus
logs/verify_gpu<N>_<run_id>.log, logs/verify_merge_<run_id>.log.
Task part: Part 2 (proof of training).
"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import ARTIFACTS, LOGS, log, new_run_id, run_logged  # noqa: E402

SPLIT = {0: ("base,zero_adapter,merged", "zero_adapter"), 1: ("base_4bit,merged_4bit", "")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-batch", type=int, default=16)
    args = parser.parse_args()

    run_id = new_run_id()
    py = sys.executable
    if not ((ARTIFACTS / "zero_merged").is_dir() and (ARTIFACTS / "base_4bit").is_dir()):
        code, _ = run_logged([py, "verify/build_controls.py", "--run-id", run_id],
                             LOGS / f"verify_controls_{run_id}.log")
        if code != 0:
            sys.exit(f"build_controls failed (exit {code})")

    codes = {}

    def part(device, variants, skip):
        cmd = [py, "verify/verify_artifacts.py", "--run-id", run_id, "--device", str(device),
               "--variants", variants, "--gen-batch", str(args.gen_batch)]
        if skip:
            cmd += ["--skip-gen-for", skip]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        codes[device], _ = run_logged(cmd, LOGS / f"verify_gpu{device}_{run_id}.log",
                                      echo_prefix=f"[gpu{device}] ")

    threads = [threading.Thread(target=part, args=(d, v, s)) for d, (v, s) in SPLIT.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"[run_parallel] parts finished: exit codes {codes}")
    merge_code, _ = run_logged([py, "verify/merge_verify.py", "--run-id", run_id],
                               LOGS / f"verify_merge_{run_id}.log")
    if any(codes.values()) or merge_code:
        sys.exit(f"verification failed: parts {codes}, merge {merge_code}")


if __name__ == "__main__":
    main()
