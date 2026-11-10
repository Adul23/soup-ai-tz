"""Poll nvidia-smi every N ms into a file with timestamps.

    python harness/nvidia_smi_logger.py --out logs/nvidia_smi_train_<run_id>.csv --interval-ms 200

What it does: runs as a separate process next to the measured one (so measuring does not
allocate GPU memory in the measured process), runs `nvidia-smi -lms N` and prefixes every sample
line with a UTC timestamp. Stops on SIGTERM / SIGINT.
Reads: nvidia-smi output.
Writes: the --out file, one line per GPU per sample:
    utc_timestamp, gpu_index, memory.used MiB, memory.total MiB, utilization.gpu %
Task part: Part 1 (the measured peak that the memory budget is compared with).
"""

import argparse
import datetime
import signal
import subprocess
import sys
from pathlib import Path

FIELDS = "index,memory.used,memory.total,utilization.gpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=None, help="only this GPU (nvidia-smi -i)")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    only = ["-i", str(args.gpu)] if args.gpu is not None else []
    proc = subprocess.Popen(["nvidia-smi", *only, f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                             "-lms", str(args.interval_ms)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def stop(*_):
        proc.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    with open(out, "a", encoding="utf-8") as handle:
        handle.write(f"# utc_timestamp,{FIELDS} (MiB, %); interval {args.interval_ms} ms\n")
        for line in proc.stdout:
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
            handle.write(f"{stamp},{line.strip()}\n")
            handle.flush()
    sys.exit(0 if proc.wait() in (0, -signal.SIGTERM) else proc.returncode)


if __name__ == "__main__":
    main()
