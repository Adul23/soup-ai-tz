"""Shared helpers for the harness, memory, verify and checks scripts.

What it does: timestamps and run ids, running a subprocess with a timestamped raw log, the
nvidia-smi logger as a context manager, torch peak memory, sha256 of files and folders, git
commit, library versions, and the run record every step writes.
Reads: data/*.jsonl (for their sha256), git metadata of the repository.
Writes: only what the callers ask for (logs/, runs/); write_json creates parent folders.
Task part: infrastructure for all parts (memory budget, proof of training, silent failures,
verdict).

kill_group, installed_packages, library_versions, file_sha256, read_json and write_json are
copied from justdoner/harness/common.py without changes.
"""

import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # the same `base` as task/task.yaml
DATA = REPO / "data"
ARTIFACTS = REPO / "artifacts"
LOGS = REPO / "logs"
RUNS = REPO / "runs"
STEPS = RUNS / "steps"

# One GPU (Kaggle gives two; TASK.md asks for one T4), unbuffered output, quiet tokenizers.
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")

VERSION_PACKAGES = ("soup-cli", "torch", "triton", "transformers", "peft", "trl", "bitsandbytes",
                    "accelerate", "datasets")

SMI_INTERVAL_MS = 200


def now_iso():
    """UTC wall-clock timestamp with milliseconds, for log lines."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def new_run_id():
    """UTC timestamp used in file names: 20261110T093000Z."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(message, path=None):
    """Print a timestamped line and append it to `path` if given."""
    line = f"{now_iso()} {message}"
    print(line, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def must_not_exist(path):
    """Artifacts and logs are never overwritten or deleted (TASK.md); refuse instead."""
    if Path(path).exists():
        sys.exit(f"{path} already exists; refusing to overwrite it. Move it away or use a new path.")


def run_logged(cmd, log_path, env=None, echo=True):
    """Run `cmd` from the repository root in its own process group. Every output line goes to
    `log_path` prefixed with a UTC timestamp (and to stdout if `echo`). Returns (exit code,
    wall seconds). A failed run keeps its log."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} $ {' '.join(map(str, cmd))}\n")
        proc = subprocess.Popen([str(part) for part in cmd], cwd=REPO, env=env or ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1, start_new_session=True)
        try:
            for line in proc.stdout:
                stamped = f"{now_iso()} {line.rstrip()}"
                handle.write(stamped + "\n")
                handle.flush()
                if echo:
                    print(stamped, flush=True)
            code = proc.wait()
        except KeyboardInterrupt:
            kill_group(proc)
            raise
        wall = round(time.perf_counter() - started, 2)
        handle.write(f"{now_iso()} exit code {code}, {wall} s\n")
    return code, wall


class SmiLogger:
    """Runs harness/nvidia_smi_logger.py in the background while the block runs. After the
    block: `peak_mib` is the highest memory.used of any sample, `baseline_mib` the first sample
    (memory already in use before the work started)."""

    def __init__(self, path, interval_ms=SMI_INTERVAL_MS):
        self.path, self.interval_ms = Path(path), interval_ms
        self.peak_mib = self.baseline_mib = self.samples = None

    def __enter__(self):
        must_not_exist(self.path)
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO / "harness" / "nvidia_smi_logger.py"), "--out", str(self.path),
             "--interval-ms", str(self.interval_ms)], env=ENV, start_new_session=True)
        time.sleep(1.0)  # let the first samples land before the work starts
        return self

    def __exit__(self, *exc):
        time.sleep(0.5)
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_group(self.proc)
        self.peak_mib, self.baseline_mib, self.samples = smi_peak(self.path)
        return False

    def summary(self):
        return {"nvidia_smi_log": rel(self.path), "nvidia_smi_peak_mib": self.peak_mib,
                "nvidia_smi_baseline_mib": self.baseline_mib, "nvidia_smi_samples": self.samples}


def smi_peak(path):
    """(peak MiB, first-sample MiB, sample count) of a nvidia_smi_logger.py file, summed over
    the GPUs of each sample."""
    totals = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 3 or not parts[2].isdigit():
                    continue
                totals.setdefault(parts[0], 0)
                totals[parts[0]] += int(parts[2])
    except OSError:
        return None, None, 0
    if not totals:
        return None, None, 0
    values = list(totals.values())
    return max(values), values[0], len(values)


def torch_peaks(torch):
    """Peak memory the PyTorch allocator saw in this process, MiB."""
    if not torch.cuda.is_available():
        return {"torch_max_allocated_mib": None, "torch_max_reserved_mib": None}
    return {"torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}


def kill_group(proc):
    """Kill the process and everything it started (a run must not leave work on the GPU)."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def installed_packages():
    """`pip freeze` lines; the harness compares them before and after the runs."""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                         check=False).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def library_versions():
    from importlib import metadata

    versions = {"python": sys.version.split()[0]}
    for name in VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout
    versions["gpus"] = [line.strip() for line in gpus.splitlines() if line.strip()]
    return versions


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def dir_sha256(path):
    """sha256 of every file in a folder (relative path -> sha256), e.g. an artifact."""
    root = Path(path)
    if not root.is_dir():
        return None
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def dir_size_mib(path):
    root = Path(path)
    if not root.is_dir():
        return None
    return round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 2**20, 1)


def git_info():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def data_sha256():
    return {p.name: file_sha256(p) for p in sorted(DATA.glob("*.jsonl"))}


def run_record(seed=None):
    """What every run writes (TASK.md): git commit, library versions, seed, sha256 of the data."""
    return {"time_utc": now_iso(), "git": git_info(), "versions": library_versions(),
            "seed": seed, "data_sha256": data_sha256()}


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def user_turn(row):
    """The user message Soup builds from an alpaca row (soup_cli/data/formats.py,
    _convert_alpaca, Soup 0.77.0)."""
    extra = row.get("input") or ""
    return f"{row['instruction']}\n{extra}".strip() if extra else row["instruction"]


def bnb_nf4_config(torch):
    """The 4-bit config Soup 0.77.0 trains with on a T4 (soup_cli/utils/quant_menu.py):
    NF4, double quantization on, fp16 compute (a T4 has no bf16 units)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
