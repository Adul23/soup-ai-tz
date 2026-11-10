"""VRAM budget for the Soup AI Engineer task, from the model config and the training config only.

    python memory/budget.py                       # -> runs/memory_budget.json + table
    python memory/budget.py --seq 512 --compare   # also compares with measured peaks in runs/

Reads (optional): data/data_report.json (longest row in tokens -> padded batch length),
runs/smoke.json and runs/kernels_experiment.json (measured peaks, --compare only).
Writes: runs/memory_budget.json.
Task part: 1 (memory budget). Every number is a formula over the constants below; the
assumptions are listed next to each item, so a gap with the measurement points at an item.

Setup it models (Soup 0.77.0 defaults seen in runs/steps/train_*.json): Qwen2.5-1.5B-Instruct,
bitsandbytes NF4 with double quantization, fp16 autocast (T4: no bf16), LoRA r=16 on q_proj and
v_proj, AdamW, batch 4, max_length 512, no gradient checkpointing, no packing (batches padded to
their longest row), tied embedding / lm_head upcast to fp32 by prepare_model_for_kbit_training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MiB = 2**20

# Qwen/Qwen2.5-1.5B-Instruct config.json
H, I, L, V = 1536, 8960, 28, 151936  # hidden, intermediate, layers, vocab
N_HEADS, N_KV, HEAD_DIM = 12, 2, 128
KV = N_KV * HEAD_DIM  # 256
TOTAL_PARAMS = 1_543_714_304  # incl. the tied embedding once

# training config (task/task.yaml + what Soup's trainer reported)
BATCH, MAX_LEN, LORA_R = 4, 512, 16
LORA_TARGETS = {"q_proj": (H, H), "v_proj": (H, KV)}  # (in, out)
NF4_BLOCK, NF4_BLOCK2 = 64, 256  # bitsandbytes blocksize and double-quant blocksize


def linear_shapes():
    return {
        "q_proj": (H, H),
        "k_proj": (H, KV),
        "v_proj": (H, KV),
        "o_proj": (H, H),
        "gate_proj": (H, I),
        "up_proj": (H, I),
        "down_proj": (I, H),
    }


def static_items():
    lin = L * sum(a * b for a, b in linear_shapes().values())
    lora = L * sum(LORA_R * (a + b) for a, b in LORA_TARGETS.values())
    norms = (2 * L + 1) * H
    return (
        lin,
        lora,
        {
            "nf4_weights": (lin * 0.5, "196 Linear4bit, 4 bits per weight"),
            "nf4_absmax": (
                lin / NF4_BLOCK + lin / NF4_BLOCK / NF4_BLOCK2 * 4,
                "double quant: 1 byte per 64 weights + fp32 per 256 blocks",
            ),
            "embedding_fp32": (
                V * H * 4,
                "tied embedding/lm_head, upcast to fp32 by "
                "prepare_model_for_kbit_training",
            ),
            "lm_head_fp16_cast": (
                V * H * 2,
                "autocast's fp16 copy of the head weight, kept for "
                "backward (grad_hidden = grad_logits @ W16)",
            ),
            "norms_fp32": (norms * 4, "57 RMSNorm weights"),
            "lora_params_fp32": (lora * 4, f"{lora:,} trainable params"),
            "lora_grads_fp32": (lora * 4, "one gradient per trainable param"),
            "adamw_states_fp32": (lora * 8, "exp_avg + exp_avg_sq"),
        },
    )


def activation_bytes_per_token_layer(kernels=False):
    """Tensors autograd keeps per token and decoder layer (no checkpointing).

    Attention: norm input fp32 4H; q/k/v each cast the fp32 input to fp16 for the NF4 matmul and
    save it, 3 x 2H; LoRA on q, v: dropout mask H + fp16 input of lora_A 2H, x2; SDPA saves q, k,
    v (k, v repeated to 12 heads) and its output in fp16, 4 x 2H; o_proj input fp16 2H. -> 26H.
    MLP (base): norm input 4H; gate/up input casts 2 x 2H; bitsandbytes returns gate and up in
    fp32 (out.to(input dtype)), and silu(gate) * up keeps gate, silu(gate) and up: 3 x 4I;
    down_proj's fp16 input 2I. -> 8H + 14I.
    MLP (kernels/mlp_fp16 + swiglu): gate and up stay fp16 and the fused SwiGLU keeps only them,
    2 x 2I, plus down's fp16 input 2I. -> 8H + 6I.
    """
    attn = 26 * H
    mlp = 8 * H + (6 * I if kernels else 14 * I)
    return attn + mlp


def logits_bytes(tokens, kernels=False):
    """Base: fp16 logits (kept by outputs.logits for TRL's metrics) 2V, their fp32 upcast in
    ForCausalLMLoss 4V and log_softmax's saved fp32 output 4V, per token -> 10V at the peak of the
    loss. Kernels (head.py): one reused fp16 buffer of 128 rows plus the fp32 hidden gradient."""
    if kernels:
        return 128 * V * 2 + tokens * H * 4
    return tokens * V * 10


def budget(seq, kernels=False):
    tokens = BATCH * seq
    lin, lora, items = static_items()
    if kernels:  # compact_tied_embedding: one fp16 copy instead of fp32 + autocast fp16
        items["embedding_fp32"] = (
            V * H * 2,
            "kernels: tied weight stored once in fp16",
        )
        items["lm_head_fp16_cast"] = (0, "kernels: head uses that fp16 copy directly")
    items["activations"] = (
        tokens * L * activation_bytes_per_token_layer(kernels),
        f"{tokens} padded tokens x {L} layers x "
        f"{activation_bytes_per_token_layer(kernels)} B",
    )
    items["logits_and_loss"] = (
        logits_bytes(tokens, kernels),
        "10 bytes/token/vocab entry" if not kernels else "chunked head",
    )
    total = sum(b for b, _ in items.values())
    return {
        "seq": seq,
        "tokens": tokens,
        "kernels": kernels,
        "items_mib": {k: round(b / MiB, 1) for k, (b, _) in items.items()},
        "assumptions": {k: why for k, (_, why) in items.items()},
        "torch_allocated_peak_mib": round(total / MiB, 1),
        "nvidia_smi_peak_mib_est": round(
            total / MiB + 350, 1
        ),  # + CUDA context (~300-400 MiB)
    }


def export_budget():
    fp16_model = TOTAL_PARAMS * 2
    lin, _, items = static_items()
    nf4 = items["nf4_weights"][0] + items["nf4_absmax"][0]
    embed16 = V * H * 2
    other16 = (TOTAL_PARAMS - lin - V * H) * 2
    return {
        "merge_peak_mib": round(fp16_model / MiB, 1),
        "merge_note": "whole fp16 model on the GPU; LoRA delta per module is transient",
        "quantize_peak_mib": round((nf4 + embed16 + other16) / MiB, 1),
        "quantize_note": "NF4 linears + fp16 embedding/head (bnb leaves lm_head unquantized) + norms",
    }


def compare(report, runs):
    def load(name):
        p = runs / name
        return json.loads(p.read_text()) if p.exists() else None

    out = {}
    smoke = load("smoke.json")
    if smoke:
        st = smoke["steps"]
        out["train_plain"] = {
            "measured_torch_alloc": st["train"]["torch_max_allocated_mib"],
            "measured_smi": st["train"]["nvidia_smi_peak_mib"],
            "measured_torch_reserved": st["train"]["torch_max_reserved_mib"],
            "budget": report["train_plain"]["torch_allocated_peak_mib"],
        }
        out["merge"] = {
            "measured_torch_alloc": st["merge"]["torch_max_allocated_mib"],
            "budget": report["export"]["merge_peak_mib"],
        }
        out["quantize"] = {
            "measured_torch_alloc": st["quantize"]["torch_max_allocated_mib"],
            "budget": report["export"]["quantize_peak_mib"],
        }
    kern = load("kernels_experiment.json")
    if kern and kern.get("peaks"):
        p = kern["peaks"]["kernels"]
        out["train_kernels"] = {
            "measured_torch_alloc": p["torch_max_allocated_mib"],
            "measured_smi": p["nvidia_smi_peak_mib"],
            "budget": report["train_kernels"]["torch_allocated_peak_mib"],
        }
    for v in out.values():
        if v.get("measured_torch_alloc"):
            v["gap_mib"] = round(v["budget"] - v["measured_torch_alloc"], 1)
            v["gap_pct"] = round(100 * v["gap_mib"] / v["measured_torch_alloc"], 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seq",
        type=int,
        default=None,
        help="padded batch length; default: longest row from data_report.json, else 512",
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    seq, seq_source = args.seq, "--seq"
    if seq is None:
        rep = root / "data/data_report.json"
        try:
            seq = int(json.loads(rep.read_text())["tokens_full"]["train"]["p100"])
            seq_source = "data_report.json p100 (the peak batch holds the longest row)"
        except Exception:  # noqa: BLE001 - no report: worst case
            seq, seq_source = MAX_LEN, "max_length (no data_report.json)"
    seq = min(seq, MAX_LEN)

    report = {
        "seq_source": seq_source,
        "train_plain": budget(seq),
        "train_kernels": budget(seq, kernels=True),
        "export": export_budget(),
    }
    if args.compare:
        report["compare"] = compare(report, root / "runs")
    out = root / "runs/memory_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"padded length {seq} ({seq_source}), tokens per batch {BATCH * seq}")
    print(f"{'item':22s}{'plain MiB':>12s}{'kernels MiB':>13s}")
    for k in report["train_plain"]["items_mib"]:
        print(
            f"{k:22s}{report['train_plain']['items_mib'][k]:>12}"
            f"{report['train_kernels']['items_mib'][k]:>13}"
        )
    print(
        f"{'TOTAL (torch alloc)':22s}{report['train_plain']['torch_allocated_peak_mib']:>12}"
        f"{report['train_kernels']['torch_allocated_peak_mib']:>13}"
    )
    print(
        f"merge peak {report['export']['merge_peak_mib']} MiB, "
        f"quantize peak {report['export']['quantize_peak_mib']} MiB"
    )
    for k, v in report.get("compare", {}).items():
        print(f"compare {k:14s} {v}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
