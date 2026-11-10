"""Kernels for the separate memory experiment (NOT the main chain).

What it does: apply() installs, in the order of justdoner/solution/__init__.py, the three patches
ported from there: the fused output head + loss (head.py, with the tied embedding stored in fp16),
the starter kit's example SwiGLU (swiglu.py, Apache-2.0, not ours) and the fp16 MLP (mlp_fp16.py).
All are class-level patches that fall back to the original code outside the situation they were
written for, so stats() counts how often each path really ran: a run where the fused head or the
fp16 MLP never engaged is a failed run (the patch silently fell back).
Not ported: offload.py, head_stream.py, packed.py, unpad.py, nf4.py, rmsnorm.py, warmup.py,
startup.py, mlp_recompute.py.
Reads / writes: nothing.
Task part: separate memory experiment (part 1 context). The artifacts for verify and the verdict
come from plain Soup (harness/smoke.py), never from this package.
"""

from . import head, mlp_fp16, swiglu
from .head import patch_fused_head
from .mlp_fp16 import patch_qwen2_mlp_fp16
from .swiglu import patch_qwen2_mlp


def apply():
    patch_fused_head()  # Triton fused head + loss: no full logits tensor
    patch_qwen2_mlp()  # starter kit's example SwiGLU (not ours)
    patch_qwen2_mlp_fp16()  # MLP activations in fp16 instead of fp32 copies, same values


def stats():
    """How often each patched path ran in this process."""
    return {**head.STATS, **mlp_fp16.STATS, **swiglu.STATS}


def engaged(counters):
    """(ok, reasons): the run counts only if the fused head and the fp16 MLP both ran."""
    reasons = [name for name in ("fused_head_steps", "mlp_fp16_calls") if not counters.get(name)]
    return not reasons, reasons
