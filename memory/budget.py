"""Estimate the VRAM before training.

Computes, from the model config and task/task.yaml, the expected memory: weights (4-bit +
fp16 parts), activations, logits, optimizer state, adapter and its gradients; plus the
peaks of merge and quantize.

Reads: task/task.yaml, the model config (config.json of Qwen2.5-1.5B-Instruct).
Writes: logs/<run_id>/budget.json and a table for README / REPORT.
Task part: Part 1 (memory budget).
"""

# TODO: Weights: NF4 bytes, absmax, fp16/fp32 non-quantized parts (embedding, norms).
# TODO: Activations per layer at batch_size x max_length.
# TODO: Logits [B, T, vocab] and their copies.
# TODO: LoRA adapter, its gradients and AdamW state.
# TODO: CUDA context and allocator overhead.
# TODO: Peak of merge (fp16 model) and of quantize.
# TODO: Record in the output JSON: git commit sha, library versions, seed, sha256 of the data files.
