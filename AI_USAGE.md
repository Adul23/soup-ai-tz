# AI usage

<!-- TODO: one entry per use. -->

| Date | What the AI suggested | Accepted? | What I checked or redid myself |
|---|---|---|---|
| 2026-11-10 | Task 1 code (harness/*.py, notebook) written by Claude; not yet run on a GPU | pending | |
| 2026-11-10 | Data splits: train rows 0–1499, heldout 40000–40299, manual_eval 41000–41039 of the pinned revision | pending | |
| 2026-11-10 | torch peaks of training read by running Soup's CLI in-process (harness/soup_train_inproc.py), with a read-only TrainerCallback; nothing patched | pending | |
| 2026-11-10 | Own merge/quantize with transformers + peft instead of `soup merge` (no NF4-dequant option there) and `soup quantize` (only prints a GGUF command) | pending | |
| 2026-11-10 | 4-bit config = the one Soup trains with: NF4, double quant, fp16 compute | pending | |
| 2026-11-10 | Task A: real first batch captured by wrapping the trainer instance's training_step inside an in-process `soup train`; code references to Soup/TRL label building | pending | |
| 2026-11-10 | Verify: own answer-only masking cross-checked against Soup's build_assistant_only_labels; greedy generation with the model's generation_config replaced (Qwen default repetition_penalty=1.05) | pending | |
| 2026-11-10 | Verify on two GPUs (run_parallel.py, merge_verify.py), paired bootstrap; fixed smi_peak grouping (was never summing over GPUs) | pending | |
| 2026-11-10 | Memory experiment: kernels/ ported from justdoner (head.py, mlp_fp16.py, swiglu.py + tests), warmup removed, head_stream unreachable behind an assert, engagement counters | pending | |
