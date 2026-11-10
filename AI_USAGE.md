# AI usage

<!-- TODO: one entry per use. -->

| Date | What the AI suggested | Accepted? | What I checked or redid myself |
|---|---|---|---|
| 2026-11-10 | Task 1 code (harness/*.py, notebook) written by Claude; not yet run on a GPU | pending | |
| 2026-11-10 | Data splits: train rows 0–1499, heldout 40000–40299, manual_eval 41000–41039 of the pinned revision | pending | |
| 2026-11-10 | torch peaks of training read by running Soup's CLI in-process (harness/soup_train_inproc.py), with a read-only TrainerCallback; nothing patched | pending | |
| 2026-11-10 | Own merge/quantize with transformers + peft instead of `soup merge` (no NF4-dequant option there) and `soup quantize` (only prints a GGUF command) | pending | |
| 2026-11-10 | 4-bit config = the one Soup trains with: NF4, double quant, fp16 compute | pending | |
