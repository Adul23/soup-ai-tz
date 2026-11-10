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
| 2026-11-10 | notebooks/kaggle_full_run.ipynb: one notebook for the whole run (setup from kaggle_run.ipynb, tests incl. kernels, smoke, task A, verify 20/300 rows, kernels experiment, summary from JSON, archive) | pending | |
| 2026-11-10 | Fix for merge failing on Kaggle (peft 0.20 rejects torchao 0.10.0): pin torchao==0.18.0, early peft+torchao check in the notebook, smoke.py --resume | pending | |
| 2026-11-10 | checks/silent_failures.py: Soup's data doctor / dry-run / ship on the real inputs; injected pad=eos, max_length 64, dirty data, renamed adapter (keys and config), no-system template; "caught" = differs from the same command on the clean 50-row copy; ship gets a 20-row --task-eval with scoring "contains" | pending | |
| 2026-11-10 | checks/language.py (3-gram loops, "X және X", Kyrgyz forms, ".Буква") and verdict.py (rules as given by the author, verify.json numbers only) | pending | |
| 2026-11-10 | Notebook cells: prepare data + memory/budget.py before smoke, silent failures, language, budget --compare after kernels, verdict | pending | |
| 2026-11-10 | verdict.py survives_4bit corrected. The first wording, "CI(merged_4bit − merged) contains 0 or > −0.02", was proposed by another AI tool, not Claude Code; it is true for almost any degradation, because merged_4bit − merged is normally positive. Claude Code found the problem and the author confirmed it. New rule: CI(merged_4bit − merged) high < +0.02 AND CI(merged_4bit − base_4bit) entirely < 0. usable keeps loop_share from verify.json (4-gram ≥3) as the only source | used (author confirmed) | |
| 2026-11-10 | task/prepare_data.py writes data/data_report.json: would-drop/change counters by reason (not applied; zero dropped/changed), token-length percentiles via the chat template, fertility, round-trip, sha256, revision. Rewritten from the author's list: the author's earlier version was not found in the repo history | pending | |
| 2026-11-10 | Notebook: silent_failures exit 1/3 are findings and no longer stop the run (only 4 does); archive adds data_report.json, pip freeze, AI_USAGE.md, CLAUDE.md; FINAL asserts a clean git tree | pending | |
