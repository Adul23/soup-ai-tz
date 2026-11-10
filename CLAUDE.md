# Notes for Claude

The task and its mandatory rules are in TASK.md. Follow them on every change, in particular:

- Plain Soup 0.77.0. Never bring over code from the hackathon repo justdoner/ solution/ (offload, kernels, patches).
- Every number in README.md / REPORT.md comes from a JSON file the code wrote; never type numbers by hand.
- Raw timestamped logs in logs/<run_id>/, failed runs included; never delete or overwrite a log.
- Failed tests or checks stop the script / notebook with assert.
- Every run records git commit sha, library versions, seed, sha256 of the data files.
- If a result looks strange, stop and describe the discrepancy; do not tune to make it look right.
- T4: fp16 only, no bf16. Peak memory from nvidia-smi (background, to a file) and torch max_memory_allocated / max_memory_reserved.
- Whenever the AI suggests something that is used, add a row to AI_USAGE.md.

Work order: (1) end-to-end chain, simple; (2) memory budget before training; (3) verify/; (4) checks/; (5) verdict and report.
