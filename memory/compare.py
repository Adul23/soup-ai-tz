"""Compare the estimate with the real peak from the logs.

Reads the budget and the nvidia-smi logs of each stage and prints estimate vs measured
with the difference and its explanation.

Reads: logs/<run_id>/budget.json, logs/<run_id>/nvidia_smi_*.csv.
Writes: logs/<run_id>/budget_vs_actual.json and a table for README / REPORT.
Task part: Part 1 (memory budget).
"""

# TODO: Parse the nvidia-smi logs, take the peak per stage, subtract the baseline.
# TODO: Join with the budget per stage, print the table.
# TODO: Record in the output JSON: git commit sha, library versions, seed, sha256 of the data files.
# TODO: Compare with both nvidia-smi and torch max_memory_allocated / max_memory_reserved.
