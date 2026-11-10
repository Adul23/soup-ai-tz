"""Blind manual evaluation.

Generates answers of the three models to the same prompts, shuffles them without labels,
and keeps the key separately so that the scores can be unblinded afterwards.

Reads: data/manual_eval.jsonl (40 rows), artifacts/merged/, artifacts/merged_4bit/, the base model.
Writes: logs/<run_id>/manual_eval_sheet.(md|csv) without labels, logs/<run_id>/manual_eval_key.json.
Task part: Part 2 (proof of training).
"""

# TODO: Pick prompts, fixed seed, greedy decoding.
# TODO: Shuffle the answers, write the sheet and the key.
# TODO: Unblind the scores after the manual rating.
# TODO: Every run writes run.json: git commit sha, library versions, seed, sha256 of the data files.
# TODO: Numbers for the report come only from the JSON this script writes.
# TODO: Stop (assert) when a control fails; do not tune: describe the discrepancy instead.
