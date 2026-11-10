"""Check the three artifacts on held-out answers.

Computes loss, perplexity and bits per character on the answer tokens of held-out rows for
base, merged and 4-bit, plus two controls: a zero adapter (must equal base) and base in 4 bits
(separates the quantization error from the training effect).

Reads: data/heldout.jsonl (300 rows), artifacts/adapter/, artifacts/merged/, artifacts/merged_4bit/.
Writes: logs/<run_id>/verify.json and a table for README / REPORT.
Task part: Part 2 (proof of training).
"""

# TODO: Mask the prompt: score only the answer tokens, with the same chat template as training.
# TODO: Metrics: loss, perplexity, bits per character.
# TODO: Control: zero adapter == base.
# TODO: Control: base in 4 bits.
# TODO: Adapter vs merged: must match within fp16 tolerance.
# TODO: Every run writes run.json: git commit sha, library versions, seed, sha256 of the data files.
# TODO: Numbers for the report come only from the JSON this script writes.
# TODO: Stop (assert) when a control fails; do not tune: describe the discrepancy instead.
