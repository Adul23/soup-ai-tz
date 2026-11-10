"""Silent-failure checks of the data.

Checks that the data did not break silently: things that do not raise an error but spoil
the result.

Reads: data/train.jsonl (1500), data/heldout.jsonl (300), data/manual_eval.jsonl (40).
Writes: logs/<run_id>/checks_data.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Empty / truncated rows, duplicates, train/eval overlap.
# TODO: Share of rows cut by max_length=512.
# TODO: Row counts 1500 / 300 / 40; no overlap between the three files.
# TODO: sha256 of every data file matches the one recorded by the run.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
