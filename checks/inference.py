"""Silent-failure checks of the inference.

Checks that the inference did not break silently: things that do not raise an error but spoil
the result.

Reads: artifacts/merged_4bit/.
Writes: logs/<run_id>/checks_inference.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Generation stops at EOS; no repetition loops.
# TODO: Answers are in Kazakh.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
