"""Silent-failure checks of the merge.

Checks that the merge did not break silently: things that do not raise an error but spoil
the result.

Reads: artifacts/adapter/, artifacts/merged/.
Writes: logs/<run_id>/checks_merge.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Merged weights differ from base exactly where the adapter applies.
# TODO: Adapter output vs merged output on the same input.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
