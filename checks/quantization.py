"""Silent-failure checks of the quantization.

Checks that the quantization did not break silently: things that do not raise an error but spoil
the result.

Reads: artifacts/merged/, artifacts/merged_4bit/.
Writes: logs/<run_id>/checks_quantization.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Every linear layer is really 4-bit; which layers stayed in fp16.
# TODO: Logit / loss drift vs merged.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
