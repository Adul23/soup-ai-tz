"""Silent-failure checks of the tokenizer and the chat template.

Checks that the tokenizer and the chat template did not break silently: things that do not raise an error but spoil
the result.

Reads: the tokenizer of the base model and of artifacts/merged/.
Writes: logs/<run_id>/checks_tokenizer_template.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Same chat template in training, merge and inference.
# TODO: EOS / pad tokens; labels masked on the prompt.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
