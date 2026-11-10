"""Silent-failure checks of the training.

Checks that the training did not break silently: things that do not raise an error but spoil
the result.

Reads: logs/<run_id>/train.log, artifacts/adapter/.
Writes: logs/<run_id>/checks_training.json (pass / fail with evidence).
Task part: Part 3 (silent failures), one link of the chain data -> tokenizer/template -> training ->
merge -> quantization -> inference.
"""

# TODO: Loss goes down; no NaN / inf.
# TODO: Adapter weights are not zero (lora_B changed from its zero init).
# TODO: Target modules as expected.
# TODO: Inject this silent failure on purpose and show that the check catches it.
# TODO: A failed check stops the pipeline (assert).
