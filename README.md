# Soup AI Engineer take-home: LoRA → merge → 4-bit on Qwen2.5-1.5B-Instruct

Fine-tune Qwen/Qwen2.5-1.5B-Instruct with Soup (soup-cli 0.77.0) on Kazakh instructions, merge the
LoRA adapter, quantize to 4-bit NF4, and decide whether the result can ship. Hardware: one NVIDIA T4
(Kaggle). Final run: commit `7a571d5`.

The full write-up is in the report (PDF sent with the submission). This README is the map of the repo.

## Verdict

**DON'T SHIP.** The fine-tune is real and survives 4-bit quantization, but the shipped 4-bit model
still loops in 43% of answers and ends cleanly in only 58%.

| Rule (fixed before the run) | Result | Pass |
| --- | --- | --- |
| trained: 95% CI of loss(merged) − loss(base) below 0 | −0.204 [−0.216, −0.192] | yes |
| survives_4bit: merged_4bit − base_4bit below 0, and merged_4bit − merged upper bound < +0.02 | −0.379 [−0.394, −0.365]; +0.005 [−0.001, +0.010] | yes |
| usable: loop share ≤ 10% and EOS share ≥ 90% (merged_4bit, greedy) | loop 43%, EOS 58% | **no** |

What must change first: stop cutting answers at 512 tokens, train LoRA on all seven linear modules
(Soup's `auto` picked only `q_proj` and `v_proj`), clean repeated words in the data, and gate on loop
and EOS share with the decoding settings that will actually ship.

## Memory budget

`memory/budget.py` estimates VRAM from the model and training config only; `--compare` checks it
against measured peaks.

| MiB | Estimate | Measured (torch allocated) |
| --- | --- | --- |
| Training, plain Soup | 14,697 | 13,409 |
| Training, with ported kernels | 6,968 | 5,191 |
| Merge | 2,944 | 2,979 |
| Quantize | 1,090 | 1,125 |

Logits (152k-token vocabulary) and fp32 copies in the MLP are about half of the training peak. The
kernels experiment (fused LM head + loss, fp16 MLP; separate from the shipped chain) cuts the peak
from 14,461 to 5,767 MiB (nvidia-smi) with the same held-out loss.

## Proof of training

`verify/verify_artifacts.py` scores answer-only loss on 300 held-out rows for five models: base,
zero adapter (B = 0, must equal base), merged, base_4bit and merged_4bit. It also diffs NF4 codes
and puts a paired bootstrap on every difference.

| Model | Loss | EOS share | Loop share |
| --- | --- | --- | --- |
| base | 2.107 | 32% | 61% |
| merged | 1.903 | 54% | 49% |
| base_4bit | 2.287 | 18% | 78% |
| merged_4bit | 1.908 | 58% | 43% |

Kyrgyz word forms fall from 5.3 per 100 words (base) to 0.00 (merged_4bit), and only the trained
modules' NF4 codes change (5% of `q_proj` / `v_proj`, 0 elsewhere).

## Silent failures

`checks/` holds the checks; the main findings:

- **81 of 1,500 training answers (5.4%) are longer than max_length 512.** Soup cuts them and they
  lose `<|im_end|>`. `soup data doctor` only says "MINOR", and its EOS check skips the cut rows.
- `target_modules: auto` trains 2 of 7 linear modules without saying so.
- An adapter with renamed tensor keys loads nothing; merge still reports "merged 56 LoRA modules".
- `soup ship` scores base + adapter in bf16, not the merged 4-bit file that ships.
- Dirty data (repeated words) and serving without the system prompt are not caught by Soup.
- nvidia-smi sampled every 200 ms missed the training peak.

Checked and fine: the prompt is masked out of the loss; pad ≠ EOS; Soup fails loudly when
max_length removes the whole answer.

## What surprised me

4-bit quantization kept the whole gain from a q/v-only adapter. A 5-row smoke test with default
decoding raised a false alarm that the 300-row greedy run overturned. Loss improved by 10%, yet half
the answers still loop, so a loss-only gate would pass a model users should not get.

## AI tools used

Claude (chat) and Claude Code helped with planning, scripts and the report draft; every rule and
threshold was fixed before the runs, and the claims were checked by runs. Details, including where
the AI was wrong, are in `AI_USAGE.md`.

## How to run

Kaggle, accelerator **GPU T4 x2** (training uses GPU 0 only), Internet on. Add a `GITHUB_TOKEN`
secret if the repo is private.

1. Open `notebooks/kaggle_full_run.ipynb`.
2. Set `COMMIT` to the commit to run.
3. Run All (about 2 h). Every failed step stops the notebook.

Pinned packages are in `requirements.txt` (soup-cli 0.77.0, transformers 5.16.1, peft 0.20.0,
trl 0.29.1, bitsandbytes 0.50.2, torch 2.11.0).

## Repository layout

| Path | What it holds |
| --- | --- |
| `task/` | `task.yaml` (Soup config), `prepare_data.py` (train / held-out / manual-eval split, `data_report.json`) |
| `harness/` | training, merge, quantize, smoke generation, nvidia-smi logger, kernels experiment |
| `memory/` | `budget.py`: VRAM estimate and comparison with measured peaks |
| `verify/` | `verify_artifacts.py` and controls: part 2, proof of training |
| `checks/` | prompt masking, injected silent failures, language statistics: part 3 |
| `kernels/` | fused LM head + loss and fp16 MLP ported from my Soup kernel-hackathon work, with tests |
| `verdict.py` | SHIP / DON'T SHIP from the JSON results only |
| `notebooks/` | the run notebook and the executed notebooks (console output = raw timestamped logs) |
| `results_*.tar.gz` | logs, nvidia-smi CSVs, every JSON result and the two adapters of the final run |

Failed runs are kept: one stopped on a full disk, and in one a notebook cell deleted the results
folder, so the run was repeated.