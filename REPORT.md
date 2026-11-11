# Soup AI Engineer take-home — Report

Adil · 11 Nov 2026 · Qwen2.5-1.5B-Instruct, Soup 0.77.0, LoRA r=16, NF4, 1,500 Kazakh rows (AmanMussa/kazakh-instruction-v2), Kaggle T4, 300 held-out rows, commit `7a571d5`

## Verdict: DON'T SHIP

The fine-tune is real and survives 4-bit quantization, but the shipped 4-bit model still loops in 43% of answers and ends cleanly in only 58%.

| Rule (fixed before the run) | Evidence | Pass |
| --- | --- | --- |
| trained: 95% CI of loss(merged) − loss(base) below 0 | −0.204 [−0.216, −0.192] | yes |
| survives_4bit: merged_4bit − base_4bit below 0; upper CI of merged_4bit − merged < +0.02 | −0.379 [−0.394, −0.365]; +0.005 [−0.001, +0.010] | yes |
| usable: loop share ≤ 10% and EOS share ≥ 90% (merged_4bit, greedy) | loop 43%, EOS 58% | **no** |

What must change first: (1) stop cutting answers: drop or shorten the 81 rows over 512 tokens so every answer keeps its stop token; (2) train LoRA on all seven linear modules — Soup's `target_modules: auto` chose only `q_proj`, `v_proj` (0.14% of the model) without saying so; (3) clean repeated words ("X және X" in 4.8% of answers) and missing spaces after full stops; (4) gate on loop and EOS share, measured with the decoding settings that will ship.

## 1. Memory budget

Training peaks at 13.4 GiB allocated on a 15 GiB T4; logits and fp32 MLP copies are about half of it. `memory/budget.py` builds the estimate from the model and training config only (batch 4, padded length 512, no gradient checkpointing).

| MiB | Estimate | Measured (torch alloc) | Gap |
| --- | --- | --- | --- |
| Training, plain Soup | 14,697 | 13,409 | +10% |
| Training, ported kernels | 6,968 | 5,191 | +34% |
| Saving from kernels | 7,729 | 8,219 | −6% |
| Merge (fp16 model) | 2,944 | 2,979 | −1% |
| Quantize (NF4 + fp16 embedding) | 1,090 | 1,125 | −3% |

Estimate by item (plain): activations 9,716 (28 layers × 2,048 tokens), logits and loss 2,968 (10 bytes per token per vocab entry), fp32 embedding + autocast fp16 head copy 1,335, NF4 weights 645, LoRA + AdamW 33. The gap sits in the shared activation term (equal in both training rows, while the kernel saving and export are within 6%): I over-count tensors saved per op in attention. Padding is not the cause: 81 rows exceed 512 tokens, so the peak batch is full length. The allocator holds 1.3 GiB more than allocated (reserved 14,732 MiB), and nvidia-smi sampled every 200 ms reported 14,461 MiB, below torch's own reserved peak, so it missed the spike. The budget was written after the first run, not before it. Kernels experiment (fused LM head + loss, fp16 MLP, ported from my Soup kernel-hackathon code; not in the shipped chain): peak 14,461 → 5,767 MiB, training 467 → 380 s, same held-out loss (1.90313 vs 1.90319), patched path on 375 of 375 steps.

## 2. Proof that the model trained

The merged model beats the base by 0.204 nats per answer token on 300 unseen rows, and 4-bit keeps all of it. Training loss fell 2.247 → 1.706, but that proves little: it is measured on rows being fitted, can fall on trivially predictable tokens, and says nothing about whether the adapter reached the merged or quantized file.

The check (`verify/verify_artifacts.py`) scores answer-only loss on 300 held-out rows (exact prompt overlap with train: 0), with labels cross-checked against those Soup's trainer builds (0 of 300 differ). Controls: a **zero adapter** (B = 0, merged by the same script) must equal the base exactly, which proves the merge path changes nothing by itself; a **base_4bit** model separates quantization damage from training gain. It also diffs NF4 codes and puts a paired bootstrap (1,000 resamples) on every difference.

| Artifact | Loss | PPL | EOS | Loop | Length | Kyrgyz forms / 100 words |
| --- | --- | --- | --- | --- | --- | --- |
| base | 2.107 | 8.23 | 32% | 61% | 219 | 5.29 |
| zero adapter | 2.107 (identical) | 8.23 | – | – | – | – |
| merged | 1.903 | 6.71 | 54% | 49% | 147 | 0.01 |
| base_4bit | 2.287 | 9.84 | 18% | 78% | 231 | 0.75 |
| merged_4bit | 1.908 | 6.74 | 58% | 43% | 133 | 0.00 |

Greedy, ≤256 new tokens; a loop is a 4-word n-gram repeated 3+ times. NF4 codes changed only where LoRA was (5.1% of `q_proj`, 5.9% of `v_proj`, 0 elsewhere). Behaviour changed too: Kyrgyz forms (0.01 per 100 words in the data) vanish, and the model copies a data quirk, missing spaces after "." (36% of references; 0% → 7% of answers). Not detected: whether answers are correct Kazakh, behaviour under sampling, and near-duplicate (not exact) leakage.

## 3. Silent failures

The worst silent problem was real: 5.4% of training answers were cut at 512 tokens and lost their stop token, and Soup only called it "MINOR". Faults were injected by `checks/silent_failures.py`.

| Stage | Silent failure | Found here? | Does Soup catch it? |
| --- | --- | --- | --- |
| Data | Answers over max_length cut, losing `<\|im_end\|>` | **Yes: 81 of 1,500 (5.4%), up to 1,486 tokens** | Doctor "MINOR"; its EOS check skips the cut rows (0 of 1,419). Pre-flight "Data OK" |
| Data | Repeated words, duplicates teach loops | Yes: 4.8% "X және X" | No: same doctor verdict on an injected dirty copy |
| Template | Prompt or system text in the loss | No: 0 of 75 prompt tokens in the real first batch | Not needed |
| Template | pad = EOS masks the stop token (injected) | No: EOS stays in labels | Robust: masks by template, not pad id |
| Template | Serving without the system prompt (injected) | Answers change | No |
| Training | `target_modules: auto` trains 2 of 7 modules | Yes | No |
| Merge | Adapter tensor keys renamed, nothing loads (injected) | Merge still says "merged 56 LoRA modules" | peft warning only; `soup ship` shows tuned = base, no alert |
| Quantize | NF4 rounds the LoRA delta away | No: 5% of q/v codes changed, gain kept | No |
| Ship gate | `soup ship` scores base + adapter in bf16, not the merged 4-bit file | Yes | It is the check; flagged safety −0.15, tool-call −0.075 |
| Measurement | nvidia-smi under-reports the peak | Yes: 14,461 < 14,732 MiB reserved | No |

Loud and fine: max_length 64 (no answer left) and renamed `target_modules` both fail with a clear error; warmup 0.03 is read as a ratio. Also seen: default generation settings in a 5-row smoke test raised a false alarm, and batch 1 vs 16 changes 1–2 of 20 greedy answers.

## What surprised me, and what still concerns me

I expected 4-bit to erase a q/v-only adapter, yet it kept the whole gain, and merged_4bit even loops less than merged (43% vs 49%), which I cannot explain yet. Reading outputs as a Kazakh speaker showed what loss cannot: the base model writes Kyrgyz forms (бул, тил, шаблондун), and training removed them — the clearest sign the adapter changed behaviour. My concern: loss improved by 10%, yet half the answers still loop, so a loss-only gate would pass a model users should not get.

**AI tools.** Claude (chat): planning, data preparation, memory formulas, verdict rules, report draft; its claims were checked by runs — its guess that the prompt was in the loss was wrong (disproved by the masking check), and its first `survives_4bit` rule was wrong (Claude Code caught it). Claude Code: harness, verify, checks and notebook code. Me: dataset choice, reading outputs in Kazakh, rules and thresholds fixed before the runs, every ship decision. Details in `AI_USAGE.md`.

**Reproduce.** Kaggle, GPU T4 x2 (training on GPU 0): `notebooks/kaggle_full_run.ipynb`, set `COMMIT`, Run All (~2 h). Logs, nvidia-smi CSVs and JSON of the final run: `results_run_head_20261110T174018.tar.gz`; executed notebooks keep the console output. Failed runs are kept (one stopped on a full disk; in one a cell deleted the results folder).