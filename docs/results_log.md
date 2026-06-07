# Results Log

## 2026-06-07 Colab A100 GPT-2 Stress Pilot

Environment visible in the saved notebook output:

- GPU: NVIDIA A100-SXM4-80GB
- Model: `gpt2`
- Data profile: `stress`
- Synthetic examples: 320
- HF feature examples: 160
- Residual patch examples: 24
- Layers: `0,6,11`

Surrogate stress split:

| Method | F1 | AUROC | FNR | FPR |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.8889 | 0.5682 | 0.0000 | 0.5909 |
| Rule features + Logistic Regression | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Surrogate PatchDistill | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

GPT-2 one-pass / patching pilot:

| Component | Result |
| --- | ---: |
| `hf-extract` examples | 160 |
| `hf-patch` examples | 24 |
| Proxy MAE | 0.0277 |
| Proxy MSE | 0.00407 |
| HF features-only detector F1 | 1.0000 |
| HF features-only detector AUROC | 1.0000 |
| Distilled patch detector F1 | 1.0000 |
| Distilled patch detector AUROC | 1.0000 |

Interpretation:

- The stress profile is harder for TF-IDF, mainly by increasing false positives.
- Rule features, surrogate PatchDistill, and GPT-2 one-pass features still solve
  the task perfectly, so the dataset remains too separable.
- The next experimental step should introduce harder benign prompts and
  paraphrased/adaptive attacks that reduce lexical and length artifacts before
  claiming any advantage from causal patch distillation.

Archive note:

- The save-results cell created a local Colab commit
  `691426c Add Colab results: gpt2_a100_stress_001_001`.
- Push failed with GitHub HTTP 403, so the full JSONL artifacts were not yet
  available in the GitHub repository at the time of this note.

## 2026-06-08 Local Matched Surrogate Pilot

The `matched` profile places the same suspicious span in both classes. In
positive examples, the span is an active instruction. In negative examples, it is
document/log/training content that should be treated as data. Negative examples
also keep `malicious_span` populated so span-token features alone cannot separate
the classes.

| Method | F1 | AUROC | FNR | FPR |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Rule features + Logistic Regression | 0.7222 | 0.7868 | 0.2041 | 0.4082 |
| Surrogate PatchDistill | 0.7222 | 0.7926 | 0.2041 | 0.4082 |

Interpretation:

- Matched spans successfully break the hand-built rule features.
- TF-IDF remains perfect, likely because source-boundary prefix wording
  (`Priority instruction`, `Log message payload`, etc.) is still separable.
- Next step: run GPT-2/Qwen HF features and residual patching on `matched`, then
  create a stricter matched profile with more varied source-boundary wording.

## 2026-06-07 Colab A100 GPT-2 Matched Pilot

Environment visible in the saved notebook output:

- GPU: NVIDIA A100-SXM4-80GB
- Model: `gpt2`
- Data profile: `matched`
- Synthetic examples: 320
- HF feature examples: 160
- Residual patch examples: 24
- Layers: `0,6,11`

Surrogate matched split:

| Method | F1 | AUROC | FNR | FPR |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Rule features + Logistic Regression | 0.7222 | 0.7868 | 0.2041 | 0.4082 |
| Surrogate PatchDistill | 0.7222 | 0.7926 | 0.2041 | 0.4082 |

GPT-2 one-pass / patching pilot:

| Component | Result |
| --- | ---: |
| `hf-extract` examples | 160 |
| `hf-patch` examples | 24 |
| Proxy MAE | 0.0528 |
| Proxy MSE | 0.01265 |
| HF features-only detector F1 | 1.0000 |
| HF features-only detector AUROC | 1.0000 |
| Distilled patch detector F1 | 1.0000 |
| Distilled patch detector AUROC | 1.0000 |

Interpretation:

- Matched spans make hand-written rule features fail, as intended.
- GPT-2 one-pass features still classify perfectly.
- Since TF-IDF is also perfect, the likely shortcut is no longer just dangerous
  span presence but the source-boundary/prefix wording that distinguishes active
  instructions from quoted/logged/training data.
- The next dataset should randomize or balance boundary wording and include
  active and data-only examples under overlapping prefixes.

## 2026-06-08 Local Contrastive Surrogate Pilot

The `contrastive` profile puts the same suspicious span in both classes and uses
the same boundary vocabulary in opposite order:

- Positive: do not treat the block as data; treat it as active.
- Negative: treat the block as data; do not treat it as active.

| Method | F1 | AUROC | FNR | FPR |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Rule features + Logistic Regression | 0.6667 | 0.5000 | 0.0000 | 1.0000 |
| Surrogate PatchDistill | 0.6667 | 0.5000 | 0.0000 | 1.0000 |

Interpretation:

- Contrastive framing fully breaks the current hand-built rule features.
- TF-IDF with unigrams and bigrams remains perfect, likely because bigram order
  captures the synthetic contrast.
- Next step: run GPT-2/Qwen HF features on `contrastive`, then add
  paraphrased contrastive frames so template bigrams do not dominate.

## 2026-06-08 Colab A100 GPT-2 Contrastive Pilot

Environment visible in the saved notebook output:

- GPU: NVIDIA A100-SXM4-80GB
- Model: `gpt2`
- Data profile: `contrastive`
- Synthetic examples: 320
- HF feature examples: 160
- Residual patch examples: 24
- Layers: `0,6,11`
- Run name: `gpt2_a100_contrastive_001`

Surrogate contrastive split:

| Method | F1 | AUROC | FNR | FPR |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Rule features + Logistic Regression | 0.6667 | 0.5000 | 0.0000 | 1.0000 |
| Surrogate PatchDistill | 0.6667 | 0.5000 | 0.0000 | 1.0000 |

GPT-2 one-pass / patching pilot:

| Component | Result |
| --- | ---: |
| `hf-extract` examples | 160 |
| `hf-patch` examples | 24 |
| Proxy MAE | 0.0902 |
| Proxy MSE | 0.03227 |
| HF features-only detector F1 | 1.0000 |
| HF features-only detector AUROC | 1.0000 |
| Distilled patch detector F1 | 1.0000 |
| Distilled patch detector AUROC | 1.0000 |

Interpretation:

- Contrastive framing breaks the current hand-built rule features but not TF-IDF
  or GPT-2 one-pass features.
- The proxy error is higher than in the stress and matched pilots, suggesting
  that contrastive source-boundary framing makes the patch-effect target harder
  to approximate.
- Since features-only and distilled detectors are both perfect, this run still
  does not show an advantage for PatchDistill. It does provide a stronger
  failure analysis: current templates remain separable by non-causal features.
- Next step: add paraphrased contrastive frames and held-out frame families, then
  compare features-only vs distilled detectors under a split that suppresses
  template-order shortcuts.
