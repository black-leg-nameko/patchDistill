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

