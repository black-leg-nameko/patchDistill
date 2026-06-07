# PatchDistill

PatchDistill is a prototype for efficient prompt-injection detection via
distilled causal patch signatures.

The research plan in `研究計画書.md` proposes three phases:

1. Offline causal discovery with activation patching.
2. Distillation of expensive patch-effect signatures into one-pass features.
3. Online detection with a lightweight classifier and optional LLM judge.

This repository now contains a minimal experiment scaffold for those phases.
It has two execution paths:

- `surrogate` path: runs locally without PyTorch/Transformers. It validates the
  data, feature, proxy, detector, and metric pipeline with synthetic direct PI
  examples and pseudo patch signatures.
- `hf` path: runs on a GPU runtime such as Colab. It extracts hidden-state and
  attention features from Hugging Face causal LMs and computes residual-stream
  patch effects for selected layers and token positions.

## Quick Local Smoke Test

```bash
python -m patchdistill.cli make-data --n 160 --out data/synthetic_direct_pi.jsonl
python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi.jsonl \
  --out runs/surrogate_mvp \
  --split template
```

Expected outputs:

- `runs/surrogate_mvp/metrics.json`
- `runs/surrogate_mvp/predictions.csv`
- `runs/surrogate_mvp/proxy_metrics.json`

## Colab / GPU Path

Install dependencies in Colab:

```bash
pip install -r requirements.txt
```

Generate a small dataset:

```bash
python -m patchdistill.cli make-data --n 80 --out data/synthetic_direct_pi.jsonl
```

Extract one-pass internal features:

```bash
python -m patchdistill.cli hf-extract \
  --model sshleifer/tiny-gpt2 \
  --data data/synthetic_direct_pi.jsonl \
  --out runs/hf_features_tiny_gpt2.jsonl \
  --max-examples 20
```

Run a small residual patching pass:

```bash
python -m patchdistill.cli hf-patch \
  --model sshleifer/tiny-gpt2 \
  --data data/synthetic_direct_pi.jsonl \
  --out runs/hf_patch_tiny_gpt2.jsonl \
  --layers 0 \
  --max-examples 5
```

For real pilot experiments, replace `sshleifer/tiny-gpt2` with a small open
model such as `gpt2`, `EleutherAI/pythia-410m`, or `Qwen/Qwen2.5-0.5B-Instruct`.
The 7B-class experiments should be run only after the small-model pipeline is
validated.

## About Connecting This IDE To Google Colab

This IDE session cannot directly become a Colab GPU runtime by itself. The
practical workflow is to run the same repository code inside Colab, usually by
one of these routes:

- push/sync this folder to GitHub and clone it in Colab;
- upload this folder to Google Drive and mount Drive in Colab;
- expose the Colab machine through SSH or a VS Code-compatible tunnel, if your
  Colab setup allows that.

The code here is CLI-first so that local and Colab commands stay nearly
identical.

