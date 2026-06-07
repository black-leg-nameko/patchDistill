# Implementation Notes

The current scaffold is intentionally split into a local path and a GPU path.

## Local Surrogate Path

Files:

- `patchdistill/data.py`: synthetic clean, injected, and benign-hard examples.
- `patchdistill/lexical.py`: one-pass rule features and pseudo patch signatures.
- `patchdistill/experiments.py`: TF-IDF baseline, rule baseline, and surrogate
  PatchDistill classifier.

Purpose:

- Validate data format, splits, metrics, and downstream reporting.
- Avoid requiring local `torch` or `transformers`.

Caveat:

- `pseudo_patch_signature` is not causal evidence. It exists only to test the
  distillation plumbing before GPU patching runs.

## HF / Colab Path

Files:

- `patchdistill/hf.py`: one-pass hidden-state, attention, and logit features.
- `patchdistill/patching.py`: selected residual-stream patch effects.
- `patchdistill/distill.py`: proxy fitting and detector fitting from HF outputs.

Suggested pilot:

1. Generate 80 to 200 synthetic examples.
2. Run `hf-extract` on all examples with a tiny or small model.
3. Run `hf-patch` on 5 to 20 positive examples and a few selected layers.
4. Run `fit-proxy` to estimate `q_phi(r(x)) ~= PE(x)`.
5. Run `fit-detector` with and without `--patch` to measure the added value of
   distilled patch features.

The patcher currently patches residual block outputs at selected layer/token
positions. This is the right MVP granularity before moving to head-level
patching with TransformerLens or model-specific attention hooks.

