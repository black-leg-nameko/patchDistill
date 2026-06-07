#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gpt2}"
RUN_NAME="${RUN_NAME:-gpt2_a100_pilot}"
LAYERS="${LAYERS:-0,6,11}"
N_DATA="${N_DATA:-160}"
MAX_FEATURE_EXAMPLES="${MAX_FEATURE_EXAMPLES:-80}"
MAX_PATCH_EXAMPLES="${MAX_PATCH_EXAMPLES:-12}"
MAX_PATCH_POSITIONS="${MAX_PATCH_POSITIONS:-3}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

ATTN_ARGS=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARGS=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

nvidia-smi || true
python -m pip install -r requirements.txt

python -m patchdistill.cli make-data --n "${N_DATA}" --out data/synthetic_direct_pi.jsonl
python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi.jsonl \
  --out runs/surrogate_mvp \
  --split template

python -m patchdistill.cli hf-extract \
  --model "${MODEL_NAME}" \
  --data data/synthetic_direct_pi.jsonl \
  --out "runs/${RUN_NAME}_features.jsonl" \
  --max-examples "${MAX_FEATURE_EXAMPLES}" \
  --layers "${LAYERS}" \
  --dtype "${DTYPE}" \
  "${ATTN_ARGS[@]}"

python -m patchdistill.cli hf-patch \
  --model "${MODEL_NAME}" \
  --data data/synthetic_direct_pi.jsonl \
  --out "runs/${RUN_NAME}_patch.jsonl" \
  --layers "${LAYERS}" \
  --max-examples "${MAX_PATCH_EXAMPLES}" \
  --max-positions "${MAX_PATCH_POSITIONS}" \
  --dtype "${DTYPE}" \
  "${ATTN_ARGS[@]}"

python -m patchdistill.cli fit-proxy \
  --features "runs/${RUN_NAME}_features.jsonl" \
  --patch "runs/${RUN_NAME}_patch.jsonl" \
  --out "runs/${RUN_NAME}_proxy"

python -m patchdistill.cli fit-detector \
  --features "runs/${RUN_NAME}_features.jsonl" \
  --out "runs/${RUN_NAME}_detector_features_only"

python -m patchdistill.cli fit-detector \
  --features "runs/${RUN_NAME}_features.jsonl" \
  --patch "runs/${RUN_NAME}_patch.jsonl" \
  --out "runs/${RUN_NAME}_detector_distilled"

python -m patchdistill.cli collect-results \
  --runs runs \
  --out runs/summary.json \
  --markdown runs/summary.md

