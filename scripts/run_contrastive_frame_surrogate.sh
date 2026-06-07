#!/usr/bin/env bash
set -euo pipefail

python -m patchdistill.cli make-data \
  --n 400 \
  --profile contrastive_frame \
  --seed 31 \
  --out data/synthetic_direct_pi_contrastive_frame.jsonl

python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi_contrastive_frame.jsonl \
  --out runs/surrogate_contrastive_frame \
  --split group \
  --seed 31

python -m patchdistill.cli collect-results \
  --runs runs \
  --out runs/summary.json \
  --markdown runs/summary.md
