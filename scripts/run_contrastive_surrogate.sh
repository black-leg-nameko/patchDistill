#!/usr/bin/env bash
set -euo pipefail

python -m patchdistill.cli make-data \
  --n 320 \
  --profile contrastive \
  --seed 29 \
  --out data/synthetic_direct_pi_contrastive.jsonl

python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi_contrastive.jsonl \
  --out runs/surrogate_contrastive \
  --split template \
  --seed 29

python -m patchdistill.cli collect-results \
  --runs runs \
  --out runs/summary.json \
  --markdown runs/summary.md

