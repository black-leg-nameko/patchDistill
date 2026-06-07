#!/usr/bin/env bash
set -euo pipefail

python -m patchdistill.cli make-data \
  --n 320 \
  --profile matched \
  --seed 23 \
  --out data/synthetic_direct_pi_matched.jsonl

python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi_matched.jsonl \
  --out runs/surrogate_matched \
  --split template \
  --seed 23

python -m patchdistill.cli collect-results \
  --runs runs \
  --out runs/summary.json \
  --markdown runs/summary.md

