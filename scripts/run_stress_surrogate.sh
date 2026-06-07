#!/usr/bin/env bash
set -euo pipefail

python -m patchdistill.cli make-data \
  --n 320 \
  --profile stress \
  --seed 17 \
  --out data/synthetic_direct_pi_stress.jsonl

python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi_stress.jsonl \
  --out runs/surrogate_stress \
  --split template \
  --seed 17

python -m patchdistill.cli collect-results \
  --runs runs \
  --out runs/summary.json \
  --markdown runs/summary.md

