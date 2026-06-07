#!/usr/bin/env bash
set -euo pipefail

python -m patchdistill.cli make-data --n 160 --out data/synthetic_direct_pi.jsonl
python -m patchdistill.cli run-surrogate \
  --data data/synthetic_direct_pi.jsonl \
  --out runs/surrogate_mvp \
  --split template

