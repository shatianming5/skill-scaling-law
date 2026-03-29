#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== RQ3: Scaling Law Experiment ==="
echo "Starting at $(date)"

python -m src.experiments.rq3_scaling configs/rq3_scaling.yaml

echo "=== RQ3 Complete ==="
echo "Results saved to results/rq3/"
