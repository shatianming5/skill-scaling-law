#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== RQ1: Information Density Experiment ==="
echo "Starting at $(date)"

python -m src.experiments.rq1_density configs/rq1_density.yaml

echo "=== RQ1 Complete ==="
echo "Results saved to results/rq1/"
