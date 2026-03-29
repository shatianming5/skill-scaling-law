#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== RQ2: Quality vs Quantity Experiment ==="
echo "Starting at $(date)"

python -m src.experiments.rq2_quality_quantity configs/rq2_quality_quantity.yaml

echo "=== RQ2 Complete ==="
echo "Results saved to results/rq2/"
