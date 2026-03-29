#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== Aggregated Analysis ==="
echo "Starting at $(date)"

python -c "
from src.analysis import StatisticalAnalyzer, Visualizer
from src.utils.io import load_json

viz = Visualizer()

# RQ1
print('Analyzing RQ1...')
rq1 = load_json('results/rq1/raw_results.json')
viz.plot_density_performance(rq1, 'results/rq1/density_performance.png')

# RQ2
print('Analyzing RQ2...')
rq2 = load_json('results/rq2/raw_results.json')
viz.plot_pareto_frontier(rq2, 'results/rq2/pareto_frontier.png')

# RQ3
print('Analyzing RQ3...')
fits = load_json('results/rq3/scaling_law_fits.json')
viz.plot_scaling_curve(fits, 'results/rq3/scaling_curve.png')

print('All analysis complete.')
"

echo "=== Analysis Complete ==="
