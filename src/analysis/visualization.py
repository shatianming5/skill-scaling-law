"""可视化模块。

生成论文级图表：scaling curve、heatmap、Pareto front 等。
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """实验结果可视化。"""

    def __init__(self, style: str = "seaborn-v0_8-paper"):
        self.style = style
        self._setup_style()

    def _setup_style(self):
        import matplotlib.pyplot as plt
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use("seaborn-v0_8")
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

    def plot_density_performance(
        self,
        results: list[dict],
        output_path: str,
    ):
        """RQ1: 信息密度 vs 性能曲线。"""
        import matplotlib.pyplot as plt
        from .statistics import StatisticalAnalyzer

        summary = StatisticalAnalyzer.aggregate_by_condition(results)

        level_tokens = {
            "L1": 50, "L2": 200, "L3": 500, "L4": 1500, "L5": 3000,
        }

        fig, ax = plt.subplots(figsize=(8, 5))

        tokens = []
        means = []
        ci_lows = []
        ci_highs = []

        for level in ["L1", "L2", "L3", "L4", "L5"]:
            if level not in summary:
                continue
            s = summary[level]
            tokens.append(level_tokens[level])
            means.append(s["mean"])
            ci_lows.append(s["ci_low"])
            ci_highs.append(s["ci_high"])

        tokens = np.array(tokens)
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        ax.plot(tokens, means, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax.fill_between(tokens, ci_lows, ci_highs, alpha=0.2, color="#2196F3")

        if "baseline" in summary:
            ax.axhline(
                y=summary["baseline"]["mean"],
                color="#F44336", linestyle="--", label="Baseline (no skill)",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Skill Token Count")
        ax.set_ylabel("Pass Rate")
        ax.set_title("RQ1: Information Density vs Agent Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved: {output_path}")

    def plot_pareto_frontier(
        self,
        results: list[dict],
        output_path: str,
    ):
        """RQ2: Quality-Quantity Pareto 曲线。"""
        import matplotlib.pyplot as plt
        from .statistics import StatisticalAnalyzer

        summary = StatisticalAnalyzer.aggregate_by_condition(results)

        fig, ax = plt.subplots(figsize=(8, 5))

        hc_sizes = {"HC-small": 100, "HC-medium": 200}
        ag_sizes = {"AG-small": 100, "AG-medium": 1000, "AG-large": 10000}

        # HC 线
        hc_x, hc_y = [], []
        for name, size in sorted(hc_sizes.items(), key=lambda x: x[1]):
            if name in summary:
                hc_x.append(size)
                hc_y.append(summary[name]["mean"])

        # AG 线
        ag_x, ag_y = [], []
        for name, size in sorted(ag_sizes.items(), key=lambda x: x[1]):
            if name in summary:
                ag_x.append(size)
                ag_y.append(summary[name]["mean"])

        if hc_x:
            ax.plot(hc_x, hc_y, "s-", color="#4CAF50", label="Human-Curated",
                    linewidth=2, markersize=8)
        if ag_x:
            ax.plot(ag_x, ag_y, "^-", color="#FF9800", label="Auto-Generated",
                    linewidth=2, markersize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Skill Pool Size")
        ax.set_ylabel("Pass Rate")
        ax.set_title("RQ2: Quality vs Quantity Pareto Frontier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved: {output_path}")

    def plot_scaling_curve(
        self,
        fit_results: dict,
        output_path: str,
    ):
        """RQ3: Scaling Law 拟合曲线。"""
        import matplotlib.pyplot as plt

        data = fit_results["data_points"]
        x = np.array(data["x"])
        y = np.array(data["y"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="#333", s=60, zorder=5, label="Observed")

        colors = {"power_law": "#2196F3", "logarithmic": "#4CAF50", "sigmoid": "#F44336"}
        x_smooth = np.linspace(max(x.min(), 1), x.max(), 200)

        from .curve_fitting import FUNCTION_REGISTRY

        best = fit_results["model_selection"]["best_by_aic"]
        for func_name, fit in fit_results["fits"].items():
            if not fit["converged"] or func_name not in FUNCTION_REGISTRY:
                continue
            spec = FUNCTION_REGISTRY[func_name]
            param_vals = [fit["params"][p]["value"] for p in spec["param_names"]]
            y_pred = spec["func"](x_smooth, *param_vals)

            style = "-" if func_name == best else "--"
            lw = 2.5 if func_name == best else 1.5
            label = f"{func_name} (R²={fit['r_squared']:.3f})"
            if func_name == best:
                label += " ★"
            ax.plot(
                x_smooth, y_pred, style,
                color=colors.get(func_name, "#999"),
                linewidth=lw, label=label,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Skill Pool Size")
        ax.set_ylabel("Pass Rate")
        ax.set_title("RQ3: Scaling Law Curve Fitting")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved: {output_path}")

    def plot_quality_quantity_heatmap(
        self,
        results: list[dict],
        output_path: str,
    ):
        """RQ3: 数量 × 质量 2D heatmap。"""
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # 按 (pool_size, hc_ratio) 聚合
        grid: dict[tuple, list] = {}
        for r in results:
            size = r.get("pool_size", 0)
            ratio = r.get("hc_ratio", 0)
            grid.setdefault((size, ratio), []).append(r["score"])

        if not grid:
            logger.warning("No data for heatmap")
            return

        sizes = sorted(set(k[0] for k in grid))
        ratios = sorted(set(k[1] for k in grid))
        matrix = np.zeros((len(ratios), len(sizes)))

        for i, ratio in enumerate(ratios):
            for j, size in enumerate(sizes):
                scores = grid.get((size, ratio), [])
                matrix[i, j] = np.mean(scores) if scores else np.nan

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, aspect="auto", origin="lower",
                        cmap="YlOrRd", norm=Normalize(vmin=0, vmax=1))
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45)
        ax.set_yticks(range(len(ratios)))
        ax.set_yticklabels([f"{r:.0%}" for r in ratios])
        ax.set_xlabel("Skill Pool Size")
        ax.set_ylabel("HC Ratio")
        ax.set_title("RQ3: Quality × Quantity Performance Heatmap")
        plt.colorbar(im, ax=ax, label="Pass Rate")

        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved: {output_path}")
