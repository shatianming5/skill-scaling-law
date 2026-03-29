"""统计检验模块。

提供 paired bootstrap test、Wilcoxon signed-rank test、Cohen's d 等统计方法。
"""

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """统计分析工具集。"""

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """计算 Cohen's d 效果量。"""
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Bootstrap 置信区间。

        Returns:
            (mean, ci_low, ci_high)
        """
        rng = np.random.RandomState(seed)
        boot_means = np.array([
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstrap)
        ])
        alpha = 1 - confidence
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return float(np.mean(data)), ci_low, ci_high

    @staticmethod
    def paired_bootstrap_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Paired bootstrap test.

        Returns:
            (observed_diff, p_value)
        """
        assert len(scores_a) == len(scores_b)
        observed_diff = float(np.mean(scores_a) - np.mean(scores_b))
        diffs = scores_a - scores_b

        rng = np.random.RandomState(seed)
        count = 0
        for _ in range(n_bootstrap):
            signs = rng.choice([-1, 1], size=len(diffs))
            boot_diff = np.mean(diffs * signs)
            if abs(boot_diff) >= abs(observed_diff):
                count += 1

        p_value = count / n_bootstrap
        return observed_diff, p_value

    @staticmethod
    def wilcoxon_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> tuple[float, float]:
        """Wilcoxon signed-rank test.

        Returns:
            (statistic, p_value)
        """
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
        return float(stat), float(p)

    @staticmethod
    def aggregate_by_condition(
        results: list[dict],
        score_key: str = "score",
    ) -> dict[str, dict]:
        """按条件聚合结果。"""
        by_condition: dict[str, list[float]] = {}
        for r in results:
            cond = r["condition"]
            by_condition.setdefault(cond, []).append(r[score_key])

        summary = {}
        for cond, scores in by_condition.items():
            arr = np.array(scores)
            mean, ci_low, ci_high = StatisticalAnalyzer.bootstrap_ci(arr)
            summary[cond] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(scores),
                "std": float(np.std(arr, ddof=1)),
            }
        return summary
