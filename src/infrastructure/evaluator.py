"""评估模块。

运行 deterministic verifier，计算 pass rate、效果量、置信区间等指标。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """单次评估结果。"""
    run_id: str
    task_id: str
    passed: bool
    score: float  # 0.0 or 1.0 for binary, continuous for partial
    error: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """聚合指标。"""
    condition: str
    n_tasks: int
    n_runs: int
    pass_rate: float
    pass_rate_ci_low: float
    pass_rate_ci_high: float
    cohens_d: Optional[float] = None


class Evaluator:
    """评估器：运行 verifier 并计算统计指标。"""

    def __init__(self, config: dict):
        self.config = config
        self.eval_cfg = config.get("evaluation", {})
        self.stat_cfg = config.get("statistics", {})

    def evaluate_single(
        self,
        run_id: str,
        task_id: str,
        agent_output: str,
        ground_truth,
    ) -> EvalResult:
        """评估单次 Agent 输出。"""
        try:
            if callable(ground_truth):
                passed = ground_truth(agent_output)
            elif isinstance(ground_truth, str):
                passed = self._string_match(agent_output, ground_truth)
            elif isinstance(ground_truth, dict):
                passed = self._structured_verify(agent_output, ground_truth)
            else:
                passed = str(ground_truth) in agent_output

            return EvalResult(
                run_id=run_id,
                task_id=task_id,
                passed=bool(passed),
                score=1.0 if passed else 0.0,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {run_id}: {e}")
            return EvalResult(
                run_id=run_id,
                task_id=task_id,
                passed=False,
                score=0.0,
                error=str(e),
            )

    @staticmethod
    def _string_match(output: str, expected: str) -> bool:
        return expected.strip().lower() in output.strip().lower()

    @staticmethod
    def _structured_verify(output: str, spec: dict) -> bool:
        """结构化验证（如检查 JSON 输出是否包含必要字段）。"""
        if "contains" in spec:
            return all(k in output for k in spec["contains"])
        if "exact" in spec:
            return output.strip() == spec["exact"].strip()
        return False

    def aggregate(
        self,
        results: list[EvalResult],
        condition: str,
        baseline_results: Optional[list[EvalResult]] = None,
    ) -> AggregatedMetrics:
        """聚合多次运行结果，计算统计指标。"""
        scores = np.array([r.score for r in results])
        pass_rate = float(np.mean(scores))

        ci_low, ci_high = self._bootstrap_ci(scores)

        cohens_d = None
        if baseline_results is not None:
            baseline_scores = np.array([r.score for r in baseline_results])
            cohens_d = self._cohens_d(scores, baseline_scores)

        return AggregatedMetrics(
            condition=condition,
            n_tasks=len(set(r.task_id for r in results)),
            n_runs=len(results),
            pass_rate=pass_rate,
            pass_rate_ci_low=ci_low,
            pass_rate_ci_high=ci_high,
            cohens_d=cohens_d,
        )

    def _bootstrap_ci(
        self,
        scores: np.ndarray,
        n_bootstrap: int = 10000,
        level: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap 置信区间。"""
        n_bootstrap = self.stat_cfg.get("confidence_interval", {}).get(
            "n_bootstrap", n_bootstrap
        )
        level = self.stat_cfg.get("confidence_interval", {}).get("level", level)

        rng = np.random.RandomState(42)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        alpha = 1 - level
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """计算 Cohen's d 效果量。"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    def save_results(self, results: list[EvalResult], output_path: str):
        """保存评估结果到 JSON。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "run_id": r.run_id,
                "task_id": r.task_id,
                "passed": r.passed,
                "score": r.score,
                "error": r.error,
            }
            for r in results
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
