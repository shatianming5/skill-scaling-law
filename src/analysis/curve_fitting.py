"""Scaling Law 曲线拟合。

尝试多种函数形式拟合 Agent 性能 vs Skill 规模关系，
使用 AIC/BIC 进行模型选择。
"""

import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def power_law(n, a, alpha, b):
    """P = a * N^alpha + b"""
    return a * np.power(np.maximum(n, 1e-10), alpha) + b


def logarithmic(n, a, b):
    """P = a * log(N) + b"""
    return a * np.log(np.maximum(n, 1e-10)) + b


def sigmoid(n, l_max, k, n0):
    """P = L / (1 + exp(-k * (N - N0)))"""
    return l_max / (1.0 + np.exp(-k * (n - n0)))


FUNCTION_REGISTRY = {
    "power_law": {
        "func": power_law,
        "param_names": ["a", "alpha", "b"],
        "p0": [0.1, 0.5, 0.3],
        "bounds": ([0, 0, 0], [10, 2, 1]),
    },
    "logarithmic": {
        "func": logarithmic,
        "param_names": ["a", "b"],
        "p0": [0.1, 0.3],
        "bounds": ([0, 0], [1, 1]),
    },
    "sigmoid": {
        "func": sigmoid,
        "param_names": ["L", "k", "N0"],
        "p0": [0.8, 0.01, 500],
        "bounds": ([0, 0, 0], [1, 1, 10000]),
    },
}


class ScalingLawFitter:
    """Scaling Law 曲线拟合器。"""

    def fit_single(
        self,
        x: np.ndarray,
        y: np.ndarray,
        func_name: str,
    ) -> dict:
        """拟合单个函数形式。"""
        spec = FUNCTION_REGISTRY[func_name]
        func = spec["func"]

        try:
            popt, pcov = curve_fit(
                func, x, y,
                p0=spec["p0"],
                bounds=spec["bounds"],
                maxfev=10000,
            )
            y_pred = func(x, *popt)
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            n = len(x)
            k = len(popt)

            # AIC / BIC
            mse = ss_res / n
            log_likelihood = -n / 2 * (np.log(2 * np.pi * mse) + 1)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # 参数置信区间
            param_std = np.sqrt(np.diag(pcov))
            params = {}
            for name, val, std in zip(spec["param_names"], popt, param_std):
                params[name] = {
                    "value": float(val),
                    "std": float(std),
                    "ci_95": [float(val - 1.96 * std), float(val + 1.96 * std)],
                }

            return {
                "function": func_name,
                "params": params,
                "aic": float(aic),
                "bic": float(bic),
                "r_squared": float(r_squared),
                "rmse": float(np.sqrt(mse)),
                "converged": True,
            }

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Fitting {func_name} failed: {e}")
            return {
                "function": func_name,
                "params": {},
                "aic": float("inf"),
                "bic": float("inf"),
                "r_squared": 0.0,
                "rmse": float("inf"),
                "converged": False,
                "error": str(e),
            }

    def fit_all(
        self,
        results: list[dict],
        fitting_config: dict,
    ) -> dict:
        """对实验结果拟合所有指定函数形式。"""
        # 提取 (pool_size, pass_rate) 数据点
        from ..analysis.statistics import StatisticalAnalyzer
        summary = StatisticalAnalyzer.aggregate_by_condition(results)

        sizes = []
        rates = []
        for cond, stats in summary.items():
            if cond.startswith("qty_"):
                size = int(cond.split("_")[1])
                sizes.append(size)
                rates.append(stats["mean"])

        x = np.array(sizes, dtype=float)
        y = np.array(rates, dtype=float)

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        func_names = [f["name"] for f in fitting_config["functions"]]
        fit_results = {}
        for name in func_names:
            if name in FUNCTION_REGISTRY:
                fit_results[name] = self.fit_single(x, y, name)

        # 模型选择
        best_aic = min(fit_results.values(), key=lambda r: r["aic"])
        best_bic = min(fit_results.values(), key=lambda r: r["bic"])

        return {
            "data_points": {"x": x.tolist(), "y": y.tolist()},
            "fits": fit_results,
            "model_selection": {
                "best_by_aic": best_aic["function"],
                "best_by_bic": best_bic["function"],
            },
        }

    def predict(
        self,
        func_name: str,
        params: dict,
        x_new: np.ndarray,
    ) -> np.ndarray:
        """使用拟合结果预测新数据点。"""
        spec = FUNCTION_REGISTRY[func_name]
        param_values = [params[name]["value"] for name in spec["param_names"]]
        return spec["func"](x_new, *param_values)
