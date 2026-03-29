"""RQ3：Scaling Law 实验。

拟合 Agent 性能随 Skill 库规模变化的数学模型，
分析数量、质量、领域覆盖度的联合效应。
"""

import asyncio
import json
import logging
from pathlib import Path

from ..infrastructure import TaskLoader, AgentRunner, SkillInjector, Evaluator
from ..skills import SkillRetriever, SkillPool
from ..analysis.curve_fitting import ScalingLawFitter
from ..utils.config import load_config

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "You are a helpful assistant solving programming tasks."


class RQ3Experiment:
    """RQ3 Scaling Law 实验。"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.task_loader = TaskLoader(self.config)
        self.agent_runner = AgentRunner(self.config)
        self.skill_injector = SkillInjector(self.config)
        self.evaluator = Evaluator(self.config)
        self.output_dir = Path(self.config["analysis"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """准备任务集和 Skill 池。"""
        self.task_loader.load_all()
        sampling_cfg = self.config["task_sampling"]
        self.tasks = self.task_loader.sample(
            n=sampling_cfg["n_tasks"],
            strategy=sampling_cfg["strategy"],
            seed=sampling_cfg["seed"],
        )

        self.full_pool = SkillPool()
        self.full_pool.load_from_dir(
            self.config["skill_pools"]["human_curated"]["path"],
            "human_curated",
        )
        self.full_pool.load_from_dir(
            self.config["skill_pools"]["auto_generated"]["path"],
            "auto_generated",
        )
        logger.info(
            f"Prepared {len(self.tasks)} tasks, {self.full_pool.size} skills"
        )

    async def run_quantity_scaling(self) -> list[dict]:
        """数量 Scaling 实验。"""
        qs_cfg = self.config["quantity_scaling"]
        size_gradient = qs_cfg["size_gradient"]
        n_subsets = qs_cfg["subset_samples"]
        top_k = qs_cfg["retrieval_top_k"]
        model_schedule = qs_cfg["model_schedule"]
        repeats = self.config["evaluation"]["repeats"]

        all_runs = []
        for size in size_gradient:
            if size == 0:
                subsets = [SkillPool()]
            else:
                subsets = [
                    self.full_pool.sample(size, source="human_curated", seed=42 + s)
                    for s in range(n_subsets)
                ]

            for s_idx, pool in enumerate(subsets):
                retriever = None
                if pool.size > 0:
                    retriever = SkillRetriever()
                    retriever.index(pool.to_retriever_format())

                for task in self.tasks:
                    skills_to_inject = []
                    if retriever:
                        results = retriever.retrieve(task.instruction, top_k)
                        skills_to_inject = [
                            {"id": r.skill_id, "content": r.content}
                            for r in results
                        ]

                    system_prompt, record = self.skill_injector.inject(
                        BASE_SYSTEM_PROMPT, skills_to_inject
                    )

                    for model_name, schedule in model_schedule.items():
                        if schedule != "all" and size not in schedule:
                            continue
                        for r in range(repeats):
                            run_id = (
                                f"rq3_qty_s{size}_sub{s_idx}"
                                f"_{task.task_id}_{model_name}_r{r}"
                            )
                            all_runs.append({
                                "run_id": run_id,
                                "task_id": task.task_id,
                                "model": model_name,
                                "system_prompt": system_prompt,
                                "user_prompt": task.instruction,
                                "condition": f"qty_{size}",
                                "subset": s_idx,
                                "pool_size": size,
                                "repeat": r,
                            })

        logger.info(f"Quantity scaling: {len(all_runs)} runs")
        responses = await self.agent_runner.run_batch(all_runs)
        return self._collect_results(all_runs, responses, "quantity_scaling")

    async def run_quality_scaling(self) -> list[dict]:
        """质量 Scaling 实验。"""
        qs_cfg = self.config["quality_scaling"]
        fixed_size = qs_cfg["fixed_size"]
        models = [m["name"] for m in self.config["agents"]["models"]]
        repeats = self.config["evaluation"]["repeats"]
        top_k = self.config.get("quantity_scaling", {}).get("retrieval_top_k", 3)

        all_runs = []
        for cond in qs_cfg["conditions"]:
            pool = self.full_pool.build_mixed(
                total=fixed_size,
                hc_ratio=cond["hc_ratio"],
            )
            retriever = SkillRetriever()
            retriever.index(pool.to_retriever_format())

            for task in self.tasks:
                results = retriever.retrieve(task.instruction, top_k)
                skills_to_inject = [
                    {"id": r.skill_id, "content": r.content}
                    for r in results
                ]
                system_prompt, _ = self.skill_injector.inject(
                    BASE_SYSTEM_PROMPT, skills_to_inject
                )

                for model in models:
                    for r in range(repeats):
                        run_id = (
                            f"rq3_qual_{cond['name']}"
                            f"_{task.task_id}_{model}_r{r}"
                        )
                        all_runs.append({
                            "run_id": run_id,
                            "task_id": task.task_id,
                            "model": model,
                            "system_prompt": system_prompt,
                            "user_prompt": task.instruction,
                            "condition": cond["name"],
                            "hc_ratio": cond["hc_ratio"],
                            "repeat": r,
                        })

        logger.info(f"Quality scaling: {len(all_runs)} runs")
        responses = await self.agent_runner.run_batch(all_runs)
        return self._collect_results(all_runs, responses, "quality_scaling")

    async def run_domain_scaling(self) -> list[dict]:
        """领域 Scaling 实验。"""
        ds_cfg = self.config["domain_scaling"]
        fixed_size = ds_cfg["fixed_size"]
        models = [m["name"] for m in self.config["agents"]["models"]]
        repeats = self.config["evaluation"]["repeats"]
        top_k = self.config.get("quantity_scaling", {}).get("retrieval_top_k", 3)

        all_runs = []
        for cond in ds_cfg["conditions"]:
            for task in self.tasks:
                pool = self.full_pool.build_domain_distributed(
                    total=fixed_size,
                    target_domain=task.domain,
                    strategy=cond["name"],
                )
                retriever = SkillRetriever()
                retriever.index(pool.to_retriever_format())
                results = retriever.retrieve(task.instruction, top_k)
                skills_to_inject = [
                    {"id": r.skill_id, "content": r.content}
                    for r in results
                ]
                system_prompt, _ = self.skill_injector.inject(
                    BASE_SYSTEM_PROMPT, skills_to_inject
                )

                for model in models:
                    for r in range(repeats):
                        run_id = (
                            f"rq3_dom_{cond['name']}"
                            f"_{task.task_id}_{model}_r{r}"
                        )
                        all_runs.append({
                            "run_id": run_id,
                            "task_id": task.task_id,
                            "model": model,
                            "system_prompt": system_prompt,
                            "user_prompt": task.instruction,
                            "condition": cond["name"],
                            "strategy": cond["strategy"],
                            "repeat": r,
                        })

        logger.info(f"Domain scaling: {len(all_runs)} runs")
        responses = await self.agent_runner.run_batch(all_runs)
        return self._collect_results(all_runs, responses, "domain_scaling")

    def _collect_results(
        self,
        all_runs: list[dict],
        responses: list,
        experiment_name: str,
    ) -> list[dict]:
        results = []
        for run_info, response in zip(all_runs, responses):
            if isinstance(response, Exception):
                logger.error(f"Run {run_info['run_id']} failed: {response}")
                continue

            task = next(
                t for t in self.tasks if t.task_id == run_info["task_id"]
            )
            eval_result = self.evaluator.evaluate_single(
                run_id=run_info["run_id"],
                task_id=run_info["task_id"],
                agent_output=response.output,
                ground_truth=task.ground_truth,
            )
            result = {
                "run_id": run_info["run_id"],
                "task_id": run_info["task_id"],
                "condition": run_info["condition"],
                "model": run_info["model"],
                "repeat": run_info["repeat"],
                "passed": eval_result.passed,
                "score": eval_result.score,
            }
            result.update(
                {k: v for k, v in run_info.items()
                 if k not in result and k not in ("system_prompt", "user_prompt")}
            )
            results.append(result)

        output_path = self.output_dir / f"{experiment_name}_results.json"
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        return results

    async def run_all(self):
        """执行所有子实验。"""
        self.prepare()

        qty_results = await self.run_quantity_scaling()
        qual_results = await self.run_quality_scaling()
        dom_results = await self.run_domain_scaling()

        # 曲线拟合
        fitter = ScalingLawFitter()
        fit_results = fitter.fit_all(qty_results, self.config["curve_fitting"])

        output_path = self.output_dir / "scaling_law_fits.json"
        output_path.write_text(json.dumps(fit_results, indent=2))
        logger.info(f"Scaling law fits saved to {output_path}")

        return {
            "quantity": qty_results,
            "quality": qual_results,
            "domain": dom_results,
            "fits": fit_results,
        }


def main(config_path: str = "configs/rq3_scaling.yaml"):
    exp = RQ3Experiment(config_path)
    asyncio.run(exp.run_all())


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/rq3_scaling.yaml"
    main(cfg)
