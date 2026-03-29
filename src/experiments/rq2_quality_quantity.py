"""RQ2：Quality vs Quantity 实验。

对比高质量少量（HC）vs 低质量大量（AG）Skill 对 Agent 性能的影响，
分析最优 quality-quantity 配比。
"""

import asyncio
import json
import logging
from pathlib import Path

from ..infrastructure import TaskLoader, AgentRunner, SkillInjector, Evaluator
from ..skills import SkillRetriever, SkillPool
from ..utils.config import load_config

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "You are a helpful assistant solving programming tasks."


class RQ2Experiment:
    """RQ2 质量 vs 数量实验。"""

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
        logger.info("Loading tasks...")
        self.task_loader.load_all()
        sampling_cfg = self.config["task_sampling"]
        self.tasks = self.task_loader.sample(
            n=sampling_cfg["n_tasks"],
            strategy=sampling_cfg["strategy"],
            seed=sampling_cfg["seed"],
        )

        # 加载 Skill 池
        self.full_pool = SkillPool()
        pool_cfg = self.config["skill_pools"]
        self.full_pool.load_from_dir(
            pool_cfg["human_curated"]["path"], "human_curated"
        )
        self.full_pool.load_from_dir(
            pool_cfg["auto_generated"]["path"], "auto_generated"
        )
        logger.info(f"Full pool: {self.full_pool.size} skills")

    def build_conditions(self) -> dict[str, SkillPool]:
        """构建所有实验条件的 Skill 池。"""
        conditions = {}

        # 固定数量条件
        for cond in self.config["fixed_count_conditions"]:
            name = cond["name"]
            if name == "Baseline":
                conditions[name] = SkillPool()
                continue

            if "composition" in cond:
                pool = self.full_pool.build_mixed(
                    total=cond["size"],
                    hc_ratio=cond["composition"]["human_curated"],
                )
            else:
                source = cond.get("pool")
                pool = self.full_pool.sample(
                    n=cond["size"],
                    source=source,
                )
            conditions[name] = pool
            logger.info(f"Condition {name}: {pool.size} skills")

        return conditions

    async def run(self):
        """执行全部实验条件。"""
        conditions = self.build_conditions()
        models = [m["name"] for m in self.config["agents"]["models"]]
        repeats = self.config["evaluation"]["repeats"]
        top_k = self.config["retrieval"]["top_k"]

        all_runs = []
        for task in self.tasks:
            for cond_name, pool in conditions.items():
                # 检索 Skill
                skills_to_inject = []
                if pool.size > 0:
                    retriever = SkillRetriever()
                    retriever.index(pool.to_retriever_format())
                    results = retriever.retrieve(task.instruction, top_k=top_k)
                    skills_to_inject = [
                        {"id": r.skill_id, "content": r.content}
                        for r in results
                    ]

                system_prompt, injection_record = self.skill_injector.inject(
                    BASE_SYSTEM_PROMPT, skills_to_inject
                )

                for model in models:
                    for r in range(repeats):
                        run_id = (
                            f"rq2_{task.task_id}_{cond_name}_{model}_r{r}"
                        )
                        all_runs.append({
                            "run_id": run_id,
                            "task_id": task.task_id,
                            "model": model,
                            "system_prompt": system_prompt,
                            "user_prompt": task.instruction,
                            "condition": cond_name,
                            "repeat": r,
                            "injection": {
                                "skill_ids": injection_record.skill_ids,
                                "total_tokens": injection_record.total_tokens,
                            },
                        })

        logger.info(f"Total runs: {len(all_runs)}")

        responses = await self.agent_runner.run_batch(all_runs)

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
            results.append({
                "run_id": run_info["run_id"],
                "task_id": run_info["task_id"],
                "condition": run_info["condition"],
                "model": run_info["model"],
                "repeat": run_info["repeat"],
                "passed": eval_result.passed,
                "score": eval_result.score,
                "injection": run_info["injection"],
            })

        output_path = self.output_dir / "raw_results.json"
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Results saved to {output_path}")
        return results


def main(config_path: str = "configs/rq2_quality_quantity.yaml"):
    exp = RQ2Experiment(config_path)
    exp.prepare()
    asyncio.run(exp.run())


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/rq2_quality_quantity.yaml"
    main(cfg)
