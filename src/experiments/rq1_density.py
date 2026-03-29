"""RQ1：Skill 信息密度实验。

测量不同颗粒度级别（L1-L5）的 Skill 对 Agent 性能的影响，
找到最优信息密度并分析其与任务难度/领域的关系。
"""

import asyncio
import json
import logging
from pathlib import Path

from ..infrastructure import TaskLoader, AgentRunner, SkillInjector, Evaluator
from ..skills import SkillGenerator
from ..utils.config import load_config

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = "You are a helpful assistant solving programming tasks."


class RQ1Experiment:
    """RQ1 信息密度实验。"""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.task_loader = TaskLoader(self.config)
        self.agent_runner = AgentRunner(self.config)
        self.skill_injector = SkillInjector(self.config)
        self.evaluator = Evaluator(self.config)
        self.skill_generator = SkillGenerator(self.config)
        self.output_dir = Path(self.config["analysis"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self):
        """准备任务集和 Skill。"""
        logger.info("Loading tasks...")
        self.task_loader.load_all()
        sampling_cfg = self.config["task_sampling"]
        self.tasks = self.task_loader.sample(
            n=sampling_cfg["n_tasks"],
            strategy=sampling_cfg["strategy"],
            seed=sampling_cfg["seed"],
        )
        logger.info(f"Selected {len(self.tasks)} tasks")

    def generate_skills(self):
        """为每个任务生成 5 级颗粒度 Skill。"""
        self.task_skills: dict[str, dict] = {}
        for task in self.tasks:
            knowledge = self._extract_knowledge(task)
            skills = self.skill_generator.generate_all_levels(
                task_id=task.task_id,
                knowledge=knowledge,
            )
            self.task_skills[task.task_id] = skills
            self.skill_generator.save_skills(
                skills,
                str(self.output_dir / "generated_skills" / task.task_id),
            )
        logger.info(f"Generated skills for {len(self.task_skills)} tasks")

    async def run(self):
        """执行全部实验条件。"""
        conditions = self.config["conditions"]
        models = [m["name"] for m in self.config["agents"]["models"]]
        repeats = self.config["evaluation"]["repeats"]

        all_runs = []
        for task in self.tasks:
            for cond in conditions:
                for model in models:
                    for r in range(repeats):
                        run_id = (
                            f"rq1_{task.task_id}_{cond['name']}_{model}_r{r}"
                        )
                        system_prompt = self._build_prompt(task, cond)
                        all_runs.append({
                            "run_id": run_id,
                            "task_id": task.task_id,
                            "model": model,
                            "system_prompt": system_prompt,
                            "user_prompt": task.instruction,
                            "condition": cond["name"],
                            "repeat": r,
                        })

        logger.info(f"Total runs: {len(all_runs)}")

        # 批量执行
        responses = await self.agent_runner.run_batch(all_runs)

        # 评估
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
                "tokens_injected": response.tokens_used,
            })

        # 保存结果
        output_path = self.output_dir / "raw_results.json"
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Results saved to {output_path}")
        return results

    def _build_prompt(self, task, condition: dict) -> str:
        """根据条件构建 system prompt。"""
        if condition["name"] == "baseline":
            return BASE_SYSTEM_PROMPT

        level = condition.get("granularity", condition["name"])
        task_skills = self.task_skills.get(task.task_id, {})
        skill = task_skills.get(level)
        if skill is None:
            return BASE_SYSTEM_PROMPT

        prompt, _ = self.skill_injector.inject(
            BASE_SYSTEM_PROMPT,
            [{"id": skill.skill_id, "content": skill.content}],
        )
        return prompt

    @staticmethod
    def _extract_knowledge(task) -> str:
        """从任务中提取核心知识（用于 Skill 生成）。"""
        return task.instruction


def main(config_path: str = "configs/rq1_density.yaml"):
    exp = RQ1Experiment(config_path)
    exp.prepare()
    exp.generate_skills()
    asyncio.run(exp.run())


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/rq1_density.yaml"
    main(cfg)
