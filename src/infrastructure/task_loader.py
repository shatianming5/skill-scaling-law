"""任务集加载与预处理。

支持从 SkillsBench、SWE-bench、MINT 等数据源加载任务，
统一为内部 Task 格式，并提供分层抽样等采样策略。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import random


@dataclass
class Task:
    """统一任务数据格式。"""
    task_id: str
    source: str               # skillsbench / swebench / mint
    domain: str               # 领域标签
    difficulty: str           # easy / medium / hard
    instruction: str          # 任务指令
    ground_truth: Any         # 正确答案或验证函数路径
    metadata: dict = field(default_factory=dict)


class TaskLoader:
    """任务加载器，支持多数据源和分层抽样。"""

    def __init__(self, config: dict):
        self.config = config
        self.tasks: list[Task] = []

    def load_all(self) -> list[Task]:
        """从所有配置的数据源加载任务。"""
        for source_cfg in self.config["tasks"]["sources"]:
            loader_fn = self._get_loader(source_cfg["name"])
            tasks = loader_fn(source_cfg["path"])
            self.tasks.extend(tasks)
        return self.tasks

    def _get_loader(self, source_name: str):
        loaders = {
            "skillsbench": self._load_skillsbench,
            "swebench": self._load_swebench,
            "mint": self._load_mint,
        }
        if source_name not in loaders:
            raise ValueError(f"Unknown source: {source_name}")
        return loaders[source_name]

    def _load_skillsbench(self, path: str) -> list[Task]:
        """加载 SkillsBench 任务。"""
        tasks = []
        task_dir = Path(path)
        if not task_dir.exists():
            return tasks
        for task_file in sorted(task_dir.glob("*.json")):
            data = json.loads(task_file.read_text())
            tasks.append(Task(
                task_id=f"sb_{data['id']}",
                source="skillsbench",
                domain=data.get("domain", "general"),
                difficulty=data.get("difficulty", "medium"),
                instruction=data["instruction"],
                ground_truth=data.get("verification"),
                metadata=data.get("metadata", {}),
            ))
        return tasks

    def _load_swebench(self, path: str) -> list[Task]:
        """加载 SWE-bench Verified 任务。"""
        tasks = []
        task_dir = Path(path)
        if not task_dir.exists():
            return tasks
        for task_file in sorted(task_dir.glob("*.json")):
            data = json.loads(task_file.read_text())
            tasks.append(Task(
                task_id=f"swe_{data['instance_id']}",
                source="swebench",
                domain="software_engineering",
                difficulty=data.get("difficulty", "medium"),
                instruction=data["problem_statement"],
                ground_truth=data.get("test_patch"),
                metadata={"repo": data.get("repo", "")},
            ))
        return tasks

    def _load_mint(self, path: str) -> list[Task]:
        """加载 MINT/AgentBench 任务。"""
        tasks = []
        task_dir = Path(path)
        if not task_dir.exists():
            return tasks
        for task_file in sorted(task_dir.glob("*.json")):
            data = json.loads(task_file.read_text())
            tasks.append(Task(
                task_id=f"mint_{data['id']}",
                source="mint",
                domain=data.get("domain", "general"),
                difficulty=data.get("difficulty", "medium"),
                instruction=data["instruction"],
                ground_truth=data.get("answer"),
                metadata=data.get("metadata", {}),
            ))
        return tasks

    def sample(
        self,
        n: int,
        strategy: str = "stratified",
        seed: int = 42,
    ) -> list[Task]:
        """从已加载任务中采样。

        Args:
            n: 采样数量
            strategy: 'stratified'（按领域分层）或 'random'
            seed: 随机种子
        """
        rng = random.Random(seed)

        if strategy == "random":
            return rng.sample(self.tasks, min(n, len(self.tasks)))

        if strategy == "stratified":
            return self._stratified_sample(n, rng)

        raise ValueError(f"Unknown strategy: {strategy}")

    def _stratified_sample(self, n: int, rng: random.Random) -> list[Task]:
        """按领域分层抽样，保证各领域比例均衡。"""
        by_domain: dict[str, list[Task]] = {}
        for t in self.tasks:
            by_domain.setdefault(t.domain, []).append(t)

        n_domains = len(by_domain)
        per_domain = n // n_domains
        remainder = n % n_domains

        sampled = []
        for i, (domain, domain_tasks) in enumerate(sorted(by_domain.items())):
            k = per_domain + (1 if i < remainder else 0)
            k = min(k, len(domain_tasks))
            sampled.extend(rng.sample(domain_tasks, k))

        return sampled

    def get_by_difficulty(self, difficulty: str) -> list[Task]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def get_by_domain(self, domain: str) -> list[Task]:
        return [t for t in self.tasks if t.domain == domain]
