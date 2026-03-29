"""Skill 池管理。

构建和管理 HC（Human-Curated）/ AG（Auto-Generated）/ Mix 类型的 Skill 池，
支持子集采样、混合构建等操作。
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Skill 数据。"""
    id: str
    content: str
    source: str       # "human_curated" or "auto_generated"
    domain: str = ""
    quality_score: float = 0.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class SkillPool:
    """Skill 池：支持 HC / AG / Mix 构建。"""

    def __init__(self):
        self.skills: dict[str, Skill] = {}

    @property
    def size(self) -> int:
        return len(self.skills)

    def load_from_dir(self, path: str, source: str) -> int:
        """从目录加载 Skill。"""
        skill_dir = Path(path)
        if not skill_dir.exists():
            logger.warning(f"Skill directory not found: {path}")
            return 0

        count = 0
        for f in sorted(skill_dir.glob("*.json")):
            data = json.loads(f.read_text())
            skill = Skill(
                id=data.get("id", f.stem),
                content=data["content"],
                source=source,
                domain=data.get("domain", ""),
                quality_score=data.get("quality_score", 0.0),
                token_count=data.get("token_count", len(data["content"].split())),
                metadata=data.get("metadata", {}),
            )
            self.skills[skill.id] = skill
            count += 1

        logger.info(f"Loaded {count} skills from {path} (source={source})")
        return count

    def add_skill(self, skill: Skill):
        self.skills[skill.id] = skill

    def get_by_source(self, source: str) -> list[Skill]:
        return [s for s in self.skills.values() if s.source == source]

    def get_by_domain(self, domain: str) -> list[Skill]:
        return [s for s in self.skills.values() if s.domain == domain]

    def sample(
        self,
        n: int,
        source: str | None = None,
        seed: int = 42,
    ) -> "SkillPool":
        """随机采样构建子池。"""
        rng = random.Random(seed)
        candidates = list(self.skills.values())
        if source:
            candidates = [s for s in candidates if s.source == source]

        sampled = rng.sample(candidates, min(n, len(candidates)))
        pool = SkillPool()
        for s in sampled:
            pool.add_skill(s)
        return pool

    def build_mixed(
        self,
        total: int,
        hc_ratio: float,
        seed: int = 42,
    ) -> "SkillPool":
        """构建 HC/AG 混合池。

        Args:
            total: 目标池大小
            hc_ratio: HC 占比 (0.0 - 1.0)
            seed: 随机种子
        """
        rng = random.Random(seed)
        hc_skills = self.get_by_source("human_curated")
        ag_skills = self.get_by_source("auto_generated")

        n_hc = int(total * hc_ratio)
        n_ag = total - n_hc

        selected_hc = rng.sample(hc_skills, min(n_hc, len(hc_skills)))
        selected_ag = rng.sample(ag_skills, min(n_ag, len(ag_skills)))

        pool = SkillPool()
        for s in selected_hc + selected_ag:
            pool.add_skill(s)

        logger.info(
            f"Mixed pool: {len(selected_hc)} HC + {len(selected_ag)} AG "
            f"= {pool.size} total"
        )
        return pool

    def build_domain_distributed(
        self,
        total: int,
        target_domain: str,
        strategy: str,
        seed: int = 42,
    ) -> "SkillPool":
        """按领域分布策略构建池。"""
        rng = random.Random(seed)
        all_skills = list(self.skills.values())
        domains = list(set(s.domain for s in all_skills if s.domain))

        if strategy == "concentrated":
            candidates = [s for s in all_skills if s.domain == target_domain]
            selected = rng.sample(candidates, min(total, len(candidates)))
        elif strategy == "broad_anchor":
            n_target = int(total * 0.7)
            n_other = total - n_target
            target_skills = [s for s in all_skills if s.domain == target_domain]
            other_skills = [s for s in all_skills if s.domain != target_domain]
            selected = (
                rng.sample(target_skills, min(n_target, len(target_skills)))
                + rng.sample(other_skills, min(n_other, len(other_skills)))
            )
        elif strategy == "balanced":
            per_domain = total // max(len(domains), 1)
            selected = []
            for d in domains:
                d_skills = [s for s in all_skills if s.domain == d]
                selected.extend(rng.sample(d_skills, min(per_domain, len(d_skills))))
        elif strategy == "random":
            selected = rng.sample(all_skills, min(total, len(all_skills)))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        pool = SkillPool()
        for s in selected:
            pool.add_skill(s)
        return pool

    def to_retriever_format(self) -> list[dict]:
        """转换为检索器所需格式。"""
        return [
            {"id": s.id, "content": s.content}
            for s in self.skills.values()
        ]
