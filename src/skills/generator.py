"""多颗粒度 Skill 生成器。

给定知识源，使用 LLM 生成 5 个颗粒度级别（L1-L5）的 Skill，
保证严格包含关系：L(n+1) ⊃ L(n)。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


LEVEL_SPECS = {
    "L1": {
        "name": "one_liner",
        "target_tokens": 50,
        "prompt_instruction": (
            "Write a single-sentence hint (~50 tokens) that captures the core "
            "knowledge needed. Be specific and actionable."
        ),
    },
    "L2": {
        "name": "checklist",
        "target_tokens": 200,
        "prompt_instruction": (
            "Write a 3-5 bullet checklist (~200 tokens). Each bullet should "
            "state WHAT to do and WHY. Include ALL information from L1."
        ),
    },
    "L3": {
        "name": "focused_sop",
        "target_tokens": 500,
        "prompt_instruction": (
            "Write a focused SOP (~500 tokens) with three sections:\n"
            "- Background: Context and rationale\n"
            "- Steps: 3-5 concrete steps\n"
            "- Verification: How to confirm correctness\n"
            "Include ALL information from L2."
        ),
    },
    "L4": {
        "name": "comprehensive_guide",
        "target_tokens": 1500,
        "prompt_instruction": (
            "Write a comprehensive SKILL.md (~1500 tokens) including:\n"
            "- Background, Use Cases, Inputs/Outputs\n"
            "- Steps (8-12 detailed steps)\n"
            "- Edge Cases and common pitfalls\n"
            "- Verification and evidence\n"
            "Include ALL information from L3."
        ),
    },
    "L5": {
        "name": "documentation",
        "target_tokens": 3000,
        "prompt_instruction": (
            "Write full documentation (~3000+ tokens) including:\n"
            "- Historical background and motivation\n"
            "- Complete API/interface reference\n"
            "- Multiple code examples\n"
            "- Troubleshooting guide\n"
            "- Related concepts and further reading\n"
            "Include ALL information from L4."
        ),
    },
}


@dataclass
class GeneratedSkill:
    """生成的 Skill。"""
    skill_id: str
    task_id: str
    level: str
    content: str
    token_count: int
    knowledge_source: str


class SkillGenerator:
    """多颗粒度 Skill 生成器。"""

    def __init__(self, config: dict):
        self.config = config
        gen_cfg = config.get("generation", {})
        self.model = gen_cfg.get("model", "gpt-4.1")
        self.ensure_containment = gen_cfg.get("ensure_containment", True)

    def generate_all_levels(
        self,
        task_id: str,
        knowledge: str,
    ) -> dict[str, GeneratedSkill]:
        """为给定知识生成全部 5 个颗粒度级别的 Skill。"""
        results = {}
        previous_content = ""

        for level in ["L1", "L2", "L3", "L4", "L5"]:
            skill = self._generate_single(
                task_id=task_id,
                knowledge=knowledge,
                level=level,
                previous_content=previous_content,
            )
            results[level] = skill
            previous_content = skill.content

        if self.ensure_containment:
            self._validate_containment(results)

        return results

    def _generate_single(
        self,
        task_id: str,
        knowledge: str,
        level: str,
        previous_content: str = "",
    ) -> GeneratedSkill:
        """生成单个颗粒度级别的 Skill。"""
        spec = LEVEL_SPECS[level]
        prompt = self._build_prompt(knowledge, level, previous_content)

        # 实际调用 LLM（此处为框架占位，需接入 API）
        content = self._call_llm(prompt)

        token_count = len(content.split())

        return GeneratedSkill(
            skill_id=f"{task_id}_{level}",
            task_id=task_id,
            level=level,
            content=content,
            token_count=token_count,
            knowledge_source=knowledge[:200],
        )

    def _build_prompt(
        self,
        knowledge: str,
        level: str,
        previous_content: str,
    ) -> str:
        spec = LEVEL_SPECS[level]
        parts = [
            f"Knowledge to encode:\n{knowledge}\n",
            f"Target level: {level} ({spec['name']}, ~{spec['target_tokens']} tokens)",
            spec["prompt_instruction"],
        ]
        if previous_content:
            parts.append(
                f"\nPrevious level content (MUST be fully contained):\n"
                f"{previous_content}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成内容。需要对接实际 API。"""
        raise NotImplementedError(
            "LLM API call not yet implemented. "
            "Integrate with OpenAI/Anthropic client here."
        )

    @staticmethod
    def _validate_containment(skills: dict[str, GeneratedSkill]):
        """验证信息包含关系（启发式检查）。"""
        levels = ["L1", "L2", "L3", "L4", "L5"]
        for i in range(len(levels) - 1):
            current = skills[levels[i]]
            next_level = skills[levels[i + 1]]
            # 简单检查：低级别的关键词应出现在高级别中
            current_words = set(current.content.lower().split())
            next_words = set(next_level.content.lower().split())
            overlap = len(current_words & next_words) / max(len(current_words), 1)
            if overlap < 0.7:
                logger.warning(
                    f"Containment check: {levels[i]}→{levels[i+1]} "
                    f"overlap={overlap:.2f} (expected ≥0.7)"
                )

    def save_skills(
        self,
        skills: dict[str, GeneratedSkill],
        output_dir: str,
    ):
        """保存生成的 Skill 到文件。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for level, skill in skills.items():
            data = {
                "skill_id": skill.skill_id,
                "task_id": skill.task_id,
                "level": skill.level,
                "content": skill.content,
                "token_count": skill.token_count,
                "knowledge_source": skill.knowledge_source,
            }
            path = out / f"{skill.skill_id}.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
