"""Skill 注入模块。

将检索到的 Skill 拼接进 system prompt，并记录注入的 token 数等元信息。
"""

from dataclasses import dataclass


@dataclass
class InjectionRecord:
    """Skill 注入记录。"""
    skill_ids: list[str]
    total_tokens: int
    per_skill_tokens: list[int]
    template_used: str


class SkillInjector:
    """将 Skill 内容注入 Agent 的 system prompt。"""

    DEFAULT_TEMPLATE = (
        "## Relevant Skills\n\n{skills_content}\n"
    )

    def __init__(self, config: dict):
        self.config = config
        tpl = config.get("injection", {}).get("template")
        self.template = tpl if tpl else self.DEFAULT_TEMPLATE

    def inject(
        self,
        base_prompt: str,
        skills: list[dict],
    ) -> tuple[str, InjectionRecord]:
        """将 Skill 列表注入 base_prompt。

        Args:
            base_prompt: 原始 system prompt
            skills: Skill 字典列表，每个包含 'id' 和 'content' 字段

        Returns:
            (注入后的 system prompt, 注入记录)
        """
        if not skills:
            record = InjectionRecord(
                skill_ids=[],
                total_tokens=0,
                per_skill_tokens=[],
                template_used=self.template,
            )
            return base_prompt, record

        skill_contents = []
        skill_ids = []
        per_skill_tokens = []

        for skill in skills:
            content = skill["content"]
            skill_contents.append(content)
            skill_ids.append(skill.get("id", "unknown"))
            per_skill_tokens.append(self._count_tokens(content))

        joined = "\n---\n".join(skill_contents)
        skills_block = self.template.format(skills_content=joined)
        full_prompt = f"{base_prompt}\n\n{skills_block}"

        record = InjectionRecord(
            skill_ids=skill_ids,
            total_tokens=sum(per_skill_tokens),
            per_skill_tokens=per_skill_tokens,
            template_used=self.template,
        )

        return full_prompt, record

    @staticmethod
    def _count_tokens(text: str) -> int:
        """粗略 token 计数（word-level 近似，实际应使用 tiktoken）。"""
        return len(text.split())

    def inject_with_budget(
        self,
        base_prompt: str,
        skills: list[dict],
        token_budget: int,
    ) -> tuple[str, InjectionRecord]:
        """在 token budget 约束下注入尽可能多的 Skill。"""
        selected = []
        total = 0
        for skill in skills:
            tok = self._count_tokens(skill["content"])
            if total + tok > token_budget:
                break
            selected.append(skill)
            total += tok
        return self.inject(base_prompt, selected)
