"""Fixed Token Budget Injection.

Selects skills to inject within an exact token budget, enabling fair
comparison across experimental conditions (e.g. "1 HC skill at 1500 tok"
vs "5 AG skills at 300 tok each").  The last selected skill is truncated
at a token boundary so the total injected count exactly equals the budget.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BudgetInjectionRecord:
    """Immutable record of a budget-constrained injection."""

    budget: int
    actual_injected_tokens: int
    skill_ids: list[str]
    per_skill_tokens: list[int]
    last_skill_truncated: bool

    @property
    def utilisation(self) -> float:
        """Fraction of budget actually used (should be 1.0 unless pool is
        smaller than budget)."""
        return self.actual_injected_tokens / self.budget if self.budget > 0 else 0.0


class TokenBudgetInjector:
    """Select and (if necessary) truncate skills to fit a fixed token budget.

    Parameters
    ----------
    budget : int
        Maximum number of tokens to inject (whitespace-level tokens).
    separator : str
        Text inserted between consecutive skill contents.  Its tokens
        count towards the budget.
    """

    def __init__(self, budget: int, separator: str = "\n---\n"):
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.budget = budget
        self.separator = separator
        self._sep_tokens: list[str] = self._tokenize(separator)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        skills: list[dict],
    ) -> tuple[list[dict], BudgetInjectionRecord]:
        """Choose and possibly truncate skills to exactly fill *budget* tokens.

        Parameters
        ----------
        skills : list[dict]
            Ordered list of candidate skills.  Each dict **must** contain
            ``"id"`` (str) and ``"content"`` (str).  The order determines
            selection priority (typically relevance-ranked).

        Returns
        -------
        selected_skills : list[dict]
            Skills (with the last potentially truncated) that fit within
            the budget.  Each dict mirrors the input format.
        record : BudgetInjectionRecord
            Metadata describing what was injected and the token counts.
        """
        if not skills:
            return [], self._empty_record()

        selected: list[dict] = []
        per_skill_tokens: list[int] = []
        remaining = self.budget
        truncated = False

        for idx, skill in enumerate(skills):
            tokens = self._tokenize(skill["content"])
            # Account for separator cost between skills.
            sep_cost = len(self._sep_tokens) if selected else 0

            available = remaining - sep_cost
            if available <= 0:
                break

            if len(tokens) <= available:
                # Whole skill fits.
                selected.append(skill)
                used = len(tokens) + sep_cost
                per_skill_tokens.append(len(tokens))
                remaining -= used
            else:
                # Truncate this skill to fill the remaining budget exactly.
                truncated_tokens = tokens[:available]
                truncated_content = " ".join(truncated_tokens)
                selected.append({
                    "id": skill["id"],
                    "content": truncated_content,
                })
                per_skill_tokens.append(available)
                remaining -= available + sep_cost
                truncated = True
                break

        actual_injected = self.budget - remaining

        record = BudgetInjectionRecord(
            budget=self.budget,
            actual_injected_tokens=actual_injected,
            skill_ids=[s["id"] for s in selected],
            per_skill_tokens=per_skill_tokens,
            last_skill_truncated=truncated,
        )

        logger.info(
            "BudgetInjector: budget=%d  actual=%d  skills=%d  truncated=%s",
            self.budget,
            actual_injected,
            len(selected),
            truncated,
        )

        return selected, record

    def format(self, skills: list[dict]) -> str:
        """Join selected skill contents with the separator."""
        return self.separator.join(s["content"] for s in skills)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Whitespace tokeniser consistent with the rest of the codebase."""
        return text.split()

    def _empty_record(self) -> BudgetInjectionRecord:
        return BudgetInjectionRecord(
            budget=self.budget,
            actual_injected_tokens=0,
            skill_ids=[],
            per_skill_tokens=[],
            last_skill_truncated=False,
        )
