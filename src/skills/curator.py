"""Skill 质量标注管理。

管理人工标注流程：分配任务、收集标注结果、计算 inter-annotator agreement。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANNOTATION_DIMENSIONS = ["correctness", "specificity", "actionability", "completeness"]


@dataclass
class Annotation:
    """单条标注记录。"""
    skill_id: str
    annotator_id: str
    scores: dict[str, int]  # dimension -> score (1-5)
    notes: str = ""


@dataclass
class SkillQuality:
    """Skill 质量汇总。"""
    skill_id: str
    mean_scores: dict[str, float]
    overall_mean: float
    n_annotators: int
    passes_threshold: bool


class SkillCurator:
    """Skill 质量标注与筛选管理。"""

    def __init__(self, config: dict):
        self.config = config
        pool_cfg = config.get("skill_pools", {}).get("human_curated", {})
        ann_cfg = pool_cfg.get("annotation", {})
        self.dimensions = ann_cfg.get("dimensions", ANNOTATION_DIMENSIONS)
        self.scale = ann_cfg.get("scale", [1, 2, 3, 4, 5])
        self.min_annotators = ann_cfg.get("min_annotators", 3)
        self.quality_threshold = pool_cfg.get("quality_threshold", 4.0)
        self.agreement_threshold = ann_cfg.get("agreement_threshold", 0.7)
        self.annotations: list[Annotation] = []

    def add_annotation(self, annotation: Annotation):
        """添加一条标注记录。"""
        for dim in self.dimensions:
            if dim not in annotation.scores:
                raise ValueError(f"Missing dimension: {dim}")
            if annotation.scores[dim] not in self.scale:
                raise ValueError(
                    f"Score {annotation.scores[dim]} not in scale {self.scale}"
                )
        self.annotations.append(annotation)

    def load_annotations(self, path: str):
        """从 JSON 文件加载标注结果。"""
        data = json.loads(Path(path).read_text())
        for item in data:
            self.add_annotation(Annotation(**item))

    def evaluate_skill(self, skill_id: str) -> SkillQuality:
        """评估单条 Skill 的质量。"""
        skill_anns = [a for a in self.annotations if a.skill_id == skill_id]

        if len(skill_anns) < self.min_annotators:
            logger.warning(
                f"Skill {skill_id}: only {len(skill_anns)} annotators "
                f"(need {self.min_annotators})"
            )

        mean_scores = {}
        for dim in self.dimensions:
            scores = [a.scores[dim] for a in skill_anns]
            mean_scores[dim] = float(np.mean(scores)) if scores else 0.0

        overall = float(np.mean(list(mean_scores.values())))

        return SkillQuality(
            skill_id=skill_id,
            mean_scores=mean_scores,
            overall_mean=overall,
            n_annotators=len(skill_anns),
            passes_threshold=overall >= self.quality_threshold,
        )

    def filter_pool(self, skill_ids: list[str]) -> list[str]:
        """筛选通过质量阈值的 Skill。"""
        passed = []
        for sid in skill_ids:
            quality = self.evaluate_skill(sid)
            if quality.passes_threshold:
                passed.append(sid)
        logger.info(f"Quality filter: {len(passed)}/{len(skill_ids)} passed")
        return passed

    def compute_agreement(self) -> float:
        """计算 Krippendorff's alpha（简化版 ordinal）。"""
        skill_ids = list(set(a.skill_id for a in self.annotations))
        annotator_ids = list(set(a.annotator_id for a in self.annotations))

        if len(annotator_ids) < 2:
            return 1.0

        # 构建 annotator × item 矩阵（每个维度分开计算再取平均）
        alphas = []
        for dim in self.dimensions:
            matrix = {}
            for ann in self.annotations:
                key = (ann.annotator_id, ann.skill_id)
                matrix[key] = ann.scores.get(dim)

            alpha = self._krippendorff_alpha_simple(
                matrix, annotator_ids, skill_ids
            )
            alphas.append(alpha)

        return float(np.mean(alphas))

    @staticmethod
    def _krippendorff_alpha_simple(
        matrix: dict,
        annotators: list[str],
        items: list[str],
    ) -> float:
        """简化版 Krippendorff's alpha 计算。"""
        pairs_observed = []
        pairs_expected_vals = []

        for item in items:
            values = []
            for ann in annotators:
                val = matrix.get((ann, item))
                if val is not None:
                    values.append(val)
            if len(values) < 2:
                continue
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    pairs_observed.append((values[i] - values[j]) ** 2)
            pairs_expected_vals.extend(values)

        if not pairs_observed or len(pairs_expected_vals) < 2:
            return 0.0

        do = np.mean(pairs_observed)

        all_vals = pairs_expected_vals
        de_pairs = []
        for i in range(len(all_vals)):
            for j in range(i + 1, len(all_vals)):
                de_pairs.append((all_vals[i] - all_vals[j]) ** 2)
        de = np.mean(de_pairs) if de_pairs else 1.0

        if de == 0:
            return 1.0
        return float(1.0 - do / de)
