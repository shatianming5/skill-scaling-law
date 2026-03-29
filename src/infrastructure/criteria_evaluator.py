"""标准化 rubric 评估器。

从 SkillsBench 的 test_outputs.py 中提取测试标准，
让 LLM Judge 逐条评判 PASS/FAIL，而非打通用分数。
比通用 0-5 分有更高的区分度。
"""

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_criteria(test_file_path: str) -> list[dict]:
    """从 test_outputs.py 提取测试标准。

    Returns:
        [{"name": "test_xxx", "description": "...", "type": "functional|structural|quality"}]
    """
    content = Path(test_file_path).read_text()
    criteria = []

    # 提取 test_ 函数 + docstring
    for m in re.finditer(
        r'def (test_\w+)\([^)]*\):\s*\n\s*"""(.*?)"""',
        content, re.DOTALL
    ):
        name = m.group(1)
        doc = m.group(2).strip().split("\n")[0]  # 只取第一行
        criteria.append({
            "name": name,
            "description": doc,
            "type": _classify_criterion(name, doc),
        })

    # 提取 assert 语句描述（作为补充）
    for m in re.finditer(r'assert\s+(.+?),\s*f?"([^"]+)"', content):
        condition = m.group(1).strip()[:60]
        msg = m.group(2).strip()[:80]
        # 去重：如果和已有 criteria 描述重复就跳过
        if not any(msg[:30] in c["description"] for c in criteria):
            criteria.append({
                "name": f"assert_{len(criteria)}",
                "description": msg,
                "type": "functional",
            })

    return criteria


def _classify_criterion(name: str, description: str) -> str:
    """分类测试标准类型。"""
    text = (name + " " + description).lower()
    if any(kw in text for kw in ["file", "exist", "output", "csv", "json", "yaml"]):
        return "structural"
    if any(kw in text for kw in ["correct", "score", "accuracy", "match", "equal"]):
        return "functional"
    return "quality"


CRITERIA_JUDGE_PROMPT = """You are evaluating an AI agent's response against SPECIFIC test criteria from a benchmark task.

## Task Instruction
{instruction}

## Agent Response
{response}

## Test Criteria to Check
{criteria_list}

For EACH criterion, determine if the agent's response would PASS or FAIL that specific test.
Consider:
- Does the response describe/implement what the criterion checks?
- Would the described approach produce output that satisfies the assertion?
- Be strict: if the response doesn't clearly address a criterion, mark it FAIL.

Output a JSON array of objects:
[{{"name": "criterion_name", "passed": true/false, "reason": "brief reason"}}]

Output ONLY the JSON array, no other text."""


def build_criteria_prompt(
    instruction: str,
    response: str,
    criteria: list[dict],
) -> str:
    """构建逐条标准的 Judge prompt。"""
    criteria_text = "\n".join(
        f"- {c['name']}: {c['description']}"
        for c in criteria
    )
    return CRITERIA_JUDGE_PROMPT.format(
        instruction=instruction[:2000],
        response=response[:2000],
        criteria_list=criteria_text,
    )


def parse_criteria_verdict(judge_output: str, n_criteria: int) -> list[dict]:
    """解析 Judge 的逐条评判结果。"""
    try:
        m = re.search(r'\[.*\]', judge_output, re.DOTALL)
        if m:
            verdicts = json.loads(m.group())
            return verdicts
    except (json.JSONDecodeError, AttributeError):
        pass

    # 解析失败，全部标 FAIL
    return [{"name": f"criterion_{i}", "passed": False, "reason": "parse error"}
            for i in range(n_criteria)]


def compute_criteria_score(verdicts: list[dict]) -> dict:
    """从逐条评判计算综合分数。

    Returns:
        {
            "pass_rate": 0.0-1.0 (通过标准的比例),
            "n_passed": int,
            "n_total": int,
            "verdicts": list[dict],
        }
    """
    if not verdicts:
        return {"pass_rate": 0.0, "n_passed": 0, "n_total": 0, "verdicts": []}

    n_passed = sum(1 for v in verdicts if v.get("passed", False))
    n_total = len(verdicts)
    return {
        "pass_rate": n_passed / n_total,
        "n_passed": n_passed,
        "n_total": n_total,
        "verdicts": verdicts,
    }


def load_all_criteria(test_dir: str = "data/tasks/skillsbench_tests") -> dict[str, list[dict]]:
    """加载所有任务的测试标准。

    Returns:
        {"task_name": [criteria_list]}
    """
    all_criteria = {}
    test_path = Path(test_dir)
    for f in sorted(test_path.glob("*_test_outputs.py")):
        task_name = f.stem.replace("_test_outputs", "")
        criteria = extract_criteria(str(f))
        if criteria:
            all_criteria[task_name] = criteria
            logger.info(f"  {task_name}: {len(criteria)} criteria")
    logger.info(f"Loaded criteria for {len(all_criteria)} tasks")
    return all_criteria
