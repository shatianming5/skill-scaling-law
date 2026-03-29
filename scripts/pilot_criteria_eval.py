#!/usr/bin/env python3
"""Pilot: 用 criteria-based evaluator 重新评估已有的 RQ1 agent 输出。

不重跑 agent，直接用已缓存的 agent 输出，换 criteria judge 重新评分。
对比新旧评估结果的区分度。
"""

import asyncio
import json
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from src.infrastructure.criteria_evaluator import (
    load_all_criteria, build_criteria_prompt, parse_criteria_verdict, compute_criteria_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5.2"
MAX_CONCURRENT = 10


async def judge_with_criteria(aclient, sem, instruction, response, criteria):
    """用标准化 rubric 评估。"""
    if not response or not criteria:
        return compute_criteria_score([])

    prompt = build_criteria_prompt(instruction, response, criteria)

    async with sem:
        try:
            resp = await aclient.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            verdicts = parse_criteria_verdict(text, len(criteria))
            return compute_criteria_score(verdicts)
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return compute_criteria_score([])


async def main():
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # 加载任务标准
    all_criteria = load_all_criteria()

    # 加载 RQ1 strong (gpt-5.2) 的 agent 输出
    rq1_runs = json.loads(Path("results/rq1/agent_runs.json").read_text())
    rq1_tasks = json.loads(Path("data/tasks/skillsbench_index.json").read_text())

    # 加载任务 instruction
    task_instructions = {}
    for f in Path("data/tasks/skillsbench").glob("*.json"):
        if f.name == "skillsbench_index.json": continue
        d = json.loads(f.read_text())
        task_instructions[d["id"]] = d.get("instruction", "")

    # 筛选有 criteria 的任务 + 有输出的 run
    valid_runs = [
        r for r in rq1_runs
        if not r.get("error")
        and r.get("output")
        and r["task_id"] in all_criteria
        and r["task_id"] in task_instructions
    ]

    # 取 5 个任务做 pilot
    pilot_tasks = sorted(set(r["task_id"] for r in valid_runs if r["task_id"] in all_criteria))[:5]
    pilot_runs = [r for r in valid_runs if r["task_id"] in pilot_tasks]

    logger.info(f"Pilot: {len(pilot_tasks)} tasks, {len(pilot_runs)} runs")
    for t in pilot_tasks:
        n_criteria = len(all_criteria[t])
        n_runs = sum(1 for r in pilot_runs if r["task_id"] == t)
        logger.info(f"  {t}: {n_criteria} criteria, {n_runs} runs")

    # 评估
    results = []
    for i, run in enumerate(pilot_runs):
        criteria = all_criteria[run["task_id"]]
        instruction = task_instructions[run["task_id"]]
        score = await judge_with_criteria(aclient, sem, instruction, run["output"], criteria)

        results.append({
            "run_id": run["run_id"],
            "task_id": run["task_id"],
            "condition": run.get("condition", "unknown"),
            "criteria_pass_rate": score["pass_rate"],
            "n_passed": score["n_passed"],
            "n_total": score["n_total"],
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated: {i+1}/{len(pilot_runs)}")

    # 分析
    print("\n" + "=" * 60)
    print("  Pilot: Criteria-Based Evaluation Results")
    print("=" * 60)

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r["criteria_pass_rate"])

    import numpy as np
    for cond in ["baseline", "L1", "L2", "L3", "L4", "L5"]:
        scores = by_condition.get(cond, [])
        if not scores: continue
        arr = np.array(scores)
        print(f"  {cond:>10}: criteria_pass_rate={np.mean(arr):.3f} ± {np.std(arr):.3f}  "
              f"(range: {np.min(arr):.2f}-{np.max(arr):.2f}, n={len(arr)})")

    # 对比旧评估
    old_eval = json.loads(Path("results/rq1/evaluated_results.json").read_text())
    old_by_cond = defaultdict(list)
    for r in old_eval:
        if r["task_id"] in pilot_tasks:
            old_by_cond[r.get("condition", "unknown")].append(r.get("score", 0))

    print("\n  --- 对比旧评估 (generic 0-5) ---")
    for cond in ["baseline", "L1", "L2", "L3", "L4", "L5"]:
        old = old_by_cond.get(cond, [])
        new = by_condition.get(cond, [])
        if old and new:
            print(f"  {cond:>10}: old={np.mean(old):.2f}/5  new_criteria={np.mean(new):.3f}  "
                  f"old_std={np.std(old):.2f}  new_std={np.std(new):.3f}")

    # 保存
    out = Path("results/rq1/pilot_criteria_eval.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
