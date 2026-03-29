#!/usr/bin/env python3
"""RQ1 完整执行：信息密度实验。

流程：
1. 选 50 个任务（分层抽样）
2. 为每个任务找到关联的 curated skill 作为知识源
3. 用 gpt-5.2 生成 L1-L5 五级颗粒度 Skill
4. 用 gpt-4.1-mini 跑 50 任务 × 6 条件 × 5 重复 = 1,500 runs
5. 用 gpt-5.2 做 LLM-as-Judge 评估
6. 统计分析 + 图表
"""

import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────
GENERATION_MODEL = "gpt-5.2"       # 生成 Skill 用强模型
AGENT_MODEL = "gpt-5.2"            # Agent 执行用强模型
JUDGE_MODEL = "gpt-5.2"            # 评估用强模型
N_TASKS = 50
N_REPEATS = 5
MAX_CONCURRENT = 15
OUTPUT_DIR = Path("results/rq1")

LEVEL_SPECS = {
    "L1": {"name": "one-liner",           "target_tokens": 50,
            "instruction": "Write a SINGLE sentence hint (~50 tokens). Be specific and actionable. No bullet points."},
    "L2": {"name": "checklist",           "target_tokens": 200,
            "instruction": "Write 3-5 bullet points (~200 tokens). Each bullet: WHAT to do and WHY. Must include ALL information from L1."},
    "L3": {"name": "focused-sop",         "target_tokens": 500,
            "instruction": "Write a focused SOP (~500 tokens) with: Background, Steps (3-5 concrete steps), Verification. Must include ALL from L2."},
    "L4": {"name": "comprehensive-guide", "target_tokens": 1500,
            "instruction": "Write a comprehensive guide (~1500 tokens): Background, Use Cases, Steps (8-12), Edge Cases, Verification. Must include ALL from L3."},
    "L5": {"name": "documentation",       "target_tokens": 3000,
            "instruction": "Write full documentation (~3000 tokens): History, Complete reference, Multiple code examples, Troubleshooting, Related topics. Must include ALL from L4."},
}


# ── Step 1: 任务选择 ─────────────────────────────
def select_tasks() -> list[dict]:
    """分层抽样选 50 个任务。"""
    task_dir = Path("data/tasks/skillsbench")
    all_tasks = []
    for f in sorted(task_dir.glob("*.json")):
        if f.name == "skillsbench_index.json":
            continue
        data = json.loads(f.read_text())
        if not data.get("instruction"):
            continue
        all_tasks.append(data)

    # 按领域分层抽样
    from collections import defaultdict
    import random
    rng = random.Random(42)

    by_domain = defaultdict(list)
    for t in all_tasks:
        by_domain[t["domain"]].append(t)

    selected = []
    n_domains = len(by_domain)
    per_domain = N_TASKS // n_domains
    remainder = N_TASKS % n_domains

    for i, (domain, tasks) in enumerate(sorted(by_domain.items())):
        k = per_domain + (1 if i < remainder else 0)
        k = min(k, len(tasks))
        selected.extend(rng.sample(tasks, k))

    selected = selected[:N_TASKS]
    logger.info(f"Selected {len(selected)} tasks from {n_domains} domains")

    domain_dist = {}
    for t in selected:
        domain_dist[t["domain"]] = domain_dist.get(t["domain"], 0) + 1
    logger.info(f"Domain distribution: {domain_dist}")

    return selected


# ── Step 2: 知识源提取 ────────────────────────────
def find_knowledge_source(task: dict) -> str:
    """找到任务关联的 curated skill 作为知识源。"""
    task_id = task["id"]
    skill_dir = Path("data/skills/human_curated")

    # 找与此任务关联的 skill
    related_skills = []
    for f in skill_dir.glob(f"sb_{task_id}_*.json"):
        data = json.loads(f.read_text())
        related_skills.append(data["content"])

    if related_skills:
        # 合并所有关联 skill 作为知识源（截断避免过长）
        combined = "\n---\n".join(related_skills)
        return combined[:4000]
    else:
        # 没有关联 skill，用 instruction 本身
        return task["instruction"][:2000]


# ── Step 3: 多颗粒度 Skill 生成 ──────────────────
def generate_skills_for_task(client: OpenAI, task: dict, knowledge: str) -> dict[str, dict]:
    """为单个任务生成 L1-L5 Skill。"""
    results = {}
    previous_content = ""

    for level, spec in LEVEL_SPECS.items():
        prompt_parts = [
            f"You are generating a knowledge skill at granularity level {level} ({spec['name']}).",
            f"\n## Source Knowledge\n{knowledge}\n",
            f"\n## Task Context\n{task['instruction'][:500]}\n",
            f"\n## Instructions\n{spec['instruction']}",
            f"\nTarget length: ~{spec['target_tokens']} tokens.",
        ]
        if previous_content:
            prompt_parts.append(
                f"\n## Previous Level Content (MUST be fully contained in your output)\n{previous_content}"
            )
        prompt_parts.append("\nOutput ONLY the skill content, no meta-commentary.")

        prompt = "\n".join(prompt_parts)

        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=spec["target_tokens"] * 3,
            temperature=0.3,
        )
        content = resp.choices[0].message.content.strip()
        token_count = resp.usage.completion_tokens if resp.usage else len(content.split())

        results[level] = {
            "level": level,
            "content": content,
            "token_count": token_count,
            "task_id": task["id"],
        }
        previous_content = content

    return results


# ── Step 4: Agent 执行 ────────────────────────────
AGENT_SYSTEM_PROMPT = "You are a skilled software engineer and problem solver. Solve the given task completely and correctly."


async def run_agent(
    aclient: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    run_id: str,
    task: dict,
    skill_content: str | None,
    repeat: int,
) -> dict:
    """执行单次 agent 调用。"""
    system_prompt = AGENT_SYSTEM_PROMPT
    if skill_content:
        system_prompt += f"\n\n## Relevant Skill\n{skill_content}"

    user_prompt = task["instruction"][:3000]

    async with semaphore:
        # 提取 condition（处理复合 task_id 中的下划线）
        # run_id 格式: rq1_{task_id}_{condition}_r{n}
        parts = run_id.rsplit("_", 2)  # [..., condition, rN]
        condition = parts[-2] if len(parts) >= 3 else "unknown"

        for attempt in range(3):
            try:
                start = time.monotonic()
                resp = await aclient.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0,
                    seed=42 + repeat,
                )
                latency = (time.monotonic() - start) * 1000

                return {
                    "run_id": run_id,
                    "task_id": task["id"],
                    "domain": task["domain"],
                    "difficulty": task["difficulty"],
                    "condition": condition,
                    "repeat": repeat,
                    "output": resp.choices[0].message.content,
                    "tokens_used": resp.usage.total_tokens if resp.usage else 0,
                    "latency_ms": latency,
                    "error": None,
                }
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "429" in str(e)):
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                logger.error(f"Run {run_id} failed after {attempt+1} attempts: {e}")
                return {
                    "run_id": run_id,
                    "task_id": task["id"],
                    "domain": task["domain"],
                    "difficulty": task["difficulty"],
                    "condition": condition,
                    "repeat": repeat,
                    "output": "",
                    "tokens_used": 0,
                    "latency_ms": 0,
                    "error": str(e),
                }


# ── Step 5: LLM-as-Judge 评估 ────────────────────
JUDGE_PROMPT = """You are an expert judge scoring an AI agent's response to a programming/analysis task.

## Task Instruction
{instruction}

## Agent Response
{response}

## Scoring (0-5 scale)
Rate the response on these criteria, then give an overall score:
- Understanding (0-5): Does it correctly identify what the task requires?
- Approach (0-5): Is the solution strategy/algorithm correct?
- Completeness (0-5): Does it address all parts of the task?
- Correctness (0-5): Would the code/logic produce correct results?

Output ONLY a JSON object: {{"understanding": N, "approach": N, "completeness": N, "correctness": N, "overall": N}}"""


async def judge_single(
    aclient: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    task: dict,
    agent_output: str,
) -> float:
    """用 LLM 多维度打分（返回 0-5 的 overall 分数）。"""
    if not agent_output:
        return 0.0

    prompt = JUDGE_PROMPT.format(
        instruction=task["instruction"][:2000],
        response=agent_output[:2000],
    )

    async with semaphore:
        try:
            resp = await aclient.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            # 解析 JSON
            import re
            m = re.search(r'\{[^}]+\}', text)
            if m:
                scores = json.loads(m.group())
                return float(scores.get("overall", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return 0.0


# ── Step 6: 分析 ─────────────────────────────────
def analyze_results(results: list[dict]):
    """统计分析并生成图表。"""
    import numpy as np
    from collections import defaultdict

    # 按条件聚合（使用 score 字段，0-5 分）
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r.get("score", 0))

    print("\n" + "=" * 60)
    print("  RQ1 Results: Information Density vs Agent Performance")
    print("  (Score: 0-5 scale from LLM-as-Judge)")
    print("=" * 60)

    token_map = {"baseline": 0, "L1": 50, "L2": 200, "L3": 500, "L4": 1500, "L5": 3000}
    summary = {}

    for cond in ["baseline", "L1", "L2", "L3", "L4", "L5"]:
        scores = by_condition.get(cond, [])
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        mean = float(np.mean(arr))
        rng = np.random.RandomState(42)
        boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(5000)]
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        n = len(scores)

        summary[cond] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n, "tokens": token_map.get(cond, 0)}

        delta = ""
        if cond != "baseline" and "baseline" in summary:
            d = mean - summary["baseline"]["mean"]
            delta = f"  (Δ={d:+.2f})"
        print(f"  {cond:>10s} (~{token_map.get(cond,0):>5d} tok): score={mean:.3f}/5 [{ci_lo:.3f}, {ci_hi:.3f}] n={n}{delta}")

    # 按难度分组
    print("\n  --- By Difficulty ---")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if not diff_results:
            continue
        by_cond_diff = defaultdict(list)
        for r in diff_results:
            by_cond_diff[r["condition"]].append(r["passed"])
        parts = []
        for cond in ["baseline", "L1", "L2", "L3", "L4", "L5"]:
            scores = by_cond_diff.get(cond, [])
            if scores:
                parts.append(f"{cond}={np.mean(scores):.2f}")
        print(f"  {diff:>8s}: {', '.join(parts)}")

    # 保存汇总
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # 生成图表
    try:
        plot_results(summary)
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    return summary


def plot_results(summary: dict):
    """生成 RQ1 核心图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ["baseline", "L1", "L2", "L3", "L4", "L5"]
    tokens = [summary[c]["tokens"] for c in conditions if c in summary]
    means = [summary[c]["mean"] for c in conditions if c in summary]
    ci_los = [summary[c]["ci_lo"] for c in conditions if c in summary]
    ci_his = [summary[c]["ci_hi"] for c in conditions if c in summary]
    labels = [c for c in conditions if c in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：条形图
    colors = ["#9E9E9E", "#FFC107", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
    bars = ax1.bar(labels, means, color=colors[:len(labels)], edgecolor="white", linewidth=0.5)
    for i, (lo, hi) in enumerate(zip(ci_los, ci_his)):
        ax1.plot([i, i], [lo, hi], color="black", linewidth=1.5)
    ax1.set_ylabel("Score (0-5)")
    ax1.set_xlabel("Skill Granularity Level")
    ax1.set_title("RQ1: Agent Performance by Skill Granularity")
    ax1.set_ylim(0, 5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10)

    # 右图：token 数 vs pass rate（对数 x 轴）
    tok_no_zero = [max(t, 1) for t in tokens]
    ax2.plot(tok_no_zero[1:], means[1:], "o-", color="#2196F3", linewidth=2, markersize=8)
    ax2.fill_between(tok_no_zero[1:], ci_los[1:], ci_his[1:], alpha=0.2, color="#2196F3")
    if "baseline" in summary:
        ax2.axhline(y=summary["baseline"]["mean"], color="#F44336", linestyle="--", label="Baseline (no skill)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Skill Token Count (log scale)")
    ax2.set_ylabel("Score (0-5)")
    ax2.set_title("RQ1: Information Density Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "rq1_density_performance.png"), dpi=200)
    plt.close(fig)
    logger.info(f"Plot saved to {OUTPUT_DIR / 'rq1_density_performance.png'}")


# ── Main ──────────────────────────────────────────
async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    aclient = AsyncOpenAI()

    # ── Step 1: 选任务 ──
    print("\n📌 Step 1: Selecting tasks...")
    tasks = select_tasks()

    # ── Step 2+3: 生成 L1-L5 Skill（增量生成，支持断点续传）──
    skills_cache_path = OUTPUT_DIR / "generated_skills.json"
    all_skills = {}
    if skills_cache_path.exists():
        all_skills = json.loads(skills_cache_path.read_text())
        logger.info(f"Loaded {len(all_skills)} cached skill sets")

    missing = [t for t in tasks if t["id"] not in all_skills]
    if missing:
        print(f"\n📌 Step 2-3: Generating L1-L5 for {len(missing)} remaining tasks with {GENERATION_MODEL}...")
        for i, task in enumerate(missing):
            knowledge = find_knowledge_source(task)
            try:
                skills = generate_skills_for_task(client, task, knowledge)
                all_skills[task["id"]] = skills
                logger.info(
                    f"  [{i+1}/{len(missing)}] {task['id']}: "
                    + ", ".join(f"{l}={s['token_count']}tok" for l, s in skills.items())
                )
            except Exception as e:
                logger.error(f"  [{i+1}/{len(missing)}] {task['id']} FAILED: {e}")
                time.sleep(5)  # 遇错等 5 秒再继续
            # 每个任务后保存
            skills_cache_path.write_text(json.dumps(all_skills, indent=2, ensure_ascii=False))
        logger.info(f"Total skills: {len(all_skills)}/{len(tasks)} tasks")
    else:
        logger.info("All skills already generated")

    # ── Step 4: Agent 执行（支持断点续传）──
    runs_cache_path = OUTPUT_DIR / "agent_runs.json"

    # 加载已有的成功结果
    cached_ok = {}
    if runs_cache_path.exists():
        prev = json.loads(runs_cache_path.read_text())
        for r in prev:
            if not r.get("error"):
                cached_ok[r["run_id"]] = r
        logger.info(f"Loaded {len(cached_ok)} successful cached runs")

    # 只构建「有 skill」的任务的 run（跳过没生成 skill 的任务在 L1-L5 上的无效 run）
    print(f"\n📌 Step 4: Running agent ({AGENT_MODEL})...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pending_runs = []
    task_lookup_local = {}

    for task in tasks:
        if task["id"] not in all_skills:
            continue  # 跳过 Skill 生成失败的任务
        task_skills = all_skills[task["id"]]
        task_lookup_local[task["id"]] = task

        for condition in ["baseline", "L1", "L2", "L3", "L4", "L5"]:
            skill_content = None
            if condition != "baseline":
                skill_content = task_skills.get(condition, {}).get("content")

            for r in range(N_REPEATS):
                run_id = f"rq1_{task['id']}_{condition}_r{r}"
                if run_id in cached_ok:
                    continue  # 跳过已成功的
                pending_runs.append((run_id, task, skill_content, r))

    logger.info(f"Pending: {len(pending_runs)}, Already done: {len(cached_ok)}")

    # 等 API 可用再开始
    if pending_runs:
        logger.info("Checking API availability...")
        for wait_round in range(30):
            try:
                test_resp = await aclient.chat.completions.create(
                    model=AGENT_MODEL, messages=[{"role": "user", "content": "OK"}],
                    max_tokens=5, temperature=0,
                )
                logger.info(f"API OK: {test_resp.choices[0].message.content}")
                break
            except Exception as e:
                logger.warning(f"API not ready ({e}), waiting 60s... ({wait_round+1}/30)")
                await asyncio.sleep(60)
        else:
            logger.error("API unavailable after 30 minutes, aborting")
            sys.exit(1)

    # 分批执行
    new_results = []
    batch_size = 50
    for batch_start in range(0, len(pending_runs), batch_size):
        batch = pending_runs[batch_start:batch_start + batch_size]
        coros = [run_agent(aclient, semaphore, rid, t, sc, r) for rid, t, sc, r in batch]
        batch_results = await asyncio.gather(*coros)
        new_results.extend(batch_results)
        done = min(batch_start + batch_size, len(pending_runs))
        ok = sum(1 for r in new_results if not r.get("error"))
        err = sum(1 for r in new_results if r.get("error"))
        logger.info(f"  Progress: {done}/{len(pending_runs)} pending, OK={ok}, Err={err}")

        # 合并并保存
        all_runs_results = list(cached_ok.values()) + new_results
        runs_cache_path.write_text(json.dumps(all_runs_results, indent=2, ensure_ascii=False))

    all_runs_results = list(cached_ok.values()) + new_results
    runs_cache_path.write_text(json.dumps(all_runs_results, indent=2, ensure_ascii=False))
    ok_total = sum(1 for r in all_runs_results if not r.get("error"))
    logger.info(f"Agent runs done: {len(all_runs_results)} total, {ok_total} OK")

    # ── Step 5: 评估 ──
    # ── Step 5: 评估（只评估有输出且未评估的 run）──
    eval_cache_path = OUTPUT_DIR / "evaluated_results.json"

    # 过滤：只保留成功的 run
    valid_runs = [r for r in all_runs_results if not r.get("error") and r.get("output")]
    logger.info(f"Valid runs for evaluation: {len(valid_runs)}/{len(all_runs_results)}")

    # 检查是否已有评估缓存
    if eval_cache_path.exists():
        cached_eval = json.loads(eval_cache_path.read_text())
        already_judged = {r["run_id"] for r in cached_eval if "passed" in r}
        # 合并已评估的结果
        for r in valid_runs:
            match = next((c for c in cached_eval if c["run_id"] == r["run_id"] and "passed" in c), None)
            if match:
                r["passed"] = match["passed"]
        logger.info(f"Loaded {len(already_judged)} cached evaluations")

    to_eval = [(i, r) for i, r in enumerate(valid_runs) if "passed" not in r]
    logger.info(f"Need to judge: {len(to_eval)} runs")

    if to_eval:
        print(f"\n📌 Step 5: Evaluating {len(to_eval)} runs with {JUDGE_MODEL}...")
        task_lookup = {t["id"]: t for t in tasks}
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        batch_size = 50
        for batch_start in range(0, len(to_eval), batch_size):
            batch = to_eval[batch_start:batch_start + batch_size]
            coros = [judge_single(aclient, semaphore, task_lookup[r["task_id"]], r["output"]) for _, r in batch]
            scores = await asyncio.gather(*coros)
            for (idx, run), sc in zip(batch, scores):
                valid_runs[idx]["score"] = float(sc)
                valid_runs[idx]["passed"] = float(sc) >= 3.0  # 3/5 为 pass 阈值
            done = min(batch_start + batch_size, len(to_eval))
            logger.info(f"  Evaluated: {done}/{len(to_eval)}")
            eval_cache_path.write_text(json.dumps(valid_runs, indent=2, ensure_ascii=False))

    eval_cache_path.write_text(json.dumps(valid_runs, indent=2, ensure_ascii=False))
    all_runs_results = valid_runs  # 用 valid runs 做分析

    # ── Step 6: 分析 ──
    print(f"\n📌 Step 6: Analysis...")
    summary = analyze_results(all_runs_results)

    print(f"\n✓ RQ1 完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
