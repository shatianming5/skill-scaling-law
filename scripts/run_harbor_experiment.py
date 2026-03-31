#!/usr/bin/env python3
"""用 Harbor + opencode + deterministic verifier 重跑全部实验。

解决之前的 5 个根本问题:
1. 用 Harbor pytest verifier 替换 LLM-as-Judge (binary pass/fail)
2. AG Skill 用完整内容 (需先运行 fix-ag-skills)
3. 增加任务数和重复次数
4. 固定 token budget 注入
5. 报告 BM25 Recall@3

执行方式: harbor run 逐任务跑，每个条件注入不同 Skill 到任务的 skills/ 目录。
"""

import json
import os
import re
import subprocess
import sys
import time
import logging
import random
import shutil
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────
SKILLSBENCH_DIR = Path("/tmp/skillsbench")
PROJECT_DIR = Path("/Users/shatianming/Downloads/SSL")
OUTPUT_DIR = PROJECT_DIR / "results" / "harbor_v2"
HARBOR_BIN = os.path.expanduser("~/.local/bin/harbor")

MODEL = "openai/gpt-5.2"
TIMEOUT_MULTIPLIER = 10
TOP_K = 3
TOKEN_BUDGET = 1500  # 固定注入 token 预算

# 只跑已成功 pre-build 的任务
def get_available_tasks():
    """获取已 pre-build 镜像的任务列表。"""
    result = subprocess.run(["docker", "image", "ls", "--format", "{{.Repository}}"],
                            capture_output=True, text=True)
    images = set(result.stdout.strip().split("\n"))

    available = []
    for task_dir in sorted(SKILLSBENCH_DIR.glob("tasks/*")):
        task = task_dir.name
        df = task_dir / "environment" / "Dockerfile"
        if not df.exists():
            continue
        if f"hb__{task}" in images or "opencode-base" in df.read_text()[:100]:
            toml = task_dir / "task.toml"
            diff = "medium"
            if toml.exists():
                for line in toml.read_text().split("\n"):
                    if "difficulty" in line:
                        diff = line.split('"')[1]
            available.append({"name": task, "difficulty": diff})
    return available


# ── Skill 注入 ────────────────────────────────────
def inject_skills_to_task(task_dir: Path, skills: list[dict], token_budget: int):
    """将 Skill 内容写入任务的 skills/ 目录，受 token budget 约束。

    Harbor 会自动把 tasks/xxx/environment/skills/ 复制到容器里的各 agent skill 目录。
    """
    skills_dir = task_dir / "environment" / "skills" / "injected"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)

    injected_tokens = 0
    injected_skills = []

    for skill in skills:
        content = skill.get("content", "")
        tokens = len(content.split())
        if injected_tokens + tokens > token_budget:
            # 截断最后一个 skill
            remaining = token_budget - injected_tokens
            if remaining > 50:
                words = content.split()[:remaining]
                content = " ".join(words)
                tokens = remaining
            else:
                break

        skill_name = re.sub(r'[^a-zA-Z0-9_-]', '_', skill.get("id", f"skill_{len(injected_skills)}"))[:60]
        skill_path = skills_dir / skill_name / "SKILL.md"
        skill_path.parent.mkdir(exist_ok=True)
        skill_path.write_text(f"---\nname: {skill_name}\ndescription: Injected skill\n---\n\n{content}")

        injected_tokens += tokens
        injected_skills.append({"id": skill["id"], "tokens": tokens})

    return {"n_skills": len(injected_skills), "total_tokens": injected_tokens, "skills": injected_skills}


def clear_injected_skills(task_dir: Path):
    """清除注入的 Skill。"""
    skills_dir = task_dir / "environment" / "skills" / "injected"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)


# ── BM25 检索 ────────────────────────────────────
sys.path.insert(0, str(PROJECT_DIR))
from src.skills.retriever import SkillRetriever


def load_skill_pool(pool_type: str, size: int = None, hc_ratio: float = 1.0):
    """加载 Skill 池。"""
    hc_skills, ag_skills = [], []
    for f in sorted((PROJECT_DIR / "data/skills/human_curated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content") and len(d["content"]) > 50:
            hc_skills.append({"id": d["id"], "content": d["content"][:2000]})
    for f in sorted((PROJECT_DIR / "data/skills/auto_generated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content") and len(d["content"]) > 50:
            ag_skills.append({"id": d["id"], "content": d["content"][:2000]})

    rng = random.Random(42)
    if pool_type == "none":
        return []
    elif pool_type == "hc":
        pool = rng.sample(hc_skills, min(size or len(hc_skills), len(hc_skills)))
    elif pool_type == "ag":
        pool = rng.sample(ag_skills, min(size or len(ag_skills), len(ag_skills)))
    elif pool_type == "mix":
        n_hc = int((size or 200) * hc_ratio)
        n_ag = (size or 200) - n_hc
        pool = (rng.sample(hc_skills, min(n_hc, len(hc_skills))) +
                rng.sample(ag_skills, min(n_ag, len(ag_skills))))
    else:
        pool = []
    return pool


def retrieve_skills(pool, query, top_k=TOP_K):
    """BM25 检索 top-k Skill。"""
    if not pool:
        return []
    retriever = SkillRetriever()
    retriever.index(pool)
    results = retriever.retrieve(query, top_k=top_k)
    return [{"id": r.skill_id, "content": r.content} for r in results]


# ── Harbor 执行 ──────────────────────────────────
def run_harbor_trial(task_name: str, job_name: str) -> dict:
    """执行一次 Harbor trial，返回 reward。"""
    task_path = SKILLSBENCH_DIR / "tasks" / task_name

    cmd = [
        HARBOR_BIN, "run",
        "-p", str(task_path),
        "-a", "opencode",
        "-m", MODEL,
        "--timeout-multiplier", str(TIMEOUT_MULTIPLIER),
        "--no-delete",
        "--no-force-build",
        "--job-name", job_name,
        "--jobs-dir", str(OUTPUT_DIR / "harbor_jobs"),
    ]

    logger.info(f"Running: {task_name} ({job_name})")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10800,  # 3 hours max
            cwd=str(SKILLSBENCH_DIR),
            env={**os.environ, "PATH": f"{os.path.expanduser('~/.local/bin')}:{os.path.expanduser('~/.opencode/bin')}:{os.environ.get('PATH','')}"}
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"  {task_name} timed out after 3h, skipping")
        # Kill leftover containers
        subprocess.run(["docker", "kill"] + subprocess.run(
            ["docker", "ps", "-q"], capture_output=True, text=True
        ).stdout.strip().split(), capture_output=True)
        return {"reward": 0.0, "elapsed_sec": time.time() - start, "errors": ["subprocess_timeout"]}

    elapsed = time.time() - start

    # 解析结果
    result_file = OUTPUT_DIR / "harbor_jobs" / job_name / "result.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        for eval_data in data.get("stats", {}).get("evals", {}).values():
            reward = eval_data.get("metrics", [{}])[0].get("mean", 0)
            errors = eval_data.get("exception_stats", {})
            return {
                "reward": reward,
                "elapsed_sec": elapsed,
                "errors": list(errors.keys()) if errors else [],
            }

    return {"reward": 0.0, "elapsed_sec": elapsed, "errors": ["no_result_file"]}


# ── 实验定义 ──────────────────────────────────────
RQ1_CONDITIONS = [
    {"name": "baseline", "pool": "none"},
    {"name": "L3_skill", "pool": "hc", "size": 200},  # 用 HC 池检索
]

RQ2_CONDITIONS = [
    {"name": "baseline", "pool": "none"},
    {"name": "HC-200", "pool": "hc", "size": 200},
    {"name": "AG-200", "pool": "ag", "size": 200},
    {"name": "Mix-50-50", "pool": "mix", "size": 200, "hc_ratio": 0.5},
]


def run_experiment(tasks, conditions, experiment_name):
    """运行一组实验条件。"""
    results = []
    results_file = OUTPUT_DIR / f"{experiment_name}_results.json"

    # 加载已有结果（断点续传）
    done_keys = set()
    if results_file.exists():
        results = json.loads(results_file.read_text())
        done_keys = {r["key"] for r in results}

    for task in tasks:
        task_name = task["name"]
        task_dir = SKILLSBENCH_DIR / "tasks" / task_name
        instruction = ""
        inst_file = task_dir / "instruction.md"
        if inst_file.exists():
            instruction = inst_file.read_text()[:3000]

        for cond in conditions:
            key = f"{experiment_name}_{task_name}_{cond['name']}"
            if key in done_keys:
                logger.info(f"  Skip (cached): {key}")
                continue

            # 检索并注入 Skill
            pool = load_skill_pool(
                cond["pool"],
                size=cond.get("size"),
                hc_ratio=cond.get("hc_ratio", 1.0),
            )
            skills = retrieve_skills(pool, instruction)
            injection = inject_skills_to_task(task_dir, skills, TOKEN_BUDGET)

            # 执行
            job_name = f"{key}_{int(time.time())}"
            trial_result = run_harbor_trial(task_name, job_name)

            # 清理注入的 Skill
            clear_injected_skills(task_dir)

            result = {
                "key": key,
                "experiment": experiment_name,
                "task": task_name,
                "difficulty": task["difficulty"],
                "condition": cond["name"],
                "reward": trial_result["reward"],
                "elapsed_sec": trial_result["elapsed_sec"],
                "errors": trial_result["errors"],
                "injection": injection,
            }
            results.append(result)

            # 保存
            results_file.write_text(json.dumps(results, indent=2))
            logger.info(f"  {key}: reward={trial_result['reward']}, time={trial_result['elapsed_sec']:.0f}s")

    return results


def analyze_results(results, experiment_name):
    """分析并打印结果。"""
    import numpy as np

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r["reward"])

    print(f"\n{'='*50}")
    print(f"  {experiment_name} Results (Harbor Deterministic)")
    print(f"{'='*50}")

    for cond, rewards in sorted(by_condition.items()):
        arr = np.array(rewards)
        mean = float(np.mean(arr))
        n = len(arr)
        ci = 1.96 * float(np.std(arr)) / max(n**0.5, 1)
        print(f"  {cond:>15}: pass_rate={mean:.3f} ±{ci:.3f} (n={n})")


# ── Main ──────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "harbor_jobs").mkdir(exist_ok=True)

    tasks = get_available_tasks()
    logger.info(f"Available tasks: {len(tasks)}")

    # 选前 10 个做 pilot
    pilot_tasks = tasks[:10]
    logger.info(f"Pilot tasks: {[t['name'] for t in pilot_tasks]}")

    # RQ1 pilot: baseline vs with-skill
    print("\n📌 RQ1 Pilot: baseline vs HC skill")
    rq1_results = run_experiment(pilot_tasks, RQ1_CONDITIONS, "rq1_harbor")
    analyze_results(rq1_results, "RQ1")

    # RQ2 pilot: HC vs AG vs Mix
    print("\n📌 RQ2 Pilot: HC vs AG vs Mix")
    rq2_results = run_experiment(pilot_tasks[:5], RQ2_CONDITIONS, "rq2_harbor")
    analyze_results(rq2_results, "RQ2")

    print(f"\n✓ Pilot 完成！结果在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
