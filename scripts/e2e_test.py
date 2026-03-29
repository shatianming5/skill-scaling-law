#!/usr/bin/env python3
"""端到端连通性测试。

用 1 个 SkillsBench 任务 + HC Skill 池 + 1 个模型，
验证完整 pipeline: TaskLoader → SkillRetriever → SkillInjector → AgentRunner → Evaluator
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_env():
    """检查 API Key 是否配置。"""
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"ANTHROPIC_API_KEY: {'✓' if has_anthropic else '✗ (未设置)'}")
    print(f"OPENAI_API_KEY:    {'✓' if has_openai else '✗ (未设置)'}")
    return has_anthropic or has_openai


def test_task_loader():
    """测试任务加载。"""
    print("\n=== 1. TaskLoader ===")
    from src.infrastructure.task_loader import TaskLoader

    config = {
        "tasks": {
            "sources": [
                {"name": "skillsbench", "path": "data/tasks/skillsbench/", "count": 88},
            ]
        }
    }
    loader = TaskLoader(config)
    tasks = loader.load_all()
    print(f"  Loaded {len(tasks)} tasks")

    if tasks:
        t = tasks[0]
        print(f"  Sample: id={t.task_id}, domain={t.domain}, difficulty={t.difficulty}")
        print(f"  Instruction: {t.instruction[:100]}...")
    assert len(tasks) > 0, "No tasks loaded!"

    sampled = loader.sample(5, strategy="random", seed=42)
    print(f"  Sampled {len(sampled)} tasks")
    return tasks


def test_skill_pool():
    """测试 Skill 池加载。"""
    print("\n=== 2. SkillPool ===")
    from src.skills.pool import SkillPool

    pool = SkillPool()
    n_hc = pool.load_from_dir("data/skills/human_curated/", "human_curated")
    n_ag = pool.load_from_dir("data/skills/auto_generated/", "auto_generated")
    print(f"  HC: {n_hc}, AG: {n_ag}, Total: {pool.size}")

    sub = pool.sample(10, source="human_curated", seed=42)
    print(f"  Sampled sub-pool: {sub.size} skills")

    mixed = pool.build_mixed(total=50, hc_ratio=0.5, seed=42)
    print(f"  Mixed pool (50/50): {mixed.size} skills")
    return pool


def test_retriever(pool, tasks):
    """测试 BM25 检索。"""
    print("\n=== 3. SkillRetriever ===")
    from src.skills.retriever import SkillRetriever

    retriever = SkillRetriever()
    retriever.index(pool.to_retriever_format())
    print(f"  Indexed {pool.size} skills")

    task = tasks[0]
    results = retriever.retrieve(task.instruction, top_k=3)
    print(f"  Query: '{task.instruction[:60]}...'")
    for r in results:
        print(f"    → {r.skill_id} (score={r.score:.2f}, {len(r.content.split())} tokens)")
    return results


def test_skill_injector(skills):
    """测试 Skill 注入。"""
    print("\n=== 4. SkillInjector ===")
    from src.infrastructure.skill_injector import SkillInjector

    injector = SkillInjector({"injection": {}})
    base_prompt = "You are a helpful assistant."
    skill_dicts = [{"id": r.skill_id, "content": r.content} for r in skills]

    full_prompt, record = injector.inject(base_prompt, skill_dicts)
    print(f"  Injected {len(record.skill_ids)} skills, {record.total_tokens} tokens")
    print(f"  Prompt length: {len(full_prompt)} chars")
    return full_prompt


def test_evaluator():
    """测试评估器。"""
    print("\n=== 5. Evaluator ===")
    from src.infrastructure.evaluator import Evaluator, EvalResult

    evaluator = Evaluator({"evaluation": {}, "statistics": {}})

    # 模拟评估
    r1 = evaluator.evaluate_single("r1", "t1", "The answer is 42", "42")
    r2 = evaluator.evaluate_single("r2", "t1", "I don't know", "42")
    print(f"  'The answer is 42' vs '42': passed={r1.passed}")
    print(f"  'I don't know' vs '42': passed={r2.passed}")

    results = [
        EvalResult("r1", "t1", True, 1.0),
        EvalResult("r2", "t1", False, 0.0),
        EvalResult("r3", "t2", True, 1.0),
        EvalResult("r4", "t2", True, 1.0),
    ]
    metrics = evaluator.aggregate(results, "test_condition")
    print(f"  Aggregated: pass_rate={metrics.pass_rate:.2f}, "
          f"CI=[{metrics.pass_rate_ci_low:.2f}, {metrics.pass_rate_ci_high:.2f}]")


async def test_agent_runner(system_prompt, task):
    """测试 Agent 执行（需要 API Key）。"""
    print("\n=== 6. AgentRunner (API call) ===")
    from src.infrastructure.agent_runner import AgentRunner

    # 检测可用模型
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if has_anthropic:
        model_name = "claude-sonnet"
        config = {
            "agents": {
                "models": [{
                    "name": "claude-sonnet",
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-20250514",
                    "temperature": 0,
                    "max_tokens": 1024,
                }],
                "concurrency": 5,
                "retry": {"max_retries": 2, "backoff_base": 2.0},
            }
        }
    elif has_openai:
        model_name = "gpt-4.1-mini"
        config = {
            "agents": {
                "models": [{
                    "name": "gpt-4.1-mini",
                    "provider": "openai",
                    "model_id": "gpt-4.1-mini",
                    "temperature": 0,
                    "max_tokens": 1024,
                }],
                "concurrency": 5,
                "retry": {"max_retries": 2, "backoff_base": 2.0},
            }
        }
    else:
        print("  ⚠ Skipped: No API key found")
        return None

    runner = AgentRunner(config)
    print(f"  Using model: {model_name}")
    print(f"  Task: {task.task_id}")

    # 截断 instruction 避免过长
    user_prompt = task.instruction[:2000]

    response = await runner.run(
        run_id="e2e_test_001",
        task_id=task.task_id,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    print(f"  ✓ Response: {response.output[:200]}...")
    print(f"  Tokens: {response.tokens_used}, Latency: {response.latency_ms:.0f}ms")
    return response


async def main():
    print("=" * 50)
    print("  端到端连通性测试（真实数据 + 真实 API）")
    print("=" * 50)

    has_api = check_env()
    if not has_api:
        print("ERROR: 需要设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")
        sys.exit(1)

    tasks = test_task_loader()
    pool = test_skill_pool()
    retrieved = test_retriever(pool, tasks)
    system_prompt = test_skill_injector(retrieved)
    test_evaluator()
    await test_agent_runner(system_prompt, tasks[0])

    print("\n" + "=" * 50)
    print("  ✓ 连通性测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
