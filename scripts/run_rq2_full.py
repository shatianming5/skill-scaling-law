#!/usr/bin/env python3
"""RQ2 完整执行：Quality vs Quantity 实验。

对比高质量少量 HC Skill vs 大量自动生成 AG Skill 对 Agent 性能的影响。
使用 BM25 从不同规模/质量的 Skill 池中检索 top-3 注入 Agent。
"""

import asyncio
import json
import os
import re
import sys
import time
import logging
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────
AGENT_MODEL = "gpt-5.2"        # 弱模型看差异
JUDGE_MODEL = "gpt-5.2"            # 评估用强模型
N_REPEATS = 3                       # 每条件 3 次（RQ2 条件多，减少重复）
MAX_CONCURRENT = 15
TOP_K = 3                           # 检索 top-k
OUTPUT_DIR = Path("results/rq2")

# ── 实验条件 ──────────────────────────────────────
CONDITIONS = [
    {"name": "Baseline",    "pool": "none",   "size": 0},
    {"name": "HC-50",       "pool": "hc",     "size": 50},
    {"name": "HC-100",      "pool": "hc",     "size": 100},
    {"name": "HC-200",      "pool": "hc",     "size": 200},
    {"name": "AG-100",      "pool": "ag",     "size": 100},
    {"name": "AG-500",      "pool": "ag",     "size": 500},
    {"name": "AG-2000",     "pool": "ag",     "size": 2000},
    {"name": "Mix-70AG30HC","pool": "mix",    "size": 500, "hc_ratio": 0.3},
    {"name": "Mix-50AG50HC","pool": "mix",    "size": 500, "hc_ratio": 0.5},
]

AGENT_SYSTEM_PROMPT = "You are a skilled software engineer and problem solver. Solve the given task completely and correctly."

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


# ── BM25 检索 ─────────────────────────────────────
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = []
        self.doc_freqs = Counter()
        self.avg_dl = 0

    def index(self, docs):
        self.corpus = []
        self.doc_freqs = Counter()
        for d in docs:
            tokens = d["content"].lower().split()
            self.corpus.append({"id": d["id"], "content": d["content"], "tokens": tokens})
            for t in set(tokens):
                self.doc_freqs[t] += 1
        n = len(self.corpus)
        self.avg_dl = sum(len(d["tokens"]) for d in self.corpus) / max(n, 1)

    def retrieve(self, query, top_k=3):
        qt = query.lower().split()
        scores = []
        n = len(self.corpus)
        for doc in self.corpus:
            tf_map = Counter(doc["tokens"])
            dl = len(doc["tokens"])
            score = 0
            for t in qt:
                if t not in tf_map: continue
                tf = tf_map[t]
                df = self.doc_freqs.get(t, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            scores.append((doc, score))
        scores.sort(key=lambda x: -x[1])
        return [{"id": d["id"], "content": d["content"], "score": s} for d, s in scores[:top_k]]


# ── 数据加载 ──────────────────────────────────────
def load_tasks():
    task_dir = Path("data/tasks/skillsbench")
    tasks = []
    for f in sorted(task_dir.glob("*.json")):
        if f.name == "skillsbench_index.json": continue
        d = json.loads(f.read_text())
        if d.get("instruction"):
            tasks.append(d)
    # 分层抽样 30 个
    rng = random.Random(42)
    from collections import defaultdict
    by_domain = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"]].append(t)
    selected = []
    per = 30 // len(by_domain)
    rem = 30 % len(by_domain)
    for i, (dom, ts) in enumerate(sorted(by_domain.items())):
        k = min(per + (1 if i < rem else 0), len(ts))
        selected.extend(rng.sample(ts, k))
    logger.info(f"Selected {len(selected)} tasks")
    return selected[:30]


def load_skill_pools():
    hc_skills = []
    for f in sorted(Path("data/skills/human_curated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content"):
            hc_skills.append({"id": d["id"], "content": d["content"][:2000]})
    ag_skills = []
    for f in sorted(Path("data/skills/auto_generated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content"):
            ag_skills.append({"id": d["id"], "content": d["content"][:2000]})
    logger.info(f"Loaded HC={len(hc_skills)}, AG={len(ag_skills)} skills")
    return hc_skills, ag_skills


def build_pool(hc_skills, ag_skills, cond):
    rng = random.Random(42)
    pool_type = cond["pool"]
    size = cond["size"]
    if pool_type == "none": return []
    if pool_type == "hc":
        return rng.sample(hc_skills, min(size, len(hc_skills)))
    if pool_type == "ag":
        return rng.sample(ag_skills, min(size, len(ag_skills)))
    if pool_type == "mix":
        hc_ratio = cond.get("hc_ratio", 0.5)
        n_hc = int(size * hc_ratio)
        n_ag = size - n_hc
        return (rng.sample(hc_skills, min(n_hc, len(hc_skills))) +
                rng.sample(ag_skills, min(n_ag, len(ag_skills))))
    return []


# ── Agent 执行 ────────────────────────────────────
async def run_agent(aclient, sem, run_id, task, skills, repeat):
    system_prompt = AGENT_SYSTEM_PROMPT
    if skills:
        skill_text = "\n---\n".join(s["content"][:1000] for s in skills[:TOP_K])
        system_prompt += f"\n\n## Relevant Skills\n{skill_text}"

    async with sem:
        for attempt in range(3):
            try:
                start = time.monotonic()
                resp = await aclient.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task["instruction"][:3000]},
                    ],
                    max_tokens=2048, temperature=0, seed=42 + repeat,
                )
                return {
                    "run_id": run_id, "task_id": task["id"],
                    "domain": task["domain"], "difficulty": task["difficulty"],
                    "condition": run_id.rsplit("_", 1)[0].split("_", 1)[1].rsplit("_r", 1)[0],
                    "repeat": repeat,
                    "output": resp.choices[0].message.content,
                    "tokens_used": resp.usage.total_tokens if resp.usage else 0,
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "error": None,
                }
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "429" in str(e)):
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                return {
                    "run_id": run_id, "task_id": task["id"],
                    "domain": task["domain"], "difficulty": task["difficulty"],
                    "condition": run_id.rsplit("_", 1)[0].split("_", 1)[1].rsplit("_r", 1)[0],
                    "repeat": repeat, "output": "", "tokens_used": 0,
                    "latency_ms": 0, "error": str(e),
                }


# ── Judge ─────────────────────────────────────────
async def judge_single(aclient, sem, task, output):
    if not output: return 0.0
    prompt = JUDGE_PROMPT.format(instruction=task["instruction"][:2000], response=output[:2000])
    async with sem:
        try:
            resp = await aclient.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=100, temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\{[^}]+\}', text)
            if m:
                return float(json.loads(m.group()).get("overall", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return 0.0


# ── 分析 ──────────────────────────────────────────
def analyze(results):
    import numpy as np
    from collections import defaultdict

    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r.get("score", 0))

    print("\n" + "=" * 60)
    print("  RQ2 Results: Quality vs Quantity")
    print("=" * 60)

    summary = {}
    cond_order = [c["name"] for c in CONDITIONS]
    for cond in cond_order:
        scores = by_cond.get(cond, [])
        if not scores: continue
        arr = np.array(scores, dtype=float)
        mean = float(np.mean(arr))
        rng = np.random.RandomState(42)
        boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(5000)]
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        summary[cond] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(scores)}
        bl = summary.get("Baseline", {}).get("mean", 0)
        delta = f"  Δ={mean - bl:+.2f}" if cond != "Baseline" else ""
        print(f"  {cond:>15}: {mean:.2f}/5 [{ci_lo:.2f}, {ci_hi:.2f}] n={len(scores)}{delta}")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_rq2(summary)
    return summary


def plot_rq2(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cond_order = [c["name"] for c in CONDITIONS]
    labels, means, ci_los, ci_his = [], [], [], []
    for c in cond_order:
        if c in summary:
            labels.append(c)
            means.append(summary[c]["mean"])
            ci_los.append(summary[c]["ci_lo"])
            ci_his.append(summary[c]["ci_hi"])

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = []
    for l in labels:
        if "HC" in l: colors.append("#4CAF50")
        elif "AG" in l: colors.append("#FF9800")
        elif "Mix" in l: colors.append("#2196F3")
        else: colors.append("#9E9E9E")

    bars = ax.bar(range(len(labels)), means, color=colors, edgecolor="white")
    for i, (lo, hi) in enumerate(zip(ci_los, ci_his)):
        ax.plot([i, i], [lo, hi], color="black", linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score (0-5)")
    ax.set_title("RQ2: Quality vs Quantity — HC (green) vs AG (orange) vs Mix (blue)")
    ax.set_ylim(0, 5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "rq2_quality_quantity.png"), dpi=200)
    plt.close(fig)
    logger.info(f"Plot saved: {OUTPUT_DIR / 'rq2_quality_quantity.png'}")


# ── Main ──────────────────────────────────────────
async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aclient = AsyncOpenAI()

    # 1. 加载数据
    tasks = load_tasks()
    hc_skills, ag_skills = load_skill_pools()

    # 2. 对每个条件：构建 pool → BM25 检索 → agent run
    runs_cache = OUTPUT_DIR / "agent_runs.json"
    cached_ok = {}
    if runs_cache.exists():
        for r in json.loads(runs_cache.read_text()):
            if not r.get("error"):
                cached_ok[r["run_id"]] = r
        logger.info(f"Loaded {len(cached_ok)} cached OK runs")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    pending = []

    for cond in CONDITIONS:
        pool = build_pool(hc_skills, ag_skills, cond)
        bm25 = None
        if pool:
            bm25 = BM25()
            bm25.index(pool)

        for task in tasks:
            retrieved = bm25.retrieve(task["instruction"], TOP_K) if bm25 else []
            for r in range(N_REPEATS):
                run_id = f"rq2_{cond['name']}_{task['id']}_r{r}"
                if run_id in cached_ok:
                    continue
                pending.append((run_id, task, retrieved, r))

    total_runs = len(CONDITIONS) * len(tasks) * N_REPEATS
    logger.info(f"Total: {total_runs}, Cached: {len(cached_ok)}, Pending: {len(pending)}")

    if pending:
        # 检查 API
        for w in range(30):
            try:
                await aclient.chat.completions.create(
                    model=AGENT_MODEL, messages=[{"role":"user","content":"OK"}],
                    max_tokens=5, temperature=0)
                break
            except:
                logger.warning(f"API not ready, waiting... ({w+1}/30)")
                await asyncio.sleep(60)

        new_results = []
        batch_size = 50
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i+batch_size]
            coros = [run_agent(aclient, sem, rid, t, sk, r) for rid, t, sk, r in batch]
            results = await asyncio.gather(*coros)
            new_results.extend(results)
            ok = sum(1 for r in new_results if not r.get("error"))
            logger.info(f"  Agent progress: {min(i+batch_size, len(pending))}/{len(pending)}, OK={ok}")
            all_so_far = list(cached_ok.values()) + new_results
            runs_cache.write_text(json.dumps(all_so_far, indent=2, ensure_ascii=False))
    else:
        new_results = []

    all_runs = list(cached_ok.values()) + new_results
    runs_cache.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False))
    valid = [r for r in all_runs if not r.get("error") and r.get("output")]
    logger.info(f"Agent done: {len(valid)} valid runs")

    # 3. Judge 评估
    eval_cache = OUTPUT_DIR / "evaluated_results.json"
    if eval_cache.exists():
        cached_eval = {r["run_id"]: r for r in json.loads(eval_cache.read_text()) if "score" in r}
        for r in valid:
            if r["run_id"] in cached_eval:
                r["score"] = cached_eval[r["run_id"]]["score"]
                r["passed"] = cached_eval[r["run_id"]].get("passed", False)

    to_eval = [(i, r) for i, r in enumerate(valid) if "score" not in r]
    logger.info(f"Judge: {len(to_eval)} to evaluate")

    if to_eval:
        task_lookup = {t["id"]: t for t in tasks}
        batch_size = 50
        for i in range(0, len(to_eval), batch_size):
            batch = to_eval[i:i+batch_size]
            coros = [judge_single(aclient, sem, task_lookup[r["task_id"]], r["output"]) for _, r in batch]
            scores = await asyncio.gather(*coros)
            for (idx, r), sc in zip(batch, scores):
                valid[idx]["score"] = float(sc)
                valid[idx]["passed"] = float(sc) >= 3.0
            logger.info(f"  Judge: {min(i+batch_size, len(to_eval))}/{len(to_eval)}")
            eval_cache.write_text(json.dumps(valid, indent=2, ensure_ascii=False))

    eval_cache.write_text(json.dumps(valid, indent=2, ensure_ascii=False))

    # 4. 分析
    print(f"\n📌 Analysis...")
    analyze(valid)
    print(f"\n✓ RQ2 完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
