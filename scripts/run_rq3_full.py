#!/usr/bin/env python3
"""RQ3 完整执行：Scaling Law 实验。

拟合 Agent 性能随 Skill 库规模变化的数学模型。
包含：数量 Scaling + 质量 Scaling + 曲线拟合。
"""

import asyncio
import json
import math
import os
import re
import sys
import time
import logging
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────
AGENT_MODEL = "gpt-4o-mini"        # 弱模型看 scaling 效果
JUDGE_MODEL = "gpt-5.2"
N_REPEATS = 3
MAX_CONCURRENT = 15
TOP_K = 3
OUTPUT_DIR = Path("results/rq3")

AGENT_SYSTEM_PROMPT = "You are a skilled software engineer and problem solver. Solve the given task completely and correctly."

JUDGE_PROMPT = """You are an expert judge scoring an AI agent's response to a programming/analysis task.

## Task Instruction
{instruction}

## Agent Response
{response}

## Scoring (0-5 scale)
- Understanding (0-5): Does it correctly identify what the task requires?
- Approach (0-5): Is the solution strategy/algorithm correct?
- Completeness (0-5): Does it address all parts of the task?
- Correctness (0-5): Would the code/logic produce correct results?

Output ONLY a JSON object: {{"understanding": N, "approach": N, "completeness": N, "correctness": N, "overall": N}}"""

# ── 数量 Scaling 梯度 ────────────────────────────
QUANTITY_GRADIENT = [0, 10, 25, 50, 100, 200, 500, 1000, 2000]

# ── 质量 Scaling 条件 ────────────────────────────
QUALITY_CONDITIONS = [
    {"name": "Q100", "hc_ratio": 1.0},
    {"name": "Q75",  "hc_ratio": 0.75},
    {"name": "Q50",  "hc_ratio": 0.50},
    {"name": "Q25",  "hc_ratio": 0.25},
    {"name": "Q0",   "hc_ratio": 0.0},
]
QUALITY_FIXED_SIZE = 200


# ── BM25 ──────────────────────────────────────────
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus, self.doc_freqs, self.avg_dl = [], Counter(), 0

    def index(self, docs):
        self.corpus, self.doc_freqs = [], Counter()
        for d in docs:
            tokens = d["content"].lower().split()
            self.corpus.append({"id": d["id"], "content": d["content"], "tokens": tokens})
            for t in set(tokens): self.doc_freqs[t] += 1
        n = len(self.corpus)
        self.avg_dl = sum(len(d["tokens"]) for d in self.corpus) / max(n, 1)

    def retrieve(self, query, top_k=3):
        qt = query.lower().split()
        scores = []
        n = len(self.corpus)
        for doc in self.corpus:
            tf = Counter(doc["tokens"]); dl = len(doc["tokens"]); s = 0
            for t in qt:
                if t not in tf: continue
                idf = math.log((n - self.doc_freqs.get(t,0) + 0.5) / (self.doc_freqs.get(t,0) + 0.5) + 1)
                s += idf * (tf[t] * (self.k1+1)) / (tf[t] + self.k1*(1-self.b+self.b*dl/self.avg_dl))
            scores.append((doc, s))
        scores.sort(key=lambda x: -x[1])
        return [{"id": d["id"], "content": d["content"]} for d, _ in scores[:top_k]]


# ── 数据加载 ──────────────────────────────────────
def load_tasks():
    tasks = []
    for f in sorted(Path("data/tasks/skillsbench").glob("*.json")):
        if f.name == "skillsbench_index.json": continue
        d = json.loads(f.read_text())
        if d.get("instruction"): tasks.append(d)
    rng = random.Random(42)
    rng.shuffle(tasks)
    selected = tasks[:30]
    logger.info(f"Selected {len(selected)} tasks")
    return selected


def load_skills():
    hc, ag = [], []
    for f in sorted(Path("data/skills/human_curated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content"): hc.append({"id": d["id"], "content": d["content"][:2000]})
    for f in sorted(Path("data/skills/auto_generated").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("content"): ag.append({"id": d["id"], "content": d["content"][:2000]})
    logger.info(f"HC={len(hc)}, AG={len(ag)}")
    return hc, ag


# ── Agent + Judge ─────────────────────────────────
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
                    messages=[{"role":"system","content":system_prompt},
                              {"role":"user","content":task["instruction"][:3000]}],
                    max_tokens=2048, temperature=0, seed=42+repeat)
                return {"run_id": run_id, "task_id": task["id"],
                        "output": resp.choices[0].message.content,
                        "tokens_used": resp.usage.total_tokens if resp.usage else 0,
                        "latency_ms": (time.monotonic()-start)*1000, "error": None}
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "429" in str(e)):
                    await asyncio.sleep(10*(attempt+1)); continue
                return {"run_id": run_id, "task_id": task["id"],
                        "output": "", "tokens_used": 0, "latency_ms": 0, "error": str(e)}


async def judge_single(aclient, sem, task, output):
    if not output: return 0.0
    prompt = JUDGE_PROMPT.format(instruction=task["instruction"][:2000], response=output[:2000])
    async with sem:
        try:
            resp = await aclient.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role":"user","content":prompt}],
                max_tokens=100, temperature=0)
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\{[^}]+\}', text)
            if m: return float(json.loads(m.group()).get("overall", 0))
            return 0.0
        except: return 0.0


# ── 实验执行 ──────────────────────────────────────
async def run_experiment(aclient, sem, tasks, task_lookup, all_runs_info):
    """执行 agent runs + judge，返回带 score 的结果列表。"""
    # Agent runs
    logger.info(f"Running {len(all_runs_info)} agent calls...")
    results = []
    batch_size = 50
    for i in range(0, len(all_runs_info), batch_size):
        batch = all_runs_info[i:i+batch_size]
        coros = [run_agent(aclient, sem, r["run_id"], r["task"], r["skills"], r["repeat"]) for r in batch]
        batch_res = await asyncio.gather(*coros)
        results.extend(batch_res)
        ok = sum(1 for r in results if not r.get("error"))
        logger.info(f"  Agent: {min(i+batch_size, len(all_runs_info))}/{len(all_runs_info)}, OK={ok}")

    # Judge
    valid = [r for r in results if not r.get("error") and r.get("output")]
    logger.info(f"Judging {len(valid)} runs...")
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i+batch_size]
        coros = [judge_single(aclient, sem, task_lookup[r["task_id"]], r["output"]) for r in batch]
        scores = await asyncio.gather(*coros)
        for r, sc in zip(batch, scores):
            r["score"] = float(sc)
        logger.info(f"  Judge: {min(i+batch_size, len(valid))}/{len(valid)}")

    # 给 error 的 0 分
    for r in results:
        if "score" not in r: r["score"] = 0.0

    return results


# ── 曲线拟合 ──────────────────────────────────────
def fit_scaling_laws(x_data, y_data):
    """拟合 power law, log, sigmoid。"""
    import numpy as np
    from scipy.optimize import curve_fit

    x, y = np.array(x_data, dtype=float), np.array(y_data, dtype=float)
    fits = {}

    # Power law: P = a * N^alpha + b
    try:
        def power_law(n, a, alpha, b): return a * np.power(np.maximum(n, 1e-10), alpha) + b
        popt, pcov = curve_fit(power_law, x, y, p0=[0.1, 0.3, 0.5], bounds=([0,0,0],[10,2,5]), maxfev=10000)
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        n, k = len(x), 3
        aic = n*np.log(ss_res/n) + 2*k
        fits["power_law"] = {"params": {"a": popt[0], "alpha": popt[1], "b": popt[2]},
                             "r2": r2, "aic": aic, "converged": True}
    except Exception as e:
        fits["power_law"] = {"converged": False, "error": str(e)}

    # Logarithmic: P = a * log(N) + b
    try:
        def logarithmic(n, a, b): return a * np.log(np.maximum(n, 1e-10)) + b
        popt, _ = curve_fit(logarithmic, x, y, p0=[0.1, 0.5], bounds=([0,0],[5,5]), maxfev=10000)
        y_pred = logarithmic(x, *popt)
        ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        n, k = len(x), 2
        aic = n*np.log(ss_res/n) + 2*k
        fits["logarithmic"] = {"params": {"a": popt[0], "b": popt[1]},
                               "r2": r2, "aic": aic, "converged": True}
    except Exception as e:
        fits["logarithmic"] = {"converged": False, "error": str(e)}

    # Sigmoid: P = L / (1 + exp(-k*(N-N0)))
    try:
        def sigmoid(n, l_max, k, n0): return l_max / (1 + np.exp(-k*(n-n0)))
        popt, _ = curve_fit(sigmoid, x, y, p0=[2, 0.01, 100], bounds=([0,0,0],[5,1,5000]), maxfev=10000)
        y_pred = sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        n, k_params = len(x), 3
        aic = n*np.log(ss_res/n) + 2*k_params
        fits["sigmoid"] = {"params": {"L": popt[0], "k": popt[1], "N0": popt[2]},
                           "r2": r2, "aic": aic, "converged": True}
    except Exception as e:
        fits["sigmoid"] = {"converged": False, "error": str(e)}

    return fits


# ── 可视化 ────────────────────────────────────────
def plot_all(qty_summary, qual_summary, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 左图：数量 Scaling ---
    ax = axes[0]
    sizes = sorted(qty_summary.keys())
    means = [qty_summary[s]["mean"] for s in sizes]
    ci_los = [qty_summary[s]["ci_lo"] for s in sizes]
    ci_his = [qty_summary[s]["ci_hi"] for s in sizes]

    ax.plot(sizes, means, "o-", color="#2196F3", linewidth=2, markersize=7, label="Observed")
    ax.fill_between(sizes, ci_los, ci_his, alpha=0.15, color="#2196F3")

    # 拟合线
    if fits:
        x_smooth = np.linspace(max(min(sizes), 1), max(sizes), 200)
        colors = {"power_law": "#F44336", "logarithmic": "#4CAF50", "sigmoid": "#FF9800"}
        for fname, fdata in fits.items():
            if not fdata.get("converged"): continue
            p = fdata["params"]
            if fname == "power_law":
                y_fit = p["a"] * np.power(x_smooth, p["alpha"]) + p["b"]
            elif fname == "logarithmic":
                y_fit = p["a"] * np.log(x_smooth) + p["b"]
            elif fname == "sigmoid":
                y_fit = p["L"] / (1 + np.exp(-p["k"]*(x_smooth - p["N0"])))
            label = f"{fname} (R²={fdata['r2']:.3f})"
            ax.plot(x_smooth, y_fit, "--", color=colors.get(fname, "#999"), label=label)

    ax.set_xscale("log")
    ax.set_xlabel("HC Skill Pool Size")
    ax.set_ylabel("Score (0-5)")
    ax.set_title("RQ3: Quantity Scaling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 中图：质量 Scaling ---
    ax = axes[1]
    conds = ["Q100", "Q75", "Q50", "Q25", "Q0"]
    labels = ["100%HC", "75%HC", "50%HC", "25%HC", "0%HC"]
    means_q = [qual_summary.get(c, {}).get("mean", 0) for c in conds]
    ci_lo_q = [qual_summary.get(c, {}).get("ci_lo", 0) for c in conds]
    ci_hi_q = [qual_summary.get(c, {}).get("ci_hi", 0) for c in conds]

    colors_q = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
    bars = ax.bar(range(len(conds)), means_q, color=colors_q, edgecolor="white")
    for i, (lo, hi) in enumerate(zip(ci_lo_q, ci_hi_q)):
        ax.plot([i, i], [lo, hi], color="black", linewidth=1.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (0-5)")
    ax.set_title(f"RQ3: Quality Scaling (pool={QUALITY_FIXED_SIZE})")
    for bar, val in zip(bars, means_q):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    # --- 右图：拟合结果汇总 ---
    ax = axes[2]
    ax.axis("off")
    text = "Curve Fitting Results\n" + "="*30 + "\n\n"
    for fname, fdata in (fits or {}).items():
        if fdata.get("converged"):
            text += f"{fname}:\n"
            text += f"  R² = {fdata['r2']:.4f}\n"
            text += f"  AIC = {fdata['aic']:.1f}\n"
            for k, v in fdata["params"].items():
                text += f"  {k} = {v:.4f}\n"
            text += "\n"
        else:
            text += f"{fname}: failed ({fdata.get('error','')})\n\n"

    best = min((f for f in (fits or {}).values() if f.get("converged")),
               key=lambda x: x["aic"], default=None)
    if best:
        best_name = [k for k, v in fits.items() if v is best][0]
        text += f"Best fit (AIC): {best_name}"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, verticalalignment="top",
            fontfamily="monospace", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "rq3_scaling_law.png"), dpi=200)
    plt.close(fig)
    logger.info(f"Plot saved: {OUTPUT_DIR / 'rq3_scaling_law.png'}")


# ── Main ──────────────────────────────────────────
async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = load_tasks()
    task_lookup = {t["id"]: t for t in tasks}
    hc_skills, ag_skills = load_skills()
    rng = random.Random(42)

    # ═══ Part A: 数量 Scaling ═══
    print("\n📌 Part A: Quantity Scaling")
    qty_cache = OUTPUT_DIR / "quantity_results.json"
    if qty_cache.exists():
        qty_all = json.loads(qty_cache.read_text())
        logger.info(f"Loaded {len(qty_all)} cached quantity results")
    else:
        qty_all = []
        for size in QUANTITY_GRADIENT:
            if size == 0:
                pool = []
            else:
                pool = rng.sample(hc_skills, min(size, len(hc_skills)))

            bm25 = None
            if pool:
                bm25 = BM25()
                bm25.index(pool)

            runs_info = []
            for task in tasks:
                retrieved = bm25.retrieve(task["instruction"], TOP_K) if bm25 else []
                for r in range(N_REPEATS):
                    runs_info.append({
                        "run_id": f"rq3_qty{size}_{task['id']}_r{r}",
                        "task": task, "skills": retrieved, "repeat": r,
                    })

            logger.info(f"  Size={size}: {len(runs_info)} runs")
            results = await run_experiment(aclient, sem, tasks, task_lookup, runs_info)
            for r in results:
                r["pool_size"] = size
                r["experiment"] = "quantity"
            qty_all.extend(results)
            qty_cache.write_text(json.dumps(qty_all, indent=2, ensure_ascii=False))

    # 聚合
    import numpy as np
    qty_summary = {}
    by_size = defaultdict(list)
    for r in qty_all:
        by_size[r["pool_size"]].append(r.get("score", 0))
    for size, scores in sorted(by_size.items()):
        arr = np.array(scores, dtype=float)
        rng_np = np.random.RandomState(42)
        boot = [float(np.mean(rng_np.choice(arr, size=len(arr), replace=True))) for _ in range(3000)]
        qty_summary[size] = {
            "mean": float(np.mean(arr)), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "n": len(scores)
        }

    print("\n  Quantity Scaling Results:")
    for size in sorted(qty_summary.keys()):
        v = qty_summary[size]
        print(f"    size={size:>5d}: {v['mean']:.2f}/5 [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")

    # ═══ Part B: 质量 Scaling ═══
    print("\n📌 Part B: Quality Scaling")
    qual_cache = OUTPUT_DIR / "quality_results.json"
    if qual_cache.exists():
        qual_all = json.loads(qual_cache.read_text())
        logger.info(f"Loaded {len(qual_all)} cached quality results")
    else:
        qual_all = []
        for cond in QUALITY_CONDITIONS:
            n_hc = int(QUALITY_FIXED_SIZE * cond["hc_ratio"])
            n_ag = QUALITY_FIXED_SIZE - n_hc
            pool = (rng.sample(hc_skills, min(n_hc, len(hc_skills))) +
                    rng.sample(ag_skills, min(n_ag, len(ag_skills))))

            bm25 = BM25()
            bm25.index(pool)

            runs_info = []
            for task in tasks:
                retrieved = bm25.retrieve(task["instruction"], TOP_K)
                for r in range(N_REPEATS):
                    runs_info.append({
                        "run_id": f"rq3_{cond['name']}_{task['id']}_r{r}",
                        "task": task, "skills": retrieved, "repeat": r,
                    })

            logger.info(f"  {cond['name']} (HC={cond['hc_ratio']*100:.0f}%): {len(runs_info)} runs")
            results = await run_experiment(aclient, sem, tasks, task_lookup, runs_info)
            for r in results:
                r["condition"] = cond["name"]
                r["hc_ratio"] = cond["hc_ratio"]
                r["experiment"] = "quality"
            qual_all.extend(results)
            qual_cache.write_text(json.dumps(qual_all, indent=2, ensure_ascii=False))

    qual_summary = {}
    by_cond = defaultdict(list)
    for r in qual_all:
        by_cond[r["condition"]].append(r.get("score", 0))
    for cond, scores in by_cond.items():
        arr = np.array(scores, dtype=float)
        rng_np = np.random.RandomState(42)
        boot = [float(np.mean(rng_np.choice(arr, size=len(arr), replace=True))) for _ in range(3000)]
        qual_summary[cond] = {
            "mean": float(np.mean(arr)), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "n": len(scores)
        }

    print("\n  Quality Scaling Results:")
    for cond in ["Q100", "Q75", "Q50", "Q25", "Q0"]:
        if cond in qual_summary:
            v = qual_summary[cond]
            print(f"    {cond}: {v['mean']:.2f}/5 [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")

    # ═══ Part C: 曲线拟合 ═══
    print("\n📌 Part C: Curve Fitting")
    # 只对 size > 0 的数据点拟合
    x_data = [s for s in sorted(qty_summary.keys()) if s > 0]
    y_data = [qty_summary[s]["mean"] for s in x_data]

    fits = {}
    if len(x_data) >= 3:
        fits = fit_scaling_laws(x_data, y_data)
        for fname, fdata in fits.items():
            if fdata.get("converged"):
                print(f"  {fname}: R²={fdata['r2']:.4f}, AIC={fdata['aic']:.1f}")
            else:
                print(f"  {fname}: failed")

    # ═══ 保存 + 可视化 ═══
    summary = {
        "quantity_scaling": {str(k): v for k, v in qty_summary.items()},
        "quality_scaling": qual_summary,
        "curve_fits": {k: {kk: vv for kk, vv in v.items() if kk != "error"} for k, v in fits.items()},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    plot_all(qty_summary, qual_summary, fits)

    print(f"\n✓ RQ3 完成！结果保存在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
