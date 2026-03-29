# 实现文档：Skill Scaling Laws for AI Agents

## 1. 研究目标

系统性量化 AI Agent 中 Skill 注入的三个核心维度：
1. **信息密度**（RQ1）：同一知识用不同详细程度表达，最优颗粒度是什么？
2. **质量-数量权衡**（RQ2）：少量高质量 vs 大量自动生成 Skill，如何权衡？
3. **Scaling Law**（RQ3）：Agent 性能随 Skill 规模是否遵循可预测的数学规律？

---

## 2. 共享基础设施

### 2.1 任务集

| 来源 | 数量 | 领域 | 用途 |
|------|------|------|------|
| SkillsBench | 84 | 多领域 | 主体任务集，自带 deterministic verifier |
| SWE-bench Verified | 50-100 | 软件工程 | 深度补充 |
| MINT / AgentBench | 30-50 | 数据分析、文档生成等 | 非编程任务覆盖 |

**总计**：150-200 个任务，覆盖 8-10 个领域，每领域 ≥15 个任务。

统一数据格式：

```python
@dataclass
class Task:
    task_id: str              # 唯一标识
    source: str               # skillsbench / swebench / mint
    domain: str               # 领域标签
    difficulty: str           # easy / medium / hard
    instruction: str          # 任务指令
    ground_truth: Any         # 正确答案或验证函数
    metadata: dict            # 额外元数据
```

### 2.2 Agent 框架

选择 3 个 agent-model 配置验证结论 generalizability：

| 类别 | 模型 | 用途 |
|------|------|------|
| 强模型 | Claude Opus / GPT-4.1 | 性能上界参考 |
| 中等模型 | Claude Sonnet / GPT-4.1-mini | 主要实验模型 |
| 开源模型 | Llama 3 / Qwen 2.5 | 开源可复现性 |

Agent 执行流程：
1. 加载 Task instruction
2. 通过 `SkillInjector` 将检索到的 Skill 注入 system prompt
3. 调用 LLM API，收集 agent 输出
4. 通过 `Evaluator` 执行 deterministic verification
5. 记录 pass/fail、token 用量、延迟等指标

### 2.3 Skill 注入方式

统一采用 **system prompt injection**：

```
[System Prompt]
You are a helpful assistant.

[Injected Skills]
## Relevant Skills
{skill_1_content}
---
{skill_2_content}
---
{skill_3_content}

[Task Instruction]
{task_instruction}
```

每次注入记录：注入的 Skill ID 列表、总 token 数、各 Skill 的 token 数。

### 2.4 统计方法

| 方法 | 用途 |
|------|------|
| Cohen's d | 效果量度量 |
| Paired Bootstrap Test | 跨条件对比显著性 |
| Wilcoxon Signed-Rank Test | 非参数配对检验 |
| 95% Bootstrap CI | 置信区间估计 |

每个 (task, skill_config) 组合运行 **5 次**取平均，与 SkillsBench 保持一致。

---

## 3. RQ1：Skill 的最优信息密度

### 3.1 实验变量

- **自变量**：Skill 颗粒度级别（L1-L5）
- **因变量**：Agent pass rate
- **控制变量**：任务集、模型、检索方法

### 3.2 Skill 颗粒度定义

| Level | 名称 | Token 数 | 内容结构 | 信息包含关系 |
|-------|------|----------|---------|-------------|
| L1 | One-liner | ~50 | 一句话提示 | 基础 |
| L2 | Checklist | ~200 | 3-5 条要点（what + why） | ⊃ L1 |
| L3 | Focused SOP | ~500 | Background + Steps(3-5) + Verification | ⊃ L2 |
| L4 | Comprehensive Guide | ~1500 | 完整 SKILL.md 格式 | ⊃ L3 |
| L5 | Documentation | ~3000+ | 原始文档级详细度 | ⊃ L4 |

**关键约束**：严格包含关系 — Level N+1 包含 Level N 的全部信息 + 额外信息。测量的是"额外信息的边际价值"。

### 3.3 Skill 生成 Protocol

对每个任务的核心知识，使用 LLM（GPT-4.1）按模板生成 5 个级别：

```python
GENERATION_PROMPT = """
Given the following knowledge needed to solve a task:
{knowledge_description}

Generate a skill description at granularity level {level}:
- L1 (~50 tokens): Single sentence hint
- L2 (~200 tokens): 3-5 bullet checklist (what + why)
- L3 (~500 tokens): Background + 3-5 Steps + Verification
- L4 (~1500 tokens): Full SKILL.md with edge cases + examples
- L5 (~3000+ tokens): Full documentation with history + code samples

IMPORTANT: L{level} must contain ALL information from L{level-1} plus additional details.
Output ONLY the skill content.
"""
```

### 3.4 实验矩阵

| 维度 | 值 |
|------|-----|
| 任务 | 50 个（从总任务集中按领域分层抽样） |
| 颗粒度 | L1, L2, L3, L4, L5, Baseline（无 Skill） = 6 条件 |
| 模型 | 3 个 |
| 重复 | 5 次 |
| **总 Agent Run** | **50 × 6 × 3 × 5 = 4,500** |

### 3.5 分析方法

1. **核心曲线**：横轴 Skill token 数（对数尺度），纵轴 pass rate → 期望观察倒 U 形或对数饱和
2. **分组分析**：按任务难度（easy/medium/hard）和领域分组，检查最优颗粒度的条件依赖性
3. **信息效率**：Δ(pass_rate) / Δ(tokens)，找到边际收益趋零的拐点
4. **统计检验**：相邻 Level 间的 Wilcoxon test，确认差异显著性

### 3.6 预期产出

- 图：Information Density vs Performance Curve（含 CI band）
- 图：Optimal Granularity by Task Difficulty（分面图）
- 图：Information Efficiency Curve（边际收益）
- 表：各 Level 在不同领域/难度下的 pass rate 对比
- **核心发现**：量化确认 ~500 tokens（L3）为最优信息密度，与 SkillsBench 定性观察吻合

---

## 4. RQ2：Quality vs Quantity 的权衡

### 4.1 Skill 池构建

**高质量池（Human-Curated, HC）**：
- 来源：SkillsBench curated skills + Anthropic 官方 skills 仓库 + 人工验证
- 规模：200-300 条
- 质量控制：3-5 名标注员，每条 Skill 在 4 个维度打分（1-5 分）
  - Correctness（正确性）
  - Specificity（具体性）
  - Actionability（可操作性）
  - Completeness（完整性）
- 入选阈值：平均分 ≥ 4.0
- Inter-annotator Agreement：Krippendorff's α ≥ 0.7

**自动生成池（Auto-Generated, AG）**：
- 来源：ClawSkills pipeline 从 GitHub / StackOverflow / 技术文档提取
- 规模：10,000-50,000 条
- 质量控制：仅基本门控（格式检查 + 去重），不做人工审核

### 4.2 实验条件

#### 固定数量实验

| 条件 | 库规模 | 来源 | 检索 top-k |
|------|--------|------|-----------|
| HC-small | 100 | HC only | 3 |
| HC-medium | 200 | HC only | 3 |
| AG-small | 100 | AG random sample | 3 |
| AG-medium | 1,000 | AG only | 3 |
| AG-large | 10,000 | AG only | 3 |
| Mix-7030 | 1,000 | 70% AG + 30% HC | 3 |
| Mix-5050 | 1,000 | 50% AG + 50% HC | 3 |
| Baseline | 0 | — | 0 |

#### 固定 Token Budget 实验

固定注入 1,500 tokens，对比：
- 1 条 L4 高质量 Skill
- 3 条 L2 中等质量 Skill
- 5 条 L1 自动生成 Skill

### 4.3 检索系统

- 算法：BM25（与 ClawSkills 一致）
- 输入：Task description
- 输出：Top-3 Skill
- 质量控制：人工标注 ground truth relevant skills，报告 Recall@3 和 nDCG@3

### 4.4 实验矩阵

| 维度 | 值 |
|------|-----|
| 任务 | 50 个 |
| 条件 | 8 个（固定数量）+ 3 个（固定 budget）= 11 条件 |
| 模型 | 3 个 |
| 重复 | 5 次 |
| **总 Agent Run** | **50 × 11 × 3 × 5 ≈ 8,250** |

### 4.5 分析方法

1. **Quality-Quantity Pareto 曲线**：横轴库规模（对数），纵轴 pass rate，分别画 HC-only 和 AG-only 曲线
2. **Mix Ratio 效果**：固定总量 1,000，HC:AG 比例从 0:100 到 100:0
3. **Noise Tolerance**：在 HC 池中逐步混入 AG（10% → 80%），观察性能退化曲线
4. **Cross-domain 分析**：检查 HC/AG 差异是否因领域而异

### 4.6 预期产出

- 图：Pareto Frontier（HC vs AG scaling）
- 图：Mix Ratio vs Performance
- 图：Noise Tolerance Curve
- 表：各条件在不同领域的 pass rate
- **核心发现**：质量远比数量重要，少量 HC 即超过大规模 AG；但 Mix 可能有覆盖面优势

---

## 5. RQ3：Scaling Law

### 5.1 数量 Scaling 实验

使用 HC skill（排除质量噪音），逐步增加库规模：

```
规模梯度: [0, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
```

每个规模随机采样 5 次不同 Skill 子集，消除选择 bias。检索 top-k 固定为 3。

| 维度 | 值 |
|------|-----|
| 任务 | 100 个 |
| 规模梯度 | 10 个 |
| 子集采样 | 5 次 |
| 模型 | 3 个 |
| 重复 | 5 次 |
| **总 Agent Run** | **100 × 10 × 5 × 3 × 5 = 75,000**（简化后 ~15,000） |

简化策略：中等模型跑全量，强模型和开源模型只跑关键规模点（0, 50, 200, 1000, 5000）。

### 5.2 曲线拟合

尝试 4 种函数形式：

| 函数 | 公式 | 参数 |
|------|------|------|
| Power Law | P = a·N^α + b | a, α, b |
| Logarithmic | P = a·log(N) + b | a, b |
| Sigmoid | P = L / (1 + e^(-k(N-N₀))) | L, k, N₀ |
| Composite Decay | 复用 Li 2026 capacity threshold | 见论文 |

模型选择：AIC / BIC 最小化。报告最佳函数的参数估计 + 95% CI。

### 5.3 质量 Scaling 实验

固定库规模 500，改变质量混合比例：

| 条件 | HC 比例 | AG 比例 |
|------|---------|---------|
| Q100 | 100% | 0% |
| Q75 | 75% | 25% |
| Q50 | 50% | 50% |
| Q25 | 25% | 75% |
| Q0 | 0% | 100% |

### 5.4 领域 Scaling 实验

固定总 Skill 500 条，改变领域分布：

| 条件 | 分布策略 |
|------|---------|
| Concentrated | 100% 在目标领域 |
| Broad-Anchor | 70% 目标领域 + 30% 分散 |
| Balanced | 均匀分布在 10 个领域 |
| Random | 随机采样 |

### 5.5 Skill-Optimal Configuration

定义优化问题：

```
max  Performance(N, Q, L)
s.t. N × L ≤ C          (Context Budget 约束)
     Q ∈ [0, 1]          (质量比例)
     L ∈ {50, 200, 500, 1500, 3000}  (颗粒度)
```

基于 RQ1（最优 L）+ RQ2（Q vs N trade-off）+ 数量 scaling 数据，给出近似最优配置公式。

### 5.6 预期产出

- 图：Quantity Scaling Curve + 拟合线（含 CI band）
- 图：2D Heatmap（数量 × 质量 → 性能），含等性能线
- 图：Domain Distribution 对比
- 表：函数拟合结果（参数 + AIC/BIC）
- 公式：Skill-Optimal Configuration 公式
- **核心发现**：存在 log-linear scaling law，有效前沿在 500-2000 条高质量 Skill

---

## 6. 实现细节

### 6.1 并发执行

Agent run 之间独立，使用 Python `asyncio` + `semaphore` 控制并发：

```python
async def run_experiment(conditions, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for condition in conditions:
        for repeat in range(5):
            tasks.append(run_single(semaphore, condition, repeat))
    results = await asyncio.gather(*tasks)
    return results
```

### 6.2 结果存储

每次 agent run 的结果存为 JSON：

```json
{
    "run_id": "rq1_task042_L3_claude_r3",
    "task_id": "task_042",
    "skill_config": {"level": "L3", "tokens": 487},
    "model": "claude-sonnet",
    "passed": true,
    "tokens_used": 3542,
    "latency_ms": 8230,
    "agent_output": "...",
    "timestamp": "2026-03-29T10:00:00Z"
}
```

### 6.3 可复现性

- 所有随机采样使用固定 seed（42）
- 记录完整的 API 调用参数（temperature=0 用于 deterministic 输出）
- 结果 JSON 包含完整的实验配置快照

---

## 7. 成本估算

| 项目 | 估算 |
|------|------|
| RQ1 Agent Run | 4,500 次，~$2,000-5,000 |
| RQ2 Agent Run | 8,250 次，~$3,000-8,000 |
| RQ3 Agent Run | 15,000 次，~$5,000-15,000 |
| 人工标注 | 3-5 人 × 2-3 周，~$3,000-5,000 |
| **总计** | **~$13,000-33,000** |

## 8. 时间线

| 阶段 | 时长 | 内容 |
|------|------|------|
| Phase 1 | 2-3 周 | 基础设施搭建（task_loader, agent_runner, evaluator） |
| Phase 2 | 3-4 周 | Skill 收集、生成、人工标注 |
| Phase 3 | 2 周 | RQ1 实验执行与初步分析 |
| Phase 4 | 2 周 | RQ2 实验执行与初步分析 |
| Phase 5 | 3 周 | RQ3 实验执行与初步分析 |
| Phase 6 | 4 周 | 综合分析、图表制作、论文写作 |
| **总计** | **~16-18 周（4-5 个月）** | |
