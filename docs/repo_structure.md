# 仓库结构说明

```
skill-scaling-law/
│
├── README.md                          # 项目概述与快速开始
├── setup.py                           # 包安装配置
├── requirements.txt                   # Python 依赖
├── .gitignore                         # Git 忽略规则
│
├── docs/                              # 文档
│   ├── implementation.md              # 完整实现文档
│   ├── flowcharts.md                  # 流程图（Mermaid）
│   └── repo_structure.md             # 本文件：仓库结构说明
│
├── configs/                           # 实验配置（YAML）
│   ├── base.yaml                      # 共享基础配置
│   ├── rq1_density.yaml              # RQ1 信息密度实验配置
│   ├── rq2_quality_quantity.yaml     # RQ2 质量vs数量实验配置
│   └── rq3_scaling.yaml              # RQ3 Scaling Law 实验配置
│
├── src/                               # 源代码
│   ├── __init__.py
│   │
│   ├── infrastructure/               # 共享基础设施
│   │   ├── __init__.py
│   │   ├── task_loader.py            # 任务集加载与预处理
│   │   ├── agent_runner.py           # Agent 执行框架（API 调用封装）
│   │   ├── skill_injector.py         # Skill → System Prompt 注入
│   │   └── evaluator.py             # Pass/Fail 评估 + 指标计算
│   │
│   ├── skills/                       # Skill 管理模块
│   │   ├── __init__.py
│   │   ├── generator.py              # 多颗粒度 Skill 生成（L1-L5）
│   │   ├── curator.py                # 人工质量标注管理
│   │   ├── retriever.py              # BM25 Skill 检索
│   │   └── pool.py                   # Skill 池：HC / AG / Mix 构建
│   │
│   ├── experiments/                  # 三个 RQ 的实验入口
│   │   ├── __init__.py
│   │   ├── rq1_density.py            # RQ1: 信息密度实验
│   │   ├── rq2_quality_quantity.py   # RQ2: 质量vs数量实验
│   │   └── rq3_scaling.py            # RQ3: Scaling Law 实验
│   │
│   ├── analysis/                     # 结果分析与可视化
│   │   ├── __init__.py
│   │   ├── statistics.py             # 统计检验（bootstrap, Wilcoxon）
│   │   ├── curve_fitting.py          # Scaling Law 曲线拟合
│   │   └── visualization.py         # 图表生成（matplotlib/seaborn）
│   │
│   └── utils/                        # 通用工具
│       ├── __init__.py
│       ├── config.py                  # YAML 配置加载与合并
│       ├── logger.py                  # 统一日志
│       └── io.py                      # 文件 I/O 与结果序列化
│
├── data/                              # 数据目录（不入 Git，仅保留 .gitkeep）
│   ├── tasks/                        # 任务数据集（SkillsBench / SWE-bench）
│   ├── skills/                       # Skill 库
│   │   ├── human_curated/            # 人工策划的高质量 Skill
│   │   └── auto_generated/           # 自动生成的 Skill
│   └── annotations/                  # 人工标注结果
│
├── results/                           # 实验结果输出（不入 Git）
│   ├── rq1/                          # RQ1 原始结果 + 分析产物
│   ├── rq2/                          # RQ2 原始结果 + 分析产物
│   └── rq3/                          # RQ3 原始结果 + 分析产物
│
└── scripts/                           # 运行脚本
    ├── run_rq1.sh                    # RQ1 实验启动
    ├── run_rq2.sh                    # RQ2 实验启动
    ├── run_rq3.sh                    # RQ3 实验启动
    └── analyze_all.sh                # 汇总分析与图表生成
```

## 模块职责

### `src/infrastructure/` — 共享基础设施

所有 RQ 共用的核心组件。任何实验都遵循相同的执行路径：
`task_loader → skill_injector → agent_runner → evaluator`

| 模块 | 职责 |
|------|------|
| `task_loader.py` | 从 SkillsBench / SWE-bench 加载任务，统一格式为 `Task` 数据类 |
| `agent_runner.py` | 封装 LLM API 调用，支持 Claude / GPT / 开源模型，管理重试与并发 |
| `skill_injector.py` | 将检索到的 Skill 拼接进 system prompt，记录注入 token 数 |
| `evaluator.py` | 运行 deterministic verifier，计算 pass rate / Cohen's d / CI |

### `src/skills/` — Skill 管理

覆盖 Skill 的完整生命周期：生成 → 标注 → 存储 → 检索。

| 模块 | 职责 |
|------|------|
| `generator.py` | 给定知识源，用 LLM 生成 5 个颗粒度级别（L1-L5）的 Skill |
| `curator.py` | 管理人工标注流程：分配、收集、计算 inter-annotator agreement |
| `retriever.py` | BM25 检索，输入 task description，输出 top-k 相关 Skill |
| `pool.py` | 构建实验所需的 Skill 池（HC / AG / Mix），支持采样与子集操作 |

### `src/experiments/` — 实验入口

每个 RQ 对应一个模块，负责编排该 RQ 的所有实验条件并调用基础设施执行。

### `src/analysis/` — 分析与可视化

| 模块 | 职责 |
|------|------|
| `statistics.py` | Paired bootstrap test, Wilcoxon signed-rank, Cohen's d, CI |
| `curve_fitting.py` | Power law / Log / Sigmoid 拟合，AIC/BIC 模型选择 |
| `visualization.py` | 生成论文级图表：scaling curve, heatmap, Pareto front |

### `configs/` — 实验配置

采用 YAML 继承机制：`base.yaml` 定义共享参数，`rq*.yaml` 通过 `_base_: base.yaml` 继承并覆盖特定参数。

### `data/` 与 `results/`

运行时数据目录，通过 `.gitignore` 排除大文件，仅保留 `.gitkeep` 占位。
