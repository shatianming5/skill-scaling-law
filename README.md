# Skill Scaling Laws for AI Agents

**探究 AI Agent 中 Skill 注入的最优信息密度、质量-数量权衡与 Scaling Law**

## 研究概述

本项目系统性地研究 AI Agent 系统中 skill（技能知识）注入的关键问题，围绕三个核心研究问题（RQ）展开：

| RQ | 研究问题 | 核心变量 |
|----|---------|---------|
| RQ1 | Skill 的最优信息密度是什么？ | Skill 的 token 长度与信息结构 |
| RQ2 | 高质量少量 vs 低质量大量 Skill，哪个更好？ | Skill 库的 quality-quantity 组合 |
| RQ3 | Agent 性能随 Skill 规模变化是否存在 Scaling Law？ | 数量、质量、领域覆盖度的联合效应 |

## 快速开始

```bash
# 安装依赖
pip install -e .

# 运行 RQ1 实验
bash scripts/run_rq1.sh

# 运行 RQ2 实验
bash scripts/run_rq2.sh

# 运行 RQ3 实验
bash scripts/run_rq3.sh

# 汇总分析
bash scripts/analyze_all.sh
```

## 项目结构

详见 [docs/repo_structure.md](docs/repo_structure.md)

## 文档

- [实现文档](docs/implementation.md) — 完整的实验设计、方法论与实现细节
- [流程图](docs/flowcharts.md) — 实验 Pipeline 与数据流可视化
- [仓库结构](docs/repo_structure.md) — 代码组织与模块职责说明

## 实验规模

- **任务集**：150-200 个任务，覆盖 8-10 个领域
- **Agent 配置**：3 个模型（强/中/开源）
- **总 Agent Run**：~25,000 次
- **统计方法**：Cohen's d / Bootstrap CI / Wilcoxon signed-rank test

## 参考

- [SkillsBench](https://github.com/anthropics/skillsbench) — 任务集与评估框架基础
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) — 软件工程任务补充
- [ClawSkills](https://github.com/clawskills) — 自动 Skill 生成 Pipeline 参考
- Li 2026 — Capacity threshold 与 phase transition 模型

## License

MIT
