# 流程图

## 1. 总体实验 Pipeline

```mermaid
graph TD
    A[开始] --> B[共享基础设施搭建]

    B --> B1[任务集构建<br/>SkillsBench + SWE-bench + MINT]
    B --> B2[Agent 框架配置<br/>Claude Opus / Sonnet / 开源]
    B --> B3[Skill 注入系统<br/>System Prompt Injection]

    B1 --> C{三个 RQ 并行}
    B2 --> C
    B3 --> C

    C --> D1[RQ1: 信息密度实验]
    C --> D2[RQ2: 质量vs数量实验]
    C --> D3[RQ3: Scaling Law 实验]

    D1 --> E1[RQ1 分析<br/>密度-性能曲线]
    D2 --> E2[RQ2 分析<br/>Pareto 曲线]
    D3 --> E3[RQ3 分析<br/>Scaling 拟合]

    E1 --> F[综合分析]
    E2 --> F
    E3 --> F

    F --> G[Skill-Optimal<br/>Configuration 公式]
    G --> H[论文写作]
    H --> I[结束]

    style A fill:#e1f5fe
    style I fill:#e8f5e9
    style C fill:#fff3e0
    style F fill:#fce4ec
    style G fill:#f3e5f5
```

## 2. 单次 Agent Run 执行流程

```mermaid
graph LR
    A[Task] --> B[TaskLoader]
    B --> C{需要 Skill?}
    C -->|是| D[SkillRetriever<br/>BM25 Top-k]
    C -->|否 Baseline| F[AgentRunner]
    D --> E[SkillInjector<br/>拼接 System Prompt]
    E --> F
    F --> G[LLM API 调用<br/>Claude/GPT/开源]
    G --> H[Agent 输出]
    H --> I[Evaluator<br/>Deterministic Verifier]
    I --> J{Pass?}
    J -->|Yes| K[记录 Pass ✓]
    J -->|No| L[记录 Fail ✗]
    K --> M[保存结果 JSON]
    L --> M

    style A fill:#e3f2fd
    style G fill:#fff8e1
    style I fill:#fce4ec
    style K fill:#e8f5e9
    style L fill:#ffebee
```

## 3. RQ1：信息密度实验流程

```mermaid
graph TD
    A[选择 50 个任务] --> B[提取每个任务的核心知识]
    B --> C[用 LLM 生成 5 级颗粒度 Skill]

    C --> C1[L1: One-liner ~50 tok]
    C --> C2[L2: Checklist ~200 tok]
    C --> C3[L3: Focused SOP ~500 tok]
    C --> C4[L4: Comprehensive ~1500 tok]
    C --> C5[L5: Documentation ~3000+ tok]

    C1 --> D[验证包含关系<br/>L(n+1) ⊃ L(n)]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D

    D --> E[50 任务 × 6 条件 × 3 模型 × 5 次]
    E --> F[收集 4,500 次 Run 结果]

    F --> G1[绘制 Token-Performance 曲线]
    F --> G2[按难度分组分析]
    F --> G3[按领域分组分析]
    F --> G4[计算信息效率 Δrate/Δtok]

    G1 --> H[确定最优颗粒度]
    G2 --> H
    G3 --> H
    G4 --> H

    H --> I[输出：最优 ~500 tokens 的量化证据]

    style A fill:#e3f2fd
    style D fill:#fff3e0
    style F fill:#f3e5f5
    style I fill:#e8f5e9
```

## 4. RQ2：质量 vs 数量实验流程

```mermaid
graph TD
    A[构建 Skill 池] --> A1[HC 池: 200-300 条<br/>人工策划 + 标注]
    A --> A2[AG 池: 10,000-50,000 条<br/>自动生成 + 基本门控]

    A1 --> B[人工质量标注<br/>4维度 × 3-5标注员]
    B --> B1{平均分 ≥ 4.0?}
    B1 -->|Yes| B2[入选 HC 池]
    B1 -->|No| B3[淘汰]

    A2 --> C[构建实验条件]
    B2 --> C

    C --> D1[固定数量实验<br/>8 个条件]
    C --> D2[固定 Token Budget 实验<br/>3 个条件]

    D1 --> E[50 任务 × 11 条件 × 3 模型 × 5 次]
    D2 --> E

    E --> F[收集 8,250 次 Run 结果]

    F --> G1[Pareto 曲线<br/>HC vs AG Scaling]
    F --> G2[Mix Ratio 效果图]
    F --> G3[Noise Tolerance 曲线]
    F --> G4[Cross-domain 分析]

    G1 --> H[确定最优 Quality-Quantity 配比]
    G2 --> H
    G3 --> H
    G4 --> H

    style A1 fill:#c8e6c9
    style A2 fill:#ffecb3
    style B fill:#e1bee7
    style F fill:#f3e5f5
    style H fill:#e8f5e9
```

## 5. RQ3：Scaling Law 实验流程

```mermaid
graph TD
    A[RQ3 Scaling Law] --> B1[数量 Scaling]
    A --> B2[质量 Scaling]
    A --> B3[领域 Scaling]

    B1 --> C1[HC Skill 池<br/>规模梯度: 0→5000]
    C1 --> D1[每规模 5 次随机采样<br/>× 100 任务 × 3 模型 × 5 重复]
    D1 --> E1[拟合 4 种函数<br/>Power/Log/Sigmoid/Composite]
    E1 --> F1[AIC/BIC 模型选择<br/>报告最佳函数 + 参数 CI]

    B2 --> C2[固定规模 500<br/>HC:AG = 0:100→100:0]
    C2 --> D2[5 质量条件<br/>× 100 任务 × 3 模型 × 5 重复]
    D2 --> E2[绘制 2D Heatmap<br/>数量 × 质量 → 性能]

    B3 --> C3[固定总量 500<br/>4 种领域分布策略]
    C3 --> D3[Concentrated / Broad-Anchor<br/>/ Balanced / Random]
    D3 --> E3[Breadth vs Depth 分析]

    F1 --> G[Skill-Optimal Configuration]
    E2 --> G
    E3 --> G

    G --> H["max Performance(N, Q, L)<br/>s.t. N × L ≤ C"]
    H --> I[输出最优配置公式]

    style A fill:#e3f2fd
    style G fill:#fce4ec
    style I fill:#e8f5e9
```

## 6. Skill 生命周期

```mermaid
graph LR
    subgraph 知识来源
        S1[GitHub Repos]
        S2[StackOverflow]
        S3[技术文档]
        S4[SkillsBench]
        S5[Anthropic Skills]
    end

    subgraph 生成
        S1 --> G1[ClawSkills Pipeline<br/>自动提取]
        S2 --> G1
        S3 --> G1
        S4 --> G2[人工策划]
        S5 --> G2
        G1 --> AG[AG 池<br/>10K-50K 条]
        G2 --> HC_RAW[HC 候选]
    end

    subgraph 质量控制
        HC_RAW --> ANN[多维度标注<br/>Correctness / Specificity<br/>Actionability / Completeness]
        ANN --> FILTER{均分 ≥ 4.0?}
        FILTER -->|Pass| HC[HC 池<br/>200-300 条]
        FILTER -->|Fail| REJECT[淘汰]
        AG --> GATE[基本门控<br/>格式 + 去重]
        GATE --> AG_CLEAN[AG 清洗池]
    end

    subgraph 检索与注入
        HC --> POOL[Skill Pool<br/>混合构建]
        AG_CLEAN --> POOL
        POOL --> BM25[BM25 检索<br/>Top-k]
        BM25 --> INJ[System Prompt<br/>注入]
    end

    style HC fill:#c8e6c9
    style AG fill:#ffecb3
    style INJ fill:#e3f2fd
```

## 7. 统计分析 Pipeline

```mermaid
graph TD
    A[原始结果 JSON] --> B[加载与聚合]
    B --> C[按条件分组计算 Pass Rate]

    C --> D1[效果量<br/>Cohen's d]
    C --> D2[显著性检验<br/>Wilcoxon / Bootstrap]
    C --> D3[置信区间<br/>95% Bootstrap CI]

    D1 --> E[统计报告表]
    D2 --> E
    D3 --> E

    C --> F{RQ3?}
    F -->|是| G[曲线拟合]
    G --> G1[Power Law: a·N^α + b]
    G --> G2[Logarithmic: a·log N + b]
    G --> G3[Sigmoid: L/(1+e^-k·N-N₀)]
    G1 --> H[AIC/BIC 模型选择]
    G2 --> H
    G3 --> H
    H --> I[最佳拟合函数 + 参数 CI]

    E --> J[可视化]
    I --> J
    J --> K[论文级图表输出]

    style A fill:#e3f2fd
    style E fill:#fff3e0
    style I fill:#f3e5f5
    style K fill:#e8f5e9
```

## 8. 项目时间线

```mermaid
gantt
    title Skill Scaling Law 项目时间线
    dateFormat  YYYY-MM-DD

    section Phase 1: 基础设施
    TaskLoader + Evaluator     :p1a, 2026-04-01, 10d
    AgentRunner + SkillInjector :p1b, 2026-04-01, 10d
    统计分析框架              :p1c, 2026-04-08, 7d

    section Phase 2: Skill 准备
    HC Skill 收集             :p2a, 2026-04-15, 14d
    AG Skill 生成             :p2b, 2026-04-15, 14d
    人工标注                  :p2c, 2026-04-22, 14d
    多颗粒度生成(RQ1)        :p2d, 2026-04-29, 7d

    section Phase 3: RQ1
    密度实验执行              :p3a, 2026-05-06, 10d
    RQ1 分析                  :p3b, 2026-05-13, 4d

    section Phase 4: RQ2
    质量-数量实验执行         :p4a, 2026-05-20, 10d
    RQ2 分析                  :p4b, 2026-05-27, 4d

    section Phase 5: RQ3
    Scaling 实验执行          :p5a, 2026-06-03, 14d
    曲线拟合 + RQ3 分析       :p5b, 2026-06-14, 7d

    section Phase 6: 写作
    综合分析                  :p6a, 2026-06-24, 7d
    论文撰写                  :p6b, 2026-07-01, 21d
```
