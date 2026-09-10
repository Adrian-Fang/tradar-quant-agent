# Tradar

Tradar 是一个正在建设中的 Agentic Quantitative Research Platform。目标不是让 Agent 写更多代码，而是让量化研究能够被稳定地理解、检索上下文、调用确定性研究能力、验证、解释、记录并复用，形成可复现、可评估、可追溯的人机协同研究系统。

## 项目目标 / Target

```text
User Question
  → Retrieval / Context Construction
  → Planning / Tool Calling
  → research engine
  → Validation / Interpretation
  → Memory / Human Review
  → Production
```

`research/` 是 deterministic quantitative engine；`agent/` 是负责编排、能力调用和评估的 harness/orchestration 层。


## 项目边界

Tradar 聚焦于量化研究流程的 Agent 化，不试图覆盖投资决策、交易执行或底层数据基础设施。

明确边界：

- 不推荐股票：不输出面向用户的个股买卖建议、目标价或仓位建议。
- 不推荐策略：可以研究、比较和评估策略，但不向用户给出“应当采用某策略进行投资”的投资建议。研究结论以可复现的实验结果、风险指标和适用条件为主。
- 不负责交易执行：不连接券商下单，不管理真实账户、资金或仓位，也不执行 autonomous trading。
- 不负责市场数据管线：行情、财务和其他研究数据由外部数据基础设施准备，Tradar 只消费规范化后的数据输入。
- 不把 LLM 当作量化计算引擎：因子计算、回测、指标和研究结果由 research/ 中的确定性代码完成；Agent 负责理解、检索、编排、调用、评估和解释。
- 不把探索性代码默认产品化：一次性研究可以存在于 scripts/；只有稳定、可复用、语义明确的能力才沉淀为 research/ 能力或 Agent Tool。

核心职责仅限：

`Research Question → Context / Retrieval → Tool Orchestration → Deterministic Research → Evaluation / Interpretation → Research Memory / Human Review`


## 当前能力 / Current Capabilities

| 能力 | 状态 |
| --- | --- |
| Tool Calling 与 fixture eval | 已落地 |
| Context Selection / Compaction / Construction 与 eval | 已落地 |
| Research Record corpus | 已落地 |
| Deterministic lexical retrieval（含 CJK character bigram） | 已落地 |
| Retrieval Eval baseline | 已落地 |
| `research/` quantitative engine、factor analysis 与 backtest | 已落地 |

## 架构快照 / Architecture Snapshot

```text
User Question
     │
     ▼
agent/                         resources/                 tests/
├── core/                       ├── eval/                  └── regression tests
├── retrieval/                  ├── prompts/
├── context/                    └── knowledge/
└── tools/
     │  harness / orchestration
     ▼
research/  ───────────────────► utils/
deterministic quantitative       data loading / infrastructure
engine
```

`resources/` 只存放非执行型 dataset、prompt 和 Research Record corpus；`tests/` 保存公共能力的回归测试。依赖方向保持为 `agent → research → utils`。

## 项目结构

```text
tradar/
├── agent/
│   ├── core/               # Contracts、provider adapters、resource loaders
│   ├── tools/              # Tool schemas、orchestration、tools、Tool Calling eval
│   ├── context/            # Context selection, compaction, construction, and Eval runner
│   └── retrieval/          # Deterministic research knowledge retrieval
├── research/               # 唯一的量化研究引擎
│   ├── panel.py            # 面板数据与 universe
│   ├── factor_analyzer.py  # 因子分析
│   ├── factor_dsl.py       # 因子 DSL
│   ├── metrics.py          # 绩效指标
│   ├── vector_backtest.py  # 向量化回测
│   └── ...
├── scripts/                # 研究实验、回测和一次性任务
├── utils/                  # 数据加载与通用基础设施
├── data/                   # Tradar 自身的过程数据和研究缓存
├── resources/
│   ├── eval/               # JSON datasets；不放执行代码
│   ├── prompts/            # 稳定的 eval/system prompt
│   └── knowledge/          # Research Records 与 retrieval corpus，不含私有历史
└── tests/                  # 公共能力回归测试
```

## Roadmap

- Multi-step / State
- Planning / Trajectory
- Grounding
- Memory
- Human-in-the-loop
- Whole-system Agent Eval

## 快速开始

注意环境变量 `DATA_PATH` 是 `tradar.duckdb` 存储的日线等数据目录，Tradar 本身只保留研究过程数据和缓存。相关的数据结构（表和关系）见 `utils/duckdb_manager.py`，使用前可以做相应的迁移/适配。

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

本地 deterministic eval（不会调用远程模型）：

```bash
python -m agent.tools.eval --provider fixture --repeats 1
python -m agent.context.eval --provider fixture --repeats 1
python -m agent.retrieval.eval
```

需要模型时，按 provider 配置对应 API key 后运行同一个 capability runner；远程 eval 不属于测试套件。

测试套件：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```


