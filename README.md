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


## 数据边界

`DATA_PATH` 是 `tradar.duckdb` 数据目录，Tradar 本身只保留研究过程数据和缓存。相关的数据结构（表和关系）见 `utils/duckdb_manager.py`，使用前可以做相应的迁移/适配。
