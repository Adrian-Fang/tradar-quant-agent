# Tradar - 量化研究与数据分析工具

Tradar 是量化研究、策略验证和 Agent 工具层。核心职责是提供可复用的研究引擎、实验脚本与数据加载能力。

## 项目结构

```text
tradar/
├── agent/
│   ├── core/               # Contracts、provider adapters、resource loaders
│   ├── tools/              # Tool schemas、orchestration、tools、Tool Calling eval
│   └── context/            # Context Eval runner
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
│   ├── eval/               # JSONL datasets；不放执行代码
│   ├── prompts/            # 稳定的 eval/system prompt
│   └── knowledge/          # Research Record 约定，不含私有历史
└── tests/                  # 公共能力回归测试
```

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
```

需要模型时，按 provider 配置对应 API key 后运行同一个 capability runner；远程 eval 不属于测试套件。

测试套件：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```


## 数据边界

`DATA_PATH` 是 `tradar.duckdb` 数据目录，Tradar 本身只保留研究过程数据和缓存。相关的数据结构（表和关系）见 `utils/duckdb_manager.py`，使用前可以做相应的迁移/适配。
