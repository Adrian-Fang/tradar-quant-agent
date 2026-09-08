# Tradar - 量化研究与数据分析工具

Tradar 是量化研究、策略验证和 Agent 工具层。核心职责是提供可复用的研究引擎、实验脚本与数据加载能力。

## 项目结构

```text
tradar/
├── agent/                  # Agent orchestration 与稳定 Tool contract
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
└── tests/                  # 公共能力回归测试
```

## 快速开始

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```


## 数据边界

`DATA_PATH` 是 `tradar.duckdb` 数据目录，Tradar 本身只保留研究过程数据和缓存。相关的数据结构（表和关系）见 `utils/duckdb_manager.py`，使用前可以做相应的迁移/适配。