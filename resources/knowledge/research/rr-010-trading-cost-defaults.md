---
research_id: RR-010
date: 2026-09-01
market: multi-asset
topic: Canonical asymmetric transaction-cost defaults
strategy: Research framework cost handling
status: validated
tags: [transaction-costs, buy-cost, sell-cost, backtest-infrastructure]
source_type: slack
source_ref: C0BTC7LU07L/1788249905.545929
artifacts: [research/vector_backtest.py]
supersedes: []
---

# Research Question

研究框架是否能让未显式覆盖成本的研究统一继承买入 10bp、卖出 15bp，并按真实买卖方向分别计费，而不是每个调仓日固定扣 25bp。

# Method

检查 `research.vector_backtest`、`BacktestConfig` 和相关 metrics；保留显式脚本/YAML override。用一笔已知买入再卖出的 round-trip smoke 检查成本传递。

# Key Findings

- canonical default 集中在 `research/vector_backtest.py`：BUY=10bp、SELL=15bp。
- engine 支持非对称成本，按实际 buy/sell 成交额分别计费；没有把 25bp 当成每个调仓日固定扣费。
- smoke 输出：`BUY_COST_BP=10.0000`、`SELL_COST_BP=15.0000`、`ROUND_TRIP_COST_BP=24.9750`、`ENGINE_COST_SUM_BP=24.9900`，与约 25bp 一致。

# Conclusion

`validated`：这是研究框架完整性验证，不是 alpha 结论。后续研究默认成本应从该 canonical engine 继承，显式 override 仍有效。

# Caveats

该任务没有批量重跑历史研究，因此不能证明所有旧脚本都已经移除自定义成本；它只验证默认 engine 和最小 round-trip。

# Provenance

Slack `#tradar-research`：global-trading-cost-defaults-001 线程 `1788249905.545929`。
