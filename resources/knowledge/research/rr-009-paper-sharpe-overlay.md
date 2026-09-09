---
research_id: RR-009
date: 2026-09-08
market: A-share
topic: Paper-inspired liquidity, Sharpe weighting and drawdown overlay
strategy: arXiv 2511.13251 PAPER_PROXY on canonical Top15
status: rejected
tags: [paper-reproduction, liquidity, weighting, drawdown, transaction-costs]
source_type: slack
source_ref: C0BTC7LU07L/1788841400.649219
artifacts: [scripts/paper_sharpe_liquidity_overlay.py]
supersedes: []
---

# Research Question

论文的流动性/市值筛选、50/50 inverse-vol/Sharpe weighting 和 drawdown exposure 是否能给 TradaR 已有 Top15 多因子组合带来独立改善；如果论文欠规格，是否应只做 proxy。

# Method

论文没有被 faithful reproduce，统一标记 `PAPER_PROXY`。canonical B0 为：低 5 日换手 34%、高 20 日波动 33%、高 60 日动量 33%，Top15 rank-weight，每 10 个交易日调仓；`build_masks()` 已有 turnover≥1% 与 market-cap 30% screen。Proxy 统一使用 buy10bp/sell15bp；B2 使用固定 60 日 Sharpe proxy、20 日 volatility，负 Sharpe 截为零；B3 使用正文/图示的 100%/80%/40%/0% drawdown exposure，并另给 60% sensitivity，exposure 下一可执行日生效。

# Key Findings

- SOURCE_GAPS：universe cutoff、ADV/bid-ask、Sharpe window/top quantile、负 Sharpe/零和处理、权重上限、benchmark、DD trigger NAV 和交易频率均欠规格；正文与 Fig.3 的 TOP_N=10 及 4%–6% exposure 也有矛盾。
- B0 FULL Ann `-19.19%`、Sharpe `-0.57`、MDD `-66.32%`；DEV_OOS Ann `+27.77%`，2026YTD `-35.61%`，显示短样本/阶段不稳定。
- B1 与 B0 相同，因为 canonical 已包含同类 liquidity/cap screen，未提供增量。
- B2 FULL Ann `-20.27%`、Sharpe `-0.60`、MDD `-68.48%`，没有稳定改善。
- B3-40% FULL Ann `-0.43%`、Sharpe `-0.17`、MDD `-6.42%`，但 98.7% 时间为 0% exposure；B3-60% FULL Ann `-0.58%`、MDD `-7.05%`。风险变好主要来自长期空仓，不是选股 alpha。

# Conclusion

`rejected`：论文高 Sharpe 不能被拆分为可复现的独立选股 edge。B1/B2/B3 均停止；不运行 B4，也不拿 overlay 去挽救其它已停止研究。唯一可迁移事实是 TradaR 已有 canonical liquidity/cap screen，论文 weighting/overlay 尚未显示增量价值。

# Caveats

这是论文启发的 proxy，不是论文策略复现；论文声称的约 5bp/side close-to-close 口径与 TradaR next-open、10/15bp 不同。

# Provenance

Slack `#tradar-research`：paper-sharpe-liquidity-overlay-001 线程 `1788841400.649219`。
