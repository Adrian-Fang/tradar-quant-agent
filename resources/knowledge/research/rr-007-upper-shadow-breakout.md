---
research_id: RR-007
date: 2026-09-07
market: A-share
topic: Limit-up upper-shadow breakout event
strategy: 仙人指路 / upper-shadow breakout
status: rejected
tags: [limit-up, upper-shadow, breakout, event-study, rejected]
source_type: slack
source_ref: C0BTC7LU07L/1788784335.256449
artifacts: [scripts/event/upper_shadow_breakout.py]
supersedes: []
---

# Research Question

涨停 L 后次日长上影 U，随后首次收盘突破 U.high 的事件，是否比普通涨停后上影线有稳定正 edge，且 2026 仍有效。

# Method

2021-01-01 至 2026-07-31；raw OHLC + `research.panel.price_limit_pct_panel` 识别涨停，U 必须是 L 次日；上影线占振幅≥50%、close position≤0.60；U 前 20 日价格高点或成交量峰值创新高；S 为 U 后 20 日内首次 `raw close > U.high`。S close→forward adjusted close 是 diagnostic，另报 S+1 open。

# Key Findings

漏斗：L 71,220 → U 长上影 13,944 → price-high 9,806 / volume-high 8,444 → OR 候选 11,046 → 成功突破 5,984；核心 EITHER 版本 4,663。

核心 S-close 结果（Mean/Median/Win/Trim5）：FULL H1 `+0.13%/-0.83%/43.0%/+0.02%`，H3 `-0.45%/-1.92%/39.2%/-1.04%`，H5 `-1.01%/-2.52%/38.5%/-1.81%`。DEV_IS、DEV_OOS、2026YTD 的 H3/H5 均普遍为负。S+1 open FULL H1/H3/H5 `-0.92%/-1.85%/-2.29%`。

# Conclusion

`rejected`：H1 微小正均值伴随负中位数，去尾后接近零；H3/H5 各阶段转负，严格 next-open 更差。不能把少数右尾或盘中/收盘假设包装成可执行 alpha。

# Caveats

本研究的 OR、price-high、volume-high 变体均只作固定语义对照，没有继续调整窗口或退出规则。

# Provenance

Slack `#tradar-research`：event-upper-shadow-breakout-001 线程 `1788784335.256449`。
