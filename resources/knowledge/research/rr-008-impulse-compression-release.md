---
research_id: RR-008
date: 2026-09-07
market: A-share
topic: Impulse to high-level compression to release
strategy: Generalized impulse → compression → release event
status: rejected
tags: [pattern, compression, release, causal, event-study]
source_type: slack
source_ref: C0BTC7LU07L/1788744544.791929
artifacts: [scripts/structure/impulse_high_compression_release.py, data/structure_runs/impulse_high_compression_release/latest/metrics.csv, data/structure_runs/impulse_high_compression_release/latest/block_stability.csv, data/structure_runs/impulse_high_compression_release/latest/recent_signals.csv]
supersedes: []
---

# Research Question

相比单一 scanner，是否存在可因果识别、跨形态复用的“强势启动 → 高位压缩/整理 → 真正 release”结构。

# Method

使用本地 A 股日线，截到 2026-09-04；非 ST、排除 688/689、保留 300。唯一验证定义为：30 日内涨幅≥40%且跑赢沪深300≥20pct 的 impulse；随后 8–20 日高位整理，低点守住 impulse 中点，振幅、绝对收益和成交量中位数都低于 impulse；再以放量阳线收盘越过整理区上沿，next-open 执行。H5/H10/H20 为 gross diagnostic，并按 phase、median、trimmed mean 检查尾部。

# Key Findings

1,211 个 signals，1,137 个 next-open executable。DEV_IS H5/H10/H20 mean `-0.92%/-1.71%/-2.66%`，median `-1.91%/-2.69%/-5.67%`；DEV_OOS `-2.13%/-2.89%/-5.45%`，2026YTD `-2.56%/-1.50%/-0.94%`。各阶段胜率约 28.8%–41.6%，trimmed mean 也全部为负。

即使补齐 prior impulse、relative strength、高位收缩和真实 release，2026 H20 最大单笔 `+107.61%` 仍未改变负 median（`-9.44%`）；signal-day 等权后的月度 median 也没有持续正向。

# Conclusion

`rejected`：形态状态可描述，但没有可复用交易 edge。Phoenix/Valley/Compression 等状态研究不能仅凭“结构看起来合理”升级为 alpha。

# Caveats

该结果是机制诊断，不是止盈止损或 regime overlay 研究；没有为负结果继续参数优化。

# Provenance

Slack `#tradar-research`：structure-pattern-synthesis-001 线程 `1788744544.791929`。
