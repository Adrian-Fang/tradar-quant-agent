---
research_id: RR-006
date: 2026-08-31
market: A-share
topic: Volume-confirmed platform breakout
strategy: 20D adjusted-close breakout with VR5 buckets
status: rejected
tags: [breakout, volume, event-study, execution, episode-dedup]
source_type: slack
source_ref: C0BTC7LU07L/1788142209.838549
artifacts: [scripts/volume_breakout/volume_platform_breakout.py]
supersedes: []
---

# Research Question

突破过去 20 日 adjusted-close 平台并伴随当日相对前 5 日均量放大，是否比普通突破有稳定增量；D0 close 的理论收益能否保留到 D+1 open。

# Method

2021-01-01 至 2026-07-31；排除 ST、688/689、上市不足 250 日；平台为 `adj_close[D0] > max(adj_close[D-20:D-1])`，`VR5=volume[D0]/mean(volume[D-5:D-1])`，比较 `<1x`、`1-1.5x`、`1.5-2x`、`2-3x`、`>=3x`。同时输出 D0 close diagnostic 和 D+1 open executable；连续突破另做 episode-first。

# Key Findings

- Event-level 共 403,192，4,506 symbols，1331 event days；episode-first 245,977。
- Event-level FULL ALL：D0 close H5/H20 `-0.51%/-0.24%`；D+1 open H5/H20 `-0.44%/+0.07%`。
- FULL `1.5-2x` D+1 open H5/H20 `-0.27%/+0.28%`；`2-3x` `-0.77%/-0.09%`；`>=3x` `-1.05%/-0.90%`。放量不存在近似单调改善，极端量比更差。
- 2026YTD ALL D+1 open H20 `-1.54%`；DEV_IS `-0.33%`，DEV_OOS `+0.98%`，阶段不稳定。episode 去重后结论不变且更接近零/负。
- D0 涨幅×VR5>=1.5 的 FULL D+1-open H20：涨幅≤2% `+0.94%`，2-5% `+0.64%`，5-8% `+0.12%`，≥8% `-1.21%`；更像大阳线/过热替代变量。

# Conclusion

`rejected`：`VR5>=1.5x` 没有稳定优于全部突破，D0-close edge 也不能可靠转移到 next-open。停止进入真实退出/持有规则研究。

# Caveats

H1/H3/H5/H10/H20 是机制诊断，不是策略 exit；结果使用 adjusted price 评价、canonical 涨停边界和可买约束。

# Provenance

Slack `#tradar-research`：volume-platform-breakout-001 线程 `1788142209.838549`。
