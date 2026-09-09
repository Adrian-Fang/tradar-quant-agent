---
research_id: RR-005
date: 2026-09-06
market: A-share
topic: Source-exact public scanner reproduction
strategy: Sequoia-X TurtleTrade / MaVolume / HighTightFlag / LimitUpShakeout / UptrendLimitDown / RpsBreakout
status: rejected
tags: [source-reproduction, TurtleTrade, MaVolume, HighTightFlag, event-scanner, momentum]
source_type: slack
source_ref: [C0BTC7LU07L/1788696332.511649, C0BTC7LU07L/1788698577.506799]
artifacts: [scripts/sequoia_x_repro/common.py, scripts/sequoia_x_repro/turtle_trade.py, scripts/sequoia_x_repro/ma_volume.py, scripts/sequoia_x_repro/high_tight_flag.py, scripts/sequoia_x_repro/limit_up_shakeout.py, scripts/sequoia_x_repro/uptrend_limit_down.py, scripts/sequoia_x_repro/rps_breakout.py, scripts/sequoia_x_repro/select.py]
supersedes: []
---

# Research Question

公开仓库 Sequoia-X 的六个 A 股 scanner，按 source code 而不是 README，是否在 TradaR 本地数据中形成可复用 edge。

# Method

逐策略 `SOURCE_EXACT` 复现；本地数据 warmup 2019-11-28 至 2026-09-04，前复权 OHLC、原始 volume/amount，非 ST、排除 688/689、保留 300，使用 canonical eligible/trading 和 next-open buyable。源 scanner 没有 exit，因此 H1/H5/H10/H20 只是统一的 gross 描述性 outcome，不是源策略退出。

主要规则保持源码语义：TurtleTrade 为前 20 日高点突破+amount>1e8+阳线/真涨；MaVolume 为 MA5 上穿 MA20 且 volume>20 日均量×1.5；HighTightFlag 为高位收缩/缩量但没有真正 breakout；LimitUpShakeout、UptrendLimitDown 和 RpsBreakout 分别按源码固定条件实现。

# Key Findings

| Strategy | Events | Executable | H5 mean/median/win | H10 mean/median/win | H20 mean/median/win |
|---|---:|---:|---|---|---|
| TurtleTrade | 193,505 | 181,126 | -0.98%/-1.34%/40.19% | -0.93%/-1.69%/41.14% | -0.85%/-2.57%/40.82% |
| MaVolume | 52,220 | 50,263 | +0.11%/-0.48%/45.62% | +0.30%/-0.44%/47.12% | +0.90%/-0.64%/47.37% |
| HighTightFlag | 6,974 | 6,873 | -0.87%/-1.52%/40.66% | -1.37%/-2.66%/37.52% | -0.75%/-2.19%/40.96% |
| LimitUpShakeout | 2,501 | 2,306 | -0.92%/-1.68%/41.49% | -1.01%/-2.09%/41.01% | -1.09%/-3.59%/40.02% |
| UptrendLimitDown | 5,215 | 4,997 | -1.15%/-2.06%/38.68% | -0.62%/-1.61%/43.98% | +0.71%/-1.74%/45.43% |
| RpsBreakout | 195,590 | 184,331 | -0.68%/-1.24%/42.19% | -0.74%/-1.90%/41.75% | -0.99%/-3.21%/40.44% |

H20 phase mean 没有一条策略同时在 DEV_IS、DEV_OOS、2026YTD 保持正向；MaVolume 和 UptrendLimitDown 仅 FULL 为正但阶段不稳定。源码/README mismatch 包括 TurtleTrade 排序、HighTightFlag 无 breakout、UptrendLimitDown 无后续反包、Rps 历史适配以及 Baostock 后复权与本地前复权差异。

# Conclusion

`rejected`：source-exact scanner 集合没有跨阶段稳定 edge。不能把个别 FULL 正值写成验证；选择器 CLI 只用于按日期人工复盘，不改变研究结论。

# Caveats

收益评价口径不是源项目的交易退出；本地 universe/filter 与源项目可能不同。没有为弱 scanner 做参数优化或 overlay。

# Provenance

Slack `#tradar-research`：sequoia-x-repro-001 `1788696332.511649`；按日期 selector 的后续 thread `1788698577.506799`。
