---
research_id: RR-004
date: 2026-08-28
market: A-share
topic: Low-amplitude momentum and executable mapping
strategy: LOW_AMP_MOM full-residual continuous rank exposure
status: rejected
tags: [momentum, low-volatility, residual, transaction-costs, execution]
source_type: slack
source_ref: [C0BTC7LU07L/1787911823.197079, C0BTC7LU07L/1787912461.251609, C0BTC7LU07L/1787913551.228009]
artifacts: [scripts/stocks_active/low_amp_mom_conditional.py]
supersedes: []
---

# Research Question

`LOW_AMP_MOM` 的弱但广泛 residual rank IC 能否经连续 long-only tilt 或 diagnostic long-short 变成可交易收益。

# Method

月末截面、H20、2021-current。因子定义是 20 日内最低振幅 20% 交易日的累计 close-to-close return；full residual controls 为 `NORMAL_M`、log market cap、`VOL20`、log ADV20。组合 mapping 用 T close→T+1 open、existing buyable/sellable，研究线程使用 buy15bp/sell25bp；后续时点诊断比较 CC20、ON1、EXEC20、NEXT_REBAL，且另做 winsorized 诊断。

# Key Findings

- `LOW_RESID_FULL` EXEC20 FULL MeanIC `+1.76%`、ICIR `0.37`、PosIC `67.2%`，但五组 G1→G5 为 `0.62%, 0.73%, 0.77%, 0.99%, 0.75%`，G5-G1 仅 `+0.13%`，不严格单调。
- EXEC20 factor-mimicking FULL 月均毛收益 `+0.08%`，年化近似 `+1.00%`，正月比例 `61.2%`；winsorization 后 `+0.08%/+0.98%/56.7%`，说明不是少数尾部异常造成的主要差异。
- 连续 mapping FULL：`LOW_RESID_LINEAR` Ann `-4.38%`、Active `+0.32%`、Sharpe `-0.87`、MDD `-25.31%`、Cost `26.37%`、Gross `+0.38%`；同口径 `NORMAL_M_LINEAR` Ann `-2.77%`、Active `+7.75%`、Gross `+2.08%`。
- diagnostic `LOW_RESID_LS` Ann `-6.18%`、Sharpe `-4.66`、MDD `-29.31%`、Gross Ann `-1.49%`；2021--2026 yearly gross 全部为负。
- `EXEC20` 相对 `CC20` 保留大部分 IC，故正 IC 不是主要来自 overnight；但 rank slope 太弱，交易成本足以吞掉 gross。

# Conclusion

`rejected` 作为 standalone executable factor：统计上有弱 residual rank 信息，但连续 mapping 和 diagnostic LS 都没有经济价值。不得因为 IC 为正就继续包装成可交易策略。

# Caveats

该结论依赖既定连续 mapping 与成本口径，不是对所有可能组合构造的证明；本研究明确没有使用 Top-N 或参数 sweep。

# Provenance

Slack `#tradar-research`：LOW_AMP_MOM-001 `1787911823.197079`、-002 `1787912461.251609`、-003 `1787913551.228009`。
