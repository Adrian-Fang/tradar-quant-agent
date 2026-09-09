---
research_id: RR-002
date: 2026-08-28
market: A-share
topic: 52-week-high proximity
strategy: HIGH52 and residual near-high penalty
status: rejected
tags: [52-week-high, momentum, residual, tail-penalty]
source_type: slack
source_ref: [C0BTC7LU07L/1787914634.094389, C0BTC7LU07L/1787932488.483929]
artifacts: [data/factor_defs/high52.yaml, scripts/stocks_active/high52.py]
supersedes: []
---

# Research Question

个股接近过去 252 日最高收盘价是否能在 H20 预测未来收益，且在控制 12-1 momentum、波动率、市值和流动性后仍有独立信息。

# Method

月末截面、H20，先做 2021+，再扩展到 2016-01-01 至 2026-07-31（127 个完整月末）。`HIGH52 = adjusted close / rolling_max(adjusted close, 252)`，higher=better；残差版本控制 `MOM12_1`，再加 `VOL20`、log market cap、log ADV20。未做参数 sweep 或 portfolio optimization。

# Key Findings

- 2016+ FULL MeanIC / ICIR / PosIC：RAW `-3.06%/-0.19/42.5%`；RESID_MOM `-2.36%/-0.16/44.1%`；RESID_FULL `-1.50%/-0.12/46.5%`。
- RESID_FULL 五组 FULL H20 为 `1.08%, 1.11%, 1.19%, 1.09%, 0.78%`，G5-G1 `-0.30%`；十组 G10 为 `0.41%`，G10-G1 `-0.63%`，monotonicity `0.02`。
- RESID_FULL 尾部 penalty 定义为 `G10 - mean(G1...G9)`：PRE 2016-20 `-0.75%`、DEV_IS `-0.67%`、DEV_OOS `-0.54%`、POST_2026 `-1.15%`、FULL `-0.71%`。G10 落后 G1-G9 的月份比例 FULL `56.7%`。
- 尾部方向跨长期阶段大体为负，但逐年反复：2016 `-1.48%`、2018 `-1.35%`、2022 `-2.26%`、2024 `-1.85%`，而 2020、2021、2023、2025 为正或接近零。
- `HIGH52` 与 `MOM12_1` 的 FULL 相关性 `+44.99%`；与 VOL20、size、ADV 的相关性分别 `+8.84%`、`+21.25%`、`+19.00%`。原始 near-high 弱势并非完全由这些暴露替代。

# Conclusion

`rejected` 作为 standalone 正向动量因子：原假设方向相反，近高尾部存在可复现的 penalty 形状，但逐年 sign 不稳定，不足以称为已验证 alpha。Slack 结论只支持做一次固定口径的 avoid-leg 诊断，不支持直接生产化。

# Caveats

尾部 penalty 是月度横截面统计，不等同于可执行 short 策略；avoid-leg 的交易可行性、借券和成本没有在该研究中验证。

# Provenance

Slack `#tradar-research`：high52-001 线程 `1787914634.094389`；high52-002 长历史扩展线程 `1787932488.483929`。
