"""
策略容量诊断：根据实际权重变化和滞后成交额，
估算不同资金规模下的市场参与率与可执行容量。

用于策略通过收益、样本外和调仓锚点稳健性验证之后，进入精细成交模拟或实盘之前。
只提供流动性与容量统计，不负责信号生成、组合构建、成交约束或绩效计算。
"""

import numpy as np
import pandas as pd


def lagged_adv(amount: pd.DataFrame, window: int = 20, lag: int = 1) -> pd.DataFrame:
    """计算仅使用交易日前历史数据的平均日成交额。"""
    if window <= 0:
        raise ValueError("window must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    return amount.rolling(window,min_periods=window).mean().shift(lag)


def trade_capacity(
    weights: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    adv_window: int = 20,
    adv_lag: int = 1,
    min_trade_weight: float = 1e-6,
) -> pd.DataFrame:
    """
    将实际执行权重变化转换为逐笔容量记录。

    amount 与组合资金使用相同货币单位。aum_at_100pct_adv 表示该笔交易
    恰好等于100% ADV时的组合规模；给定参与率p时，该笔容量为该值乘以p。
    """
    if min_trade_weight < 0:
        raise ValueError("min_trade_weight must be non-negative")
    if (weights.index.has_duplicates or weights.columns.has_duplicates):
        raise ValueError("weights index and columns must be unique")

    adv = lagged_adv(amount, window=adv_window, lag=adv_lag).reindex_like(weights)

    change = weights.diff()
    if len(change):
        change.iloc[0] = weights.iloc[0]

    signed = change.to_numpy(dtype=float)
    absolute = np.abs(signed)
    row_idx, col_idx = np.nonzero(absolute > min_trade_weight)

    columns = ["side", "weight_change", "trade_weight", "adv", "aum_at_100pct_adv", "valid_liquidity"]

    if not len(row_idx):
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        return pd.DataFrame(columns=columns, index=index,)

    trade_weight = absolute[row_idx, col_idx]
    trade_adv = adv.to_numpy(dtype=float)[row_idx, col_idx]

    valid = (np.isfinite(trade_adv) & (trade_adv > 0))

    unit_capacity = np.full(len(row_idx), np.nan,)
    unit_capacity[valid] = (trade_adv[valid] / trade_weight[valid])

    index = pd.MultiIndex.from_arrays(
        [weights.index.to_numpy()[row_idx], weights.columns.to_numpy()[col_idx]],
        names=["date", "symbol"],
    )

    return pd.DataFrame(
        {
            "side": np.where(signed[row_idx, col_idx] > 0, "buy", "sell",),
            "weight_change": signed[row_idx, col_idx],
            "trade_weight": trade_weight,
            "adv": trade_adv,
            "aum_at_100pct_adv": unit_capacity,
            "valid_liquidity": valid,
        },
        index=index,
    ).sort_index()


def summarize_capacity(
    trades: pd.DataFrame,
    *,
    participation_rates=(0.01, 0.05, 0.10),
    coverage_levels=(0.95, 0.99),
) -> pd.DataFrame:
    """
    汇总订单覆盖率对应的最大组合规模。

    coverage=95%使用逐笔容量的5%分位数，即该资金规模下至少95%的订单
    不超过指定参与率。缺失或非正ADV按零容量处理，并单独报告有效率。
    """
    required = {"side", "aum_at_100pct_adv", "valid_liquidity"}
    missing = required.difference(trades.columns)

    if missing:
        raise ValueError(f"missing trade columns: {sorted(missing)}")

    rates = tuple(float(value) for value in participation_rates)
    levels = tuple(float(value) for value in coverage_levels)

    if any(value <= 0 or value > 1 for value in rates):
        raise ValueError("participation rates must satisfy " "0 < rate <= 1")

    if any(value <= 0 or value > 1 for value in levels):
        raise ValueError("coverage levels must satisfy " "0 < coverage <= 1")

    orders = len(trades)
    valid_rate = (trades["valid_liquidity"].mean() if orders else np.nan)

    unit_capacity = trades["aum_at_100pct_adv"].where(trades["valid_liquidity"], 0.0,).fillna(0.0)

    rows = []

    for rate in rates:
        for coverage in levels:
            max_aum = (
                unit_capacity.quantile(1 - coverage) * rate if orders else np.nan
            )

            rows.append({
                "participation_rate": rate,
                "coverage": coverage,
                "max_aum": max_aum,
                "orders": orders,
                "buy_orders": int(
                    trades["side"].eq("buy").sum()
                ),
                "sell_orders": int(
                    trades["side"].eq("sell").sum()
                ),
                "valid_liquidity_rate": valid_rate,
            })

    return pd.DataFrame(rows)