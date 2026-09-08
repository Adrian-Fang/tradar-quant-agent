# research/vector_backtest.py
"""Fast, stateful portfolio backtest for research targets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.loader import load_trading_calendar


REBALANCE_MASK_ATTR = "rebalance_mask"
DEFAULT_BUY_COST = 0.0010
DEFAULT_SELL_COST = 0.0015


@dataclass
class BacktestConfig:
    buy_cost: float = DEFAULT_BUY_COST
    sell_cost: float = DEFAULT_SELL_COST
    slippage: float = 0.0
    t_plus_1: bool = True


class BacktestResult:
    def __init__(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        *,
        signal_weights: pd.DataFrame | None = None,
        requested_weights: pd.DataFrame | None = None,
        executed_weights: pd.DataFrame | None = None,
        buys: pd.Series | None = None,
        sells: pd.Series | None = None,
        turnover: pd.Series | None = None,
        cost: pd.Series | None = None,
    ):
        self.returns = returns
        self.weights = weights
        self.signal_weights = signal_weights
        self.requested_weights = requested_weights
        self.executed_weights = executed_weights
        empty = pd.Series(0.0, index=returns.index)
        self.buys = empty.copy() if buys is None else buys
        self.sells = empty.copy() if sells is None else sells
        self.turnover = empty.copy() if turnover is None else turnover
        self.cost = empty.copy() if cost is None else cost
        self.equity = (1 + returns).cumprod()


def _apply_execution_constraints(
    target: np.ndarray,
    position_value: np.ndarray,
    cash: float,
    nav: float,
    buyable: np.ndarray,
    sellable: np.ndarray,
    buy_rate: float,
    sell_rate: float,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    """Execute one opening auction target; return values, cash and trade dollars."""
    requested_gross = float(target.sum())
    if requested_gross > 1.0:
        target = target / requested_gross
    target_gross = min(requested_gross, 1.0)
    target_value = target * nav
    tolerance = max(nav, 1.0) * 1e-12

    reductions = target_value < position_value - tolerance
    blocked_sales = reductions & ~sellable
    sale = np.where(reductions & sellable, position_value - target_value, 0.0)
    sale_total = float(sale.sum())
    position_value = position_value - sale
    sell_cost = sale_total * sell_rate
    cash += sale_total - sell_cost

    desired = np.maximum(target_value - position_value, 0.0)
    blocked_buys = (desired > tolerance) & ~buyable
    desired[~buyable] = 0.0
    remaining_budget = max(0.0, target_gross * nav - float(position_value.sum()))
    buy_total = min(float(desired.sum()), remaining_budget, max(cash, 0.0) / (1 + buy_rate))
    if desired.sum() > 0 and buy_total > 0:
        buy = desired * (buy_total / float(desired.sum()))
        position_value = position_value + buy
    buy_cost = buy_total * buy_rate
    cash -= buy_total + buy_cost
    if cash > -tolerance:
        cash = max(0.0, cash)

    return (
        position_value,
        cash,
        buy_total,
        sale_total,
        buy_cost + sell_cost,
        bool(blocked_sales.any() or blocked_buys.any()),
    )


def run_backtest(
    weight_panel,
    price_panel,
    buyable=None,
    sellable=None,
    config: BacktestConfig = None,
    *,
    open_panel=None,
) -> BacktestResult:
    """Run targets formed at close and, by default, execute them next open."""
    config = config or BacktestConfig()
    close = price_panel.astype(float)
    signal_weight = weight_panel.reindex(
        index=close.index, columns=close.columns
    ).fillna(0.0)
    values = signal_weight.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite")
    if (values < 0).any():
        raise ValueError("run_backtest only supports long-only target weights")
    if open_panel is None:
        if config.t_plus_1:
            raise ValueError("open_panel is required for T+1 open execution")
        open_price = close
    else:
        open_price = open_panel.reindex_like(close).astype(float)

    requested_weight = (
        signal_weight.shift(1).fillna(0.0) if config.t_plus_1 else signal_weight
    )
    signal_dates = weight_panel.attrs.get(REBALANCE_MASK_ATTR)
    if signal_dates is None:
        signal_dates = pd.Series(True, index=signal_weight.index)
    else:
        signal_dates = pd.Series(signal_dates, index=signal_weight.index).fillna(False)
    execution_dates = (
        signal_dates.shift(1, fill_value=False) if config.t_plus_1 else signal_dates
    )
    changed = requested_weight.ne(requested_weight.shift(1).fillna(0.0)).any(axis=1)
    execution_dates = execution_dates | changed

    execution_buyable = (
        pd.DataFrame(True, index=close.index, columns=close.columns)
        if buyable is None else buyable.reindex_like(close).fillna(False)
    )
    execution_sellable = (
        pd.DataFrame(True, index=close.index, columns=close.columns)
        if sellable is None else sellable.reindex_like(close).fillna(False)
    )

    opens = open_price.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    targets = requested_weight.to_numpy(dtype=float)
    can_buy = execution_buyable.to_numpy(dtype=bool)
    can_sell = execution_sellable.to_numpy(dtype=bool)
    actual = np.zeros_like(targets)
    executed = np.zeros_like(targets)
    returns = np.zeros(len(close))
    buys = np.zeros(len(close))
    sells = np.zeros(len(close))
    costs = np.zeros(len(close))

    position_value = np.zeros(close.shape[1], dtype=float)
    valuation_price = np.full(close.shape[1], np.nan)
    cash = previous_nav = 1.0
    pending = False
    buy_rate = config.buy_cost + config.slippage
    sell_rate = config.sell_cost + config.slippage

    for row in range(len(close)):
        open_row = opens[row]
        valid_open = np.isfinite(open_row) & (open_row > 0)
        markable = valid_open & np.isfinite(valuation_price)
        position_value[markable] *= open_row[markable] / valuation_price[markable]
        valuation_price[valid_open] = open_row[valid_open]
        open_nav = cash + float(position_value.sum())
        if open_nav <= 0:
            raise RuntimeError(f"portfolio NAV is non-positive on {close.index[row].date()}")

        if bool(execution_dates.iloc[row]) or pending:
            position_value, cash, buy, sell, cost, pending = _apply_execution_constraints(
                targets[row], position_value, cash, open_nav,
                can_buy[row], can_sell[row], buy_rate, sell_rate,
            )
            buys[row] = buy / open_nav
            sells[row] = sell / open_nav
            costs[row] = cost / previous_nav

        post_trade_nav = cash + float(position_value.sum())
        executed[row] = position_value / post_trade_nav

        close_row = closes[row]
        valid_close = np.isfinite(close_row) & (close_row > 0)
        markable = valid_close & np.isfinite(valuation_price)
        position_value[markable] *= close_row[markable] / valuation_price[markable]
        valuation_price[valid_close] = close_row[valid_close]
        close_nav = cash + float(position_value.sum())
        returns[row] = close_nav / previous_nav - 1.0
        actual[row] = position_value / close_nav
        previous_nav = close_nav

    index = close.index
    return BacktestResult(
        pd.Series(returns, index=index, name="return"),
        pd.DataFrame(actual, index=index, columns=close.columns),
        signal_weights=signal_weight,
        requested_weights=requested_weight,
        executed_weights=pd.DataFrame(executed, index=index, columns=close.columns),
        buys=pd.Series(buys, index=index, name="buys"),
        sells=pd.Series(sells, index=index, name="sells"),
        turnover=pd.Series((buys + sells) / 2, index=index, name="turnover"),
        cost=pd.Series(costs, index=index, name="cost"),
    )


def apply_rebalance_frequency(
    weight_panel: pd.DataFrame,
    freq: int,
    offset: int = 0,
) -> pd.DataFrame:
    """Hold each target until its next global-calendar rebalance date."""
    if freq <= 0:
        raise ValueError("freq must be positive")
    if offset < 0 or offset >= freq:
        raise ValueError(f"offset must satisfy 0 <= offset < {freq}")

    calendar = load_trading_calendar()
    ordinals = calendar.get_indexer(weight_panel.index)
    rebalance_mask = pd.Series(
        (ordinals >= 0) & (ordinals % freq == offset), index=weight_panel.index
    )
    mask = np.broadcast_to(rebalance_mask.to_numpy()[:, None], weight_panel.shape)
    result = weight_panel.where(mask).ffill().fillna(0.0)
    result.attrs[REBALANCE_MASK_ATTR] = rebalance_mask
    return result


def top_n_rank_weight(score: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top N and assign rank-proportional long-only weights."""
    def _row(row):
        top = row.nlargest(top_n).dropna()
        if top.empty or top.sum() <= 0:
            return pd.Series(0.0, index=row.index)
        return (top / top.sum()).reindex(row.index).fillna(0.0)
    return score.apply(_row, axis=1)
