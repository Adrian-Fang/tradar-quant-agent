"""纯函数绩效指标计算，只吃收益率 Series，不关心数据从哪来。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    return (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    if excess.std() == 0 or excess.empty:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum / peak - 1).min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if not mdd:
        return np.nan
    return annualized_return(returns) / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def calc_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    returns = returns.dropna()
    result = {
        "total_return": (1 + returns).prod() - 1,
        "annual_return": annualized_return(returns),
        "annual_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "win_rate": win_rate(returns),
        "n_days": len(returns),
    }
    if benchmark is not None:
        common = returns.index.intersection(benchmark.dropna().index)
        r, b = returns.loc[common], benchmark.loc[common]
        cov = np.cov(r, b)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] else np.nan
        result.update({
            "beta": beta,
            "alpha": annualized_return(r) - beta * annualized_return(b),
            "benchmark_annual_return": annualized_return(b),
            "excess_annual_return": annualized_return(r) - annualized_return(b),
        })
    return result