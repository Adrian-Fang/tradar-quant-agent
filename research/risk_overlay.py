# research/risk_overlay.py
"""
通用风险叠加层：输入策略生成的 weight_panel，输出风控调整后的 weight_panel。
和具体因子解耦，可用于任意选股策略。

包含：
1. regime_filter             单指数趋势状态控制
2. multi_index_regime_filter 多指数趋势状态控制
3. volatility_scaling        股票横截面风险平衡
4. portfolio_vol_target      组合波动率目标控制
5. drawdown_filter            组合回撤控制
6. crowding_filter            因子拥挤度控制
"""
import numpy as np
import pandas as pd


def regime_filter(weight_panel: pd.DataFrame,index_close: pd.Series,ma_window: int = 60,
                  scale_when_below: float = 0.2,min_history_scale=None) -> pd.DataFrame:
    if min_history_scale is None:
        min_history_scale = scale_when_below
    ma = index_close.rolling(ma_window,min_periods=ma_window).mean()
    strong = (index_close >= ma).reindex(weight_panel.index)
    scale = strong.map({True:1.0,False:scale_when_below})
    scale = scale.where(strong.notna(),min_history_scale)
    result = weight_panel.mul(scale,axis=0)
    result.attrs.update(weight_panel.attrs)
    return result


def multi_index_regime_filter(weight_panel: pd.DataFrame,index_close_map: dict,
                              ma_window: int = 120,weights: dict = None,
                              scale_points=None) -> pd.DataFrame:
    """
    多指数趋势融合。
    index_close_map:
        {"HS300":series,"ZZ500":series,"ZZ1000":series,"CYB":series}

    根据指数相对MA距离生成综合regime分数。
    """
    if scale_points is None:
        scale_points = [(0.05,1.0),(0.0,0.7),(-0.05,0.4),(-np.inf,0.2)]

    score = pd.Series(0.0,index=weight_panel.index)
    valid = pd.Series(0,index=weight_panel.index)

    if weights is None:
        weights = {k:1/len(index_close_map) for k in index_close_map}

    for name,close in index_close_map.items():
        ma = close.rolling(ma_window,min_periods=ma_window).mean()
        s = (close / ma - 1).reindex(weight_panel.index)
        score += s.fillna(0) * weights.get(name,0)
        valid += s.notna().astype(int)

    score = score.where(valid > 0,np.nan)

    scale = pd.Series(index=score.index,dtype=float)
    for threshold,value in scale_points:
        scale.loc[(scale.isna()) & (score >= threshold)] = value

    return weight_panel.mul(scale.fillna(0.2),axis=0)


def volatility_scaling(weight_panel: pd.DataFrame,price_panel: pd.DataFrame,
                       vol_window: int = 20) -> pd.DataFrame:
    ret = price_panel.pct_change()
    vol = ret.rolling(vol_window,min_periods=max(5,vol_window//2)).std()
    inv_vol = 1 / vol.replace(0,np.nan)
    active = weight_panel > 0
    weighted = inv_vol.where(active)
    return weighted.div(weighted.sum(axis=1),axis=0).fillna(0)


def portfolio_vol_target(weight_panel: pd.DataFrame,price_panel: pd.DataFrame,
                         target_vol: float = 0.15,window: int = 20,
                         max_scale: float = 1.0) -> pd.DataFrame:
    """
    组合级波动率控制。
    高波动阶段降低整体仓位，低波动阶段恢复。
    """
    ret = price_panel.pct_change()
    portfolio_ret = (weight_panel.shift(1).fillna(0) * ret).sum(axis=1)
    vol = portfolio_ret.rolling(window,min_periods=max(5,window//2)).std() * np.sqrt(252)
    scale = target_vol / vol.replace(0,np.nan)
    scale = scale.clip(lower=0,upper=max_scale).fillna(max_scale)
    return weight_panel.mul(scale,axis=0)


def drawdown_filter(weight_panel: pd.DataFrame,portfolio_returns: pd.Series,
                    levels=(0.05,0.10,0.20),
                    scales=(0.7,0.4,0.2)) -> pd.DataFrame:
    """
    根据组合历史回撤降低仓位。
    """
    equity = (1 + portfolio_returns.fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1

    scale = pd.Series(1.0,index=dd.index)
    scale.loc[dd <= -levels[0]] = scales[0]
    scale.loc[dd <= -levels[1]] = scales[1]
    scale.loc[dd <= -levels[2]] = scales[2]

    return weight_panel.mul(scale.reindex(weight_panel.index).fillna(1),axis=0)


def crowding_filter(weight_panel: pd.DataFrame,crowding_score: pd.Series,
                    threshold: float = 0.8,scale: float = 0.5) -> pd.DataFrame:
    """
    因子拥挤控制。
    crowding_score:
        越高代表越拥挤。
    """
    adjust = pd.Series(1.0,index=weight_panel.index)
    adjust.loc[crowding_score >= threshold] = scale
    return weight_panel.mul(adjust.reindex(weight_panel.index).fillna(1),axis=0)
