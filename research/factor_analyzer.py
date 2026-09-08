# research/factor_analyzer.py
"""
因子分析核心：只吃 factor panel + price panel，不关心因子怎么算出来的。
"""
import numpy as np
import pandas as pd


class FactorAnalyzer:
    def __init__(self, factor: pd.DataFrame, price: pd.DataFrame, tradable: pd.DataFrame = None,
                 return_clip: float = None, label: str = None):
        """
        label: 可选，这次分析的因子名/标签。多因子并排对比场景（如 scripts/run_factor.py）
        下用于区分 recorder 记录里哪条 ic_summary/group_diagnostics 属于哪个因子。
        单因子脚本可以不传，recorder 记录里 factor 字段会是 None。
        """
        idx = factor.index.union(price.index)
        cols = factor.columns.union(price.columns)
        self.price = price.reindex(index=idx, columns=cols)
        factor = factor.reindex(index=idx, columns=cols)
        self.return_clip = return_clip
        self.label = label

        if tradable is not None:
            self.tradable = tradable.reindex(index=idx, columns=cols).fillna(False)
            factor = factor.where(self.tradable)
        else:
            self.tradable = pd.DataFrame(True, index=idx, columns=cols)

        self.factor = factor


    @staticmethod
    def _at_dates(values, dates=None):
        """可选地选取固定观察日期，例如非重叠20日采样。"""
        if dates is None:
            return values
        return values.reindex(pd.DatetimeIndex(dates))

    def forward_return(self, period: int = 1) -> pd.DataFrame:
        """t 日因子预测 t -> t+period 收益，强制 shift 防止未来函数。
        return_clip: 若设置，将收益率截断在 [-clip, clip] 区间，避免极端个股
        （停牌复牌/科创板20%涨跌幅等）污染 IC 和分层收益等对极端值敏感的统计量。
        """
        fwd = self.price.pct_change(period).shift(-period)
        if self.return_clip is not None:
            fwd = fwd.clip(lower=-self.return_clip, upper=self.return_clip)
        return fwd

    def calc_ic(self, period: int = 1, method: str = "spearman",
                dates=None) -> pd.Series:
        """逐日IC；dates可传固定的非重叠观察日期。"""
        fwd = self.forward_return(period)
        ic = self.factor.corrwith(fwd, axis=1, method=method)
        return self._at_dates(ic, dates).dropna()

    def ic_summary(self, period: int = 1, method: str = "spearman", dates=None) -> dict:
        ic = self.calc_ic(period, method, dates=dates)
        if ic.std() == 0 or ic.empty:
            result = {"ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                      "ic_win_rate": np.nan, "t_stat": np.nan, "n_days": len(ic)}
        else:
            result = {
                "ic_mean": ic.mean(),
                "ic_std": ic.std(),
                "ir": ic.mean() / ic.std(),
                "ic_win_rate": (ic > 0).mean(),
                "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
                "n_days": len(ic),
            }

        from research.recorder import get_session
        get_session().add("ic_summary", {
            "factor": self.label,
            "data": {str(period): {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in result.items()}},
        })
        return result

    def cross_section_corr(self, other: pd.DataFrame, dates=None,
                           method: str = "spearman",
                           min_names: int = 30) -> pd.Series:
        """因子与另一宽表的逐日截面相关性，用于检查风格暴露。"""
        if min_names < 2:
            raise ValueError("min_names must be at least 2")

        other = other.reindex_like(self.factor)
        valid = self.factor.notna() & other.notna()
        corr = self.factor.corrwith(other, axis=1, method=method)
        corr = corr.where(valid.sum(axis=1) >= min_names)
        return self._at_dates(corr, dates).dropna()

    def coverage(self, dates=None) -> pd.DataFrame:
        """因子相对tradable mask的逐日覆盖率和样本数。"""
        valid_names = self.factor.notna().sum(axis=1)
        eligible_names = self.tradable.sum(axis=1)

        result = pd.DataFrame({
            "valid_names": valid_names,
            "eligible_names": eligible_names,
            "coverage": valid_names / eligible_names.replace(0, np.nan),
        })
        return self._at_dates(result, dates)

    def quantile(self, n_groups: int = 5) -> pd.DataFrame:
        """逐日按因子截面分组，返回 group 标签 panel（0 = 最低分, n_groups-1 = 最高分）"""
        ranks = self.factor.rank(axis=1, pct=True)
        groups = np.ceil(ranks * n_groups).clip(1, n_groups) - 1
        return groups

    def group_returns(self, period: int = 1, n_groups: int = 5) -> pd.DataFrame:
        """每组等权日收益（未做非重叠调整，period>1 时仅做定性参考）"""
        groups = self.quantile(n_groups)
        fwd = self.forward_return(period)
        result = {}
        for g in range(n_groups):
            result[f"G{g + 1}"] = fwd.where(groups == g).mean(axis=1)
        df = pd.DataFrame(result)
        df["long_short"] = df[f"G{n_groups}"] - df["G1"]
        return df

    def group_diagnostics(self, period: int = 10, n_groups: int = 5) -> pd.DataFrame:
        """
        各组 forward_return 的 median/mean 对比表，用于快速判断因子是否被极端值污染
        （median和mean方向一致 → 信号真实；背离 → 警惕极端值主导）。
        """
        groups = self.quantile(n_groups)
        fwd = self.forward_return(period)
        rows = []
        for g in range(n_groups):
            vals = fwd.where(groups == g).stack()
            rows.append({"group": f"G{g + 1}", "median": vals.median(), "mean": vals.mean()})
        df = pd.DataFrame(rows).set_index("group")

        from research.recorder import get_session
        get_session().add("group_diagnostics", {
            "factor": self.label,
            "data": {idx: {"median": round(float(row["median"]), 6), "mean": round(float(row["mean"]), 6)}
                     for idx, row in df.iterrows()},
        })
        return df

    def group_equity_curves(self, period: int = 1, n_groups: int = 5) -> pd.DataFrame:
        """
        各分层组的累计净值曲线（从1.0起步的复利曲线），用于判断分层效果是否
        在整个回测窗口内保持稳定，还是只在某一段时间内有效、其余时间失效。
        比 group_returns() 的静态均值更能反映因子表现的时间稳定性。
        """
        daily_returns = self.group_returns(period=1, n_groups=n_groups)
        daily_returns = daily_returns.drop(columns=["long_short"], errors="ignore")
        equity = (1 + daily_returns.fillna(0)).cumprod()

        summary = {}
        for col in equity.columns:
            series = equity[col]
            peak = series.cummax()
            mdd = (series / peak - 1).min()
            summary[col] = {
                "final_equity": round(float(series.iloc[-1]), 4),
                "peak_equity": round(float(series.max()), 4),
                "max_drawdown": round(float(mdd), 4),
            }
        quarterly = equity.resample("QE").last()
        monotonic_count = int((quarterly.iloc[:, -1] >= quarterly.iloc[:, 0]).sum()) if len(quarterly) > 0 else 0
        monotonicity = {
            "quarters_checked": len(quarterly),
            "quarters_monotonic": monotonic_count,
            "monotonicity_pct": round(monotonic_count / len(quarterly), 4) if len(quarterly) else None,
        }

        from research.recorder import get_session
        get_session().add("group_equity", {
            "factor": self.label,
            "summary": summary,
            "monotonicity": monotonicity,
        })
        return equity

    def group_weight(self, group: int,
                     n_groups: int = 5) -> pd.DataFrame:
        """指定分组的逐日等权权重，可直接喂给vector_backtest。"""
        if not 0 <= group < n_groups:
            raise ValueError(
                f"group must be in [0, {n_groups - 1}], got {group}"
            )

        members = self.quantile(n_groups).eq(group).astype(float)
        return members.div(
            members.sum(axis=1).replace(0, np.nan),
            axis=0,
        ).fillna(0)

    def top_group_weight(self, n_groups: int = 5) -> pd.DataFrame:
        return self.group_weight(n_groups - 1, n_groups)

    def bottom_group_weight(self, n_groups: int = 5) -> pd.DataFrame:
        return self.group_weight(0, n_groups)

    def turnover(self, n_groups: int = 5) -> pd.Series:
        groups = self.quantile(n_groups)
        top = groups == n_groups - 1
        prev = top.shift(1)
        changed = (top != prev) & (top | prev)
        return changed.sum(axis=1) / prev.sum(axis=1).replace(0, np.nan)

    def full_report(self, periods=(1, 5, 10, 20), n_groups: int = 5) -> dict:
        return {
            **{f"ic_period_{p}": self.ic_summary(p) for p in periods},
            "group_returns": self.group_returns(period=periods[0], n_groups=n_groups),
            "turnover_mean": self.turnover(n_groups).mean(),
        }