# research/report.py
"""格式化打印 IC/回测指标，避免每个研究脚本自己写打印逻辑。"""
import unicodedata
import pandas as pd
from research.metrics import calc_metrics
from research.recorder import get_session


def _fmt_pct(x, digits=2):
    return f"{x * 100:.{digits}f}%" if pd.notna(x) else "N/A"


def _fmt_num(x, digits=3):
    return f"{x:.{digits}f}" if pd.notna(x) else "N/A"


def _dw(s: str) -> int:
    """Display width of a string (CJK full-width/wide chars count as 2)."""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))


def _print_aligned_table(df: pd.DataFrame) -> None:
    """
    CJK-aware DataFrame printer.
    pandas to_string() uses len() which under-counts double-width CJK characters,
    producing misaligned columns when headers or index labels contain Chinese text.
    Index column: left-aligned.  Data columns: right-aligned.
    """
    idx_name = str(df.index.name) if df.index.name else ""
    idx_strs = [str(i) for i in df.index]
    col_strs = {col: [str(v) for v in df[col]] for col in df.columns}

    idx_w  = max(_dw(idx_name), *(_dw(s) for s in idx_strs))
    col_ws = {col: max(_dw(str(col)), *(_dw(s) for s in col_strs[col])) for col in df.columns}

    SEP = "  "

    def _l(s, w):  # left-align (right-pad)
        s = str(s); return s + " " * max(0, w - _dw(s))

    def _r(s, w):  # right-align (left-pad)
        s = str(s); return " " * max(0, w - _dw(s)) + s

    header = _l(idx_name, idx_w) + SEP + SEP.join(_r(c, col_ws[c]) for c in df.columns)
    total_w = idx_w + len(SEP) * len(df.columns) + sum(col_ws[c] for c in df.columns)
    print(header)
    print("─" * total_w)
    for row_i, idx_str in enumerate(idx_strs):
        print(_l(idx_str, idx_w) + SEP + SEP.join(_r(col_strs[c][row_i], col_ws[c]) for c in df.columns))


def ic_table(analyzer, periods=(1, 5, 10, 20)) -> pd.DataFrame:
    """多周期IC汇总表"""
    rows = []
    for p in periods:
        s = analyzer.ic_summary(period=p)
        rows.append({
            "period": p, "ic_mean": s["ic_mean"], "ic_std": s["ic_std"],
            "ir": s["ir"], "win_rate": s["ic_win_rate"],
            "t_stat": s["t_stat"], "n_days": s["n_days"],
        })
    df = pd.DataFrame(rows).set_index("period")
    return df.round(3)


def print_ic_table(analyzer, periods=(1, 5, 10, 20)):
    df = ic_table(analyzer, periods)
    print("\n── IC ──")
    print(df.to_string())


def print_group_equity_summary(analyzer, n_groups: int = 5):
    equity = analyzer.group_equity_curves(n_groups=n_groups)
    print("\n── 分层 ──")
    rows = []
    for col in equity.columns:
        series = equity[col]
        peak = series.cummax()
        mdd = (series / peak - 1).min()
        rows.append({
            "group": col,
            "累计净值": series.iloc[-1],
            "最高点": series.max(),
            "最大回撤": mdd,
        })
    df = pd.DataFrame(rows).set_index("group")
    df["累计净值"] = df["累计净值"].round(3)
    df["最高点"] = df["最高点"].round(3)
    df["最大回撤"] = (df["最大回撤"] * 100).round(1).astype(str) + "%"
    _print_aligned_table(df)

    # 季度单调性：每季度末 G5 >= G1
    quarterly = equity.resample("QE").last()
    n_groups_actual = len(equity.columns)
    monotonic_count = sum(
        1 for i in range(len(quarterly)) if quarterly.iloc[i].iloc[-1] >= quarterly.iloc[i].iloc[0]
    )
    print(f"季度单调: {monotonic_count}/{len(quarterly)} ({monotonic_count / len(quarterly):.0%})")


def performance_table(results: dict, benchmark: pd.Series = None) -> pd.DataFrame:
    """
    results: {name: BacktestResult} 字典
    输出一张对比表，每行一个变体（Baseline / +Regime / ...）
    """
    rows = []
    for name, result in results.items():
        m = calc_metrics(result.returns, benchmark=benchmark)
        rows.append({
            "variant": name,
            "annual_return": m["annual_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "beta": m.get("beta"),
            "alpha": m.get("alpha"),
        })
    df = pd.DataFrame(rows).set_index("variant")
    for col in ("annual_return", "max_drawdown", "alpha"):
        df[col] = (df[col] * 100).round(1).astype(str) + "%"
    df["sharpe"] = df["sharpe"].round(2)
    df["beta"] = df["beta"].round(2)
    return df


def print_performance_table(results: dict, benchmark: pd.Series = None, title="Performance Comparison"):
    print(f"\n── {title} ──")
    df = performance_table(results, benchmark)
    _print_aligned_table(df)

    for variant_name, result in results.items():
        m = calc_metrics(result.returns, benchmark=benchmark)
        weight = result.weights
        daily_holdings = (weight > 0).sum(axis=1)
        changed = (weight.diff().abs().sum(axis=1) > 1e-9)
        n_rebalances = int(changed.sum())
        avg_interval = round(len(weight) / n_rebalances, 1) if n_rebalances > 0 else None
        get_session().add("backtest", {
            "title": title,
            "variant": variant_name,
            "top_n": int(daily_holdings[daily_holdings > 0].median()) if daily_holdings.max() > 0 else 0,
            "rebalance_freq_days": avg_interval,
            "metrics": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in m.items()},
        })


def print_performance(metrics: dict, title: str = "Performance"):
    """单个变体的详细打印（旧接口，仍保留供单次结果查看）"""
    print(f"\n── {title} " + "─" * max(1, 40 - len(title)))
    rows = [
        ("Total return", _fmt_pct(metrics.get("total_return"))),
        ("Annual return", _fmt_pct(metrics.get("annual_return"))),
        ("Annual vol", _fmt_pct(metrics.get("annual_vol"))),
        ("Sharpe", _fmt_num(metrics.get("sharpe"), 2)),
        ("Max drawdown", _fmt_pct(metrics.get("max_drawdown"))),
        ("Calmar", _fmt_num(metrics.get("calmar"), 2)),
        ("Win rate", _fmt_pct(metrics.get("win_rate"))),
    ]
    if "beta" in metrics:
        rows += [
            ("Beta", _fmt_num(metrics.get("beta"), 2)),
            ("Alpha (annual)", _fmt_pct(metrics.get("alpha"))),
            ("Benchmark annual return", _fmt_pct(metrics.get("benchmark_annual_return"))),
            ("Excess annual return", _fmt_pct(metrics.get("excess_annual_return"))),
        ]
    rows.append(("Trading days", str(metrics.get("n_days"))))

    label_width = max(len(r[0]) for r in rows) + 2
    for label, val in rows:
        print(f"  {label:<{label_width}}{val}")