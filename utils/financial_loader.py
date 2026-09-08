"""原始财务读取与公告日可见的 PIT 财务事件转换。"""

import numpy as np
import pandas as pd

from utils.duckdb_manager import FINANCIAL_KEY_COLS, FINANCIAL_TABLES, get_conn
from utils.loader import load_trading_calendar


FINANCIAL_FIELD_SPECS = {
    "revenue_ttm": ("income", "revenue", "flow"),
    "operating_cost_ttm": ("income", "operating_cost", "flow"),
    "net_income_attributable_ttm": (
        "income", "net_income_attributable", "flow"
    ),
    "net_operating_cash_flow_ttm": (
        "cash_flow", "net_operating_cash_flow", "flow"
    ),
    "capex_ttm": ("cash_flow", "capex", "flow"),
    "total_assets": ("balance_sheet", "total_assets", "stock"),
    "total_liabilities": ("balance_sheet", "total_liabilities", "stock"),
    "total_equity": ("balance_sheet", "total_equity", "stock"),
    "equity_attributable": (
        "balance_sheet", "equity_attributable", "stock"
    ),
}
FINANCIAL_LOOKBACK_MONTHS = 24


def _table(statement: str) -> str:
    if statement not in FINANCIAL_TABLES:
        raise ValueError(f"unsupported financial statement: {statement}")
    return FINANCIAL_TABLES[statement]


def load_financial_statement(
    statement: str,
    symbols=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    table = _table(statement)
    where, params = [], []
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).split(".")[0].zfill(6) for s in symbols]
        if not symbols:
            return pd.DataFrame()
        where.append(f"symbol IN ({','.join('?' for _ in symbols)})")
        params.extend(symbols)
    if start_date is not None:
        where.append("period_end >= ?")
        params.append(start_date)
    if end_date is not None:
        where.append("period_end <= ?")
        params.append(end_date)

    conn = get_conn()
    try:
        df = conn.execute(
            f"SELECT * FROM {table} "
            + (f"WHERE {' AND '.join(where)} " if where else "")
            + "ORDER BY symbol, period_end, announce_date",
            params,
        ).fetchdf()
    finally:
        conn.close()
    for col in df:
        if col == "fetched_at" or col.casefold().endswith("_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def prepare_financial_history(
    df: pd.DataFrame,
    statement: str,
) -> pd.DataFrame:
    """标准化版本键，并令数据从公告后的下一个交易日起可见。"""
    _table(statement)
    missing = [col for col in FINANCIAL_KEY_COLS if col not in df]
    if missing:
        raise ValueError(f"missing financial keys: {missing}")

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    for col in ("period_end", "announce_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[FINANCIAL_KEY_COLS].isna().any(axis=1).any():
        raise ValueError("null financial key")

    calendar = load_trading_calendar().sort_values()
    positions = np.searchsorted(
        calendar.to_numpy(dtype="datetime64[ns]"),
        out["announce_date"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    out["available_date"] = pd.NaT
    valid = positions < len(calendar)
    out.loc[valid, "available_date"] = calendar.take(positions[valid]).to_numpy()
    return out.sort_values(FINANCIAL_KEY_COLS).reset_index(drop=True)


def _number(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _snapshot(
    versions: dict[pd.Timestamp, dict],
    specs: dict[str, tuple[str, str]],
) -> dict[str, float]:
    periods = sorted(versions)
    latest = periods[-1]
    result = {}
    quarter_index = {period: period.year * 4 + period.quarter for period in periods}
    period_by_quarter = {value: period for period, value in quarter_index.items()}
    trailing = periods[-4:]
    consecutive = (
        len(trailing) == 4
        and all(
            quarter_index[right] - quarter_index[left] == 1
            for left, right in zip(trailing, trailing[1:])
        )
    )

    for output, (source, kind) in specs.items():
        if kind == "stock":
            result[output] = _number(versions[latest].get(source))
            continue

        quarterly = []
        for period in trailing:
            cumulative = _number(versions[period].get(source))
            if period.quarter == 1:
                quarterly.append(cumulative)
                continue
            previous = period_by_quarter.get(quarter_index[period] - 1)
            prior = _number(versions[previous].get(source)) if previous else np.nan
            quarterly.append(
                cumulative - prior
                if pd.notna(cumulative) and pd.notna(prior)
                else np.nan
            )
        result[output] = (
            float(sum(quarterly))
            if consecutive and all(pd.notna(value) for value in quarterly)
            else np.nan
        )
    return result


def load_pit_financial_snapshot(
    as_of_date,
    fields,
    symbols=None,
    history_start=None,
) -> pd.DataFrame:
    """Return one latest visible PIT snapshot per symbol."""
    selected = [fields] if isinstance(fields, str) else list(fields)
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - set(FINANCIAL_FIELD_SPECS))
    if unknown:
        raise ValueError(f"unknown financial fields: {unknown}")
    if not selected:
        return pd.DataFrame()

    end_ts = pd.Timestamp(as_of_date)
    start_ts = pd.Timestamp(history_start) if history_start else end_ts
    if start_ts > end_ts:
        raise ValueError("history_start must not be after as_of_date")
    report_start = start_ts - pd.DateOffset(months=FINANCIAL_LOOKBACK_MONTHS)
    by_statement = {}
    for output in selected:
        statement, source, kind = FINANCIAL_FIELD_SPECS[output]
        by_statement.setdefault(statement, {})[output] = (source, kind)

    snapshots = []
    for statement, specs in by_statement.items():
        raw = load_financial_statement(
            statement,
            symbols=symbols,
            start_date=report_start,
            end_date=end_ts,
        )
        if raw.empty:
            continue
        frame = prepare_financial_history(raw, statement)
        frame = frame[
            frame["available_date"].notna()
            & frame["available_date"].le(end_ts)
            & frame["period_end"].dt.strftime("%m-%d").isin(
                ("03-31", "06-30", "09-30", "12-31")
            )
        ].copy()
        if frame.empty:
            continue
        sort_cols = [
            col for col in
            ("symbol", "period_end", "available_date", "announce_date", "fetched_at")
            if col in frame
        ]
        frame = (
            frame.sort_values(sort_cols)
            .drop_duplicates(["symbol", "period_end"], keep="last")
        )
        source_fields = list(
            dict.fromkeys(source for source, _ in specs.values())
        )
        records = []
        columns = ["period_end", *source_fields]
        for symbol, symbol_rows in frame.groupby("symbol", sort=False):
            visible = {
                period_end: dict(zip(source_fields, values))
                for period_end, *values in symbol_rows[
                    columns
                ].itertuples(index=False, name=None)
            }
            records.append({
                "symbol": symbol,
                **_snapshot(visible, specs),
            })
        snapshots.append(pd.DataFrame(records).set_index("symbol"))

    if not snapshots:
        return pd.DataFrame(columns=selected)
    return pd.concat(snapshots, axis=1).reindex(columns=selected).sort_index()


def load_pit_financial_events(start, end, fields, symbols=None) -> pd.DataFrame:
    """返回公告事件后的最新 PIT 值；每行是一个 symbol 的一次状态变化。"""
    selected = [fields] if isinstance(fields, str) else list(fields)
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - set(FINANCIAL_FIELD_SPECS))
    if unknown:
        raise ValueError(f"unknown financial fields: {unknown}")
    if not selected:
        return pd.DataFrame(columns=["available_date", "symbol"])

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must not be after end")
    report_start = start_ts - pd.DateOffset(months=FINANCIAL_LOOKBACK_MONTHS)
    records = []
    by_statement = {}
    for output in selected:
        statement, source, kind = FINANCIAL_FIELD_SPECS[output]
        by_statement.setdefault(statement, {})[output] = (source, kind)

    for statement, specs in by_statement.items():
        raw = load_financial_statement(
            statement,
            symbols=symbols,
            start_date=report_start,
            end_date=end_ts,
        )
        if raw.empty:
            continue
        frame = prepare_financial_history(raw, statement)
        frame = frame[
            frame["available_date"].notna()
            & frame["available_date"].le(end_ts)
            & frame["period_end"].dt.strftime("%m-%d").isin(
                ("03-31", "06-30", "09-30", "12-31")
            )
        ].copy()
        sort_cols = [
            col for col in
            ("symbol", "available_date", "announce_date", "fetched_at")
            if col in frame
        ]
        frame = frame.sort_values(sort_cols)

        source_fields = list(dict.fromkeys(source for source, _ in specs.values()))
        for symbol, symbol_rows in frame.groupby("symbol", sort=False):
            visible = {}
            current_date = None
            columns = ["available_date", "period_end", *source_fields]
            for row in symbol_rows[columns].itertuples(index=False, name=None):
                available_date, period_end, *values = row
                if current_date is not None and available_date != current_date:
                    # Before the requested panel starts, only its final visible
                    # state is needed. Emit that baseline when the first
                    # in-window change arrives, then retain every later change.
                    if current_date >= start_ts or available_date >= start_ts:
                        records.append({
                            "available_date": current_date,
                            "symbol": symbol,
                            "_statement": statement,
                            **_snapshot(visible, specs),
                        })
                visible[period_end] = dict(zip(source_fields, values))
                current_date = available_date
            if current_date is not None:
                records.append({
                    "available_date": current_date,
                    "symbol": symbol,
                    "_statement": statement,
                    **_snapshot(visible, specs),
                })

    if not records:
        return pd.DataFrame(
            columns=["available_date", "symbol", "_statement", *selected]
        )
    events = pd.DataFrame(records)
    return events.sort_values(
        ["available_date", "symbol", "_statement"]
    ).reset_index(drop=True)
