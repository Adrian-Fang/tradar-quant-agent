"""统一的数据加载入口。价格/基本面/指数/元数据，只在这里决定一次。"""

from functools import lru_cache
import os
import pandas as pd
from utils.duckdb_manager import get_conn, DB_PATH, DATA_PATH
from pathlib import Path

CACHE_DIR = DATA_PATH / "cache"
CACHE_PATH = CACHE_DIR / "cum_factor.parquet"

INDEX_CODE_ALIASES = {
    # canonical: 真实存在于 market_index 表里的数字代码
    "000001": "000001",
    "000300": "000300",
    "000852": "000852",
    "000905": "000905",
    "399006": "399006",
    "000688": "000688",
    # 常见字母别名
    "HS300": "000300",
    "SH": "000001",
    "ZZ500": "000905",
    "ZZ1000": "000852",
    "CYB": "399006",
    "STAR": "000688",
    # 常见中文别名
    "沪深300": "000300",
    "上证": "000001",
    "上证指数": "000001",
    "中证500": "000905",
    "中证1000": "000852",
    "创业板": "399006",
    "科创板": "000688",
}

def load_trading_calendar() -> pd.DatetimeIndex:
    """全局交易日历（history 表去重日期升序），用于跨窗口一致的交易日序号锚定。"""
    con = get_conn()
    try:
        df = con.execute("SELECT DISTINCT date FROM history ORDER BY date").fetchdf()
    finally:
        con.close()
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    return CACHE_PATH.stat().st_mtime >= Path(DB_PATH).stat().st_mtime

def _compute_full_cum_factor(price_table: str = "history", symbols=None) -> pd.DataFrame:
    """扫描全历史，计算全量累计复权因子（不做日期过滤）。
    用 merge_asof(direction="backward") 而不是精确日期 join：除权事件生效
    不要求事件当天恰好是交易日（可能停牌），只需要找每个交易日之前最近一次
    已发生的事件即可。
    """
    params = []
    symbol_filter = ""
    if symbols is not None:
        symbol_filter = f" WHERE symbol IN ({','.join('?' for _ in symbols)})"
        params.extend(symbols)
    con = get_conn()
    try:
        events = con.execute(f"""
            SELECT symbol, date, ex_factor FROM ex_factors{symbol_filter} ORDER BY symbol, date
        """, params).df()
        trade_dates = con.execute(
            f"SELECT DISTINCT symbol, date FROM {price_table}{symbol_filter}", params
        ).df()
    finally:
        con.close()

    events["date"] = pd.to_datetime(events["date"])
    events["symbol"] = events["symbol"].astype(str).str.zfill(6)
    events = events.sort_values(["symbol", "date"])
    events["cum_factor"] = events.groupby("symbol")["ex_factor"].cumprod()

    trade_dates["date"] = pd.to_datetime(trade_dates["date"])
    trade_dates["symbol"] = trade_dates["symbol"].astype(str).str.zfill(6)
    trade_dates = trade_dates.sort_values(["date", "symbol"])
    events_sorted = events[["symbol", "date", "cum_factor"]].sort_values(["date", "symbol"])

    merged = pd.merge_asof(
        trade_dates, events_sorted, on="date", by="symbol", direction="backward"
    )
    merged["cum_factor"] = merged["cum_factor"].fillna(1.0)
    return merged[["date", "symbol", "cum_factor"]]


def _load_full_cum_factor(force_refresh: bool = False, symbols=None) -> pd.DataFrame:
    normalized = None if symbols is None else [str(symbol).strip().split(".")[0].zfill(6) for symbol in symbols if str(symbol).strip()]
    if not force_refresh and _cache_is_fresh():
        con = get_conn()
        try:
            if normalized:
                placeholders = ",".join("?" for _ in normalized)
                return con.execute(
                    f"SELECT date, symbol, cum_factor FROM read_parquet(?) WHERE symbol IN ({placeholders})",
                    [str(CACHE_PATH), *normalized],
                ).fetchdf()
            return con.execute("SELECT date, symbol, cum_factor FROM read_parquet(?)", [str(CACHE_PATH)]).fetchdf()
        finally:
            con.close()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = _compute_full_cum_factor(symbols=normalized)
    if normalized:
        return df
    con = get_conn()
    try:
        con.register("_cum_factor_cache", df)
        escaped_path = str(CACHE_PATH).replace("'", "''")
        con.execute(
            f"COPY _cum_factor_cache TO '{escaped_path}' (FORMAT PARQUET)"
        )
    finally:
        con.close()
    return df


# ── 价格 ──────────────────────────────────────────────────────────────────────

def _apply_price_adjustment(
    raw: pd.DataFrame,
    cum_factor_all: pd.DataFrame,
    adjust: str,
) -> pd.DataFrame:
    """股票和 ETF 共用的运行时复权数学逻辑。"""
    df = raw.merge(cum_factor_all, on=["symbol", "date"], how="left")
    df["cum_factor"] = df["cum_factor"].fillna(1.0)
    factor = df["cum_factor"]
    if adjust == "forward":
        factor = factor / df.groupby("symbol")["cum_factor"].transform("last")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    return df.drop(columns=["cum_factor"])


def load_prices(start: str, end: str, adjust: str = "forward", force_refresh_cache: bool = False, symbols=None) -> pd.DataFrame:
    """价格数据, MultiIndex(date, symbol)。
     history 存不复权原始数据，通过 ex_factors.ex_factor 累乘计算复权价。
    累计复权因子缓存在共享目录的 cum_factor.parquet，按 DB mtime 判断是否过期。

    adjust='none'    -> 不复权
    adjust='forward' -> 前复权: price × cum_factor / latest_cum_factor（按窗口内最后一天归一化）
    adjust='backward'-> 后复权: price × cum_factor（从上市第一天累计，不受窗口影响）
    force_refresh_cache: 强制忽略缓存重新计算（怀疑缓存和最新数据不一致时用）
    symbols: 可选股票代码过滤，保持默认全市场行为。
    """
    if adjust not in {"none", "forward", "backward"}:
        raise ValueError(f"unknown adjust: {adjust}")

    params = [start, end]
    symbol_filter = ""
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        normalized = [str(symbol).strip().split(".")[0].zfill(6) for symbol in symbols if str(symbol).strip()]
        if not normalized:
            index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
            return pd.DataFrame(index=index, columns=["open", "high", "low", "close", "volume", "amount"], dtype=float)
        symbol_filter = f" AND symbol IN ({','.join('?' for _ in normalized)})"
        params.extend(normalized)

    con = get_conn()
    try:
        raw = con.execute(f"""
            SELECT symbol, date, open, high, low, close, volume, amount
            FROM history
            WHERE date >= ? AND date <= ?{symbol_filter}
            ORDER BY date, symbol
        """, params).df()
    finally:
        con.close()

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["date"] = pd.to_datetime(raw["date"])

    if adjust == "none":
        df = raw
    else:
        cum_factor_all = _load_full_cum_factor(
            force_refresh=force_refresh_cache,
            symbols=normalized if symbols is not None else None,
        )
        cum_factor_all["date"] = pd.to_datetime(cum_factor_all["date"])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        cum_factor_all = cum_factor_all[
            (cum_factor_all["date"] >= start_ts) & (cum_factor_all["date"] <= end_ts)
        ]
        df = _apply_price_adjustment(raw, cum_factor_all, adjust)

    columns = ["open", "high", "low", "close", "volume", "amount"]
    return df.set_index(["date", "symbol"])[columns].sort_index()


def load_etf_prices(
    start: str,
    end: str,
    adjust: str = "forward",
    symbols=None,
) -> pd.DataFrame:
    """ETF 价格，返回结构和复权口径与 load_prices 一致。"""
    if adjust not in {"none", "forward", "backward"}:
        raise ValueError(f"unknown adjust: {adjust}")

    params = [start, end]
    symbol_filter = ""
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        normalized = [
            str(symbol).strip().split(".")[0].zfill(6)
            for symbol in symbols
            if str(symbol).strip()
        ]
        if not normalized:
            index = pd.MultiIndex.from_arrays(
                [[], []], names=["date", "symbol"]
            )
            return pd.DataFrame(
                index=index,
                columns=["open", "high", "low", "close", "volume", "amount"],
                dtype=float,
            )
        symbol_filter = f" AND symbol IN ({','.join('?' for _ in normalized)})"
        params.extend(normalized)

    con = get_conn()
    try:
        raw = con.execute(f"""
            SELECT symbol, date, open, high, low, close, volume, amount
            FROM etf
            WHERE date >= ? AND date <= ?{symbol_filter}
            ORDER BY date, symbol
        """, params).df()
    finally:
        con.close()

    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["date"] = pd.to_datetime(raw["date"])
    if adjust == "none" or raw.empty:
        df = raw
    else:
        factors = _compute_full_cum_factor("etf")
        factors["date"] = pd.to_datetime(factors["date"])
        factors = factors[
            factors["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        df = _apply_price_adjustment(raw, factors, adjust)

    columns = ["open", "high", "low", "close", "volume", "amount"]
    return df.set_index(["date", "symbol"])[columns].sort_index()

# ── 基本面 ────────────────────────────────────────────────────────────────────

def load_fundamentals(start: str, end: str) -> pd.DataFrame:
    """基本面快照 —— 永远从 hist_ext 取，不受复权/后续覆盖影响。
    exchange_change_pct 是交易所实际涨跌幅（百分数），只用于交易状态判断。
    """
    con = get_conn()
    try:
        df = con.execute("""
            SELECT symbol, date, is_st, is_trading, turnover_rate, market_cap, pe, pb,
                   change_pct AS exchange_change_pct
            FROM hist_ext
            WHERE date >= ? AND date <= ?
            ORDER BY date, symbol
        """, [start, end]).df()
    finally:
        con.close()

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "symbol"]).sort_index()


def merge_price_fundamental(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """价格 + 基本面 left join"""
    return prices.join(fundamentals, how="left")


def load_history_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """history 表日期范围"""
    con = get_conn()
    try:
        row = con.execute("SELECT MIN(date), MAX(date) FROM history").fetchone()
    finally:
        con.close()

    if not row or row[0] is None or row[1] is None:
        raise ValueError("history table is empty")

    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


# ── 指数 ──────────────────────────────────────────────────────────────────────

def resolve_index_code(code: str) -> str:
    """Normalize index code aliases to market_index.symbol 里实际存在的数字代码"""
    key = str(code).strip()
    if not key:
        raise ValueError("index code cannot be empty")
    if key in INDEX_CODE_ALIASES:
        return INDEX_CODE_ALIASES[key]
    key_u = key.upper()
    return INDEX_CODE_ALIASES.get(key_u, key_u)

def load_index(code: str, start_date=None, end_date=None) -> pd.Series:
    """指数收盘价序列。支持别名: HS300/SH/ZZ500/CYB/000300/000001/沪深300 等。
    market_index 表实际存储的是 symbol 列（数字代码），不是 code。
    """
    db_code = resolve_index_code(code)
    conditions = ["symbol = ?"]
    params = [db_code]
    if start_date is not None:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("date <= ?")
        params.append(end_date)

    where_clause = "WHERE " + " AND ".join(conditions)
    query = f"SELECT date, close FROM market_index {where_clause} ORDER BY date"

    con = get_conn()
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(
            f"index code '{code}' resolved to '{db_code}' but no data found in market_index"
        )

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(db_code)


# ── 元数据 ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_stock_info() -> pd.DataFrame:
    """股票静态信息。symbol 始终是六位字符串。"""
    con = get_conn()
    try:
        df = con.execute("""
            SELECT symbol, list_date, exchange
            FROM universe
            ORDER BY symbol
        """).fetchdf()
    finally:
        con.close()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["list_date"] = pd.to_datetime(df["list_date"])
    return df


def load_stocks() -> pd.DataFrame:
    """股票元数据: symbol index → name, lv1/lv2/lv3 行业（来自 industry 表，东方财富行业板块）。
    universe.industry 是旧字段，industry 表缺失时用它兜底，避免个别未同步的股票丢失行业信息。
    """
    con = get_conn()
    try:
        df = con.execute("""
            SELECT u.symbol, u.name, u.industry AS legacy_industry,
                   i.lv1_code, i.lv1_name, i.lv2_code, i.lv2_name, i.lv3_code, i.lv3_name
            FROM universe u
            LEFT JOIN industry i ON u.symbol = i.symbol
        """).fetchdf()
    finally:
        con.close()

    df["name"] = df["name"].fillna("Unknown")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["lv1_name"] = df["lv1_name"].fillna(df["legacy_industry"]).fillna("Unknown")
    df["lv2_name"] = df["lv2_name"].fillna("Unknown")
    df["lv3_name"] = df["lv3_name"].fillna("Unknown")
    df = df.drop(columns=["legacy_industry"])
    out = df.set_index("symbol")
    out["industry"] = out["lv1_name"]
    return out


def load_industry_map(level: int = 1) -> pd.Series:
    """symbol -> 行业名称的映射（用于给 factor panel 分组聚合）。
    level: 1/2/3，对应 lv1_name / lv2_name / lv3_name（东方财富行业板块粒度，1最粗）。
    """
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2, or 3, got {level}")
    col = f"lv{level}_name"
    con = get_conn()
    try:
        df = con.execute(f"SELECT symbol, {col} FROM industry WHERE {col} IS NOT NULL").fetchdf()
    finally:
        con.close()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df.set_index("symbol")[col]
