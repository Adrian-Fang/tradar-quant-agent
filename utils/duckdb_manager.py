"""utils/duckdb_manager.py — DuckDB 数据库表结构与读写管理"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)
DATA_PATH = Path(os.environ["DATA_PATH"]).expanduser()
DB_PATH = str(DATA_PATH / "tradar.duckdb")
HISTORY_COLS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
FINANCIAL_TABLES = {
    "income": "financial_income",
    "balance_sheet": "financial_balance_sheet",
    "cash_flow": "financial_cash_flow",
    "metrics": "financial_metrics",
    "shares": "financial_shares",
}
FINANCIAL_KEY_COLS = ["symbol", "period_end", "announce_date"]
FINANCIAL_CONFLICT_PATH = Path(DB_PATH).parent / "logs" / "financial_sync_conflicts.jsonl"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _ensure_financial_tables(conn: duckdb.DuckDBPyConnection) -> None:
    for table in FINANCIAL_TABLES.values():
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_quote_identifier(table)} (
                symbol VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                announce_date DATE NOT NULL,
                fetched_at TIMESTAMP NOT NULL,
                PRIMARY KEY (symbol, period_end, announce_date)
            )
        """)

def get_conn(readonly: bool = True) -> duckdb.DuckDBPyConnection: 
    return duckdb.connect(DB_PATH, read_only=readonly)

def init_database():
    conn = get_conn(readonly=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS universe (symbol VARCHAR PRIMARY KEY, name VARCHAR, industry VARCHAR,
        exchange VARCHAR(2), total_shares DOUBLE, circulating_shares DOUBLE, list_date DATE, update_time TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS history (symbol VARCHAR NOT NULL, date DATE NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE, PRIMARY KEY (symbol, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS etf (symbol VARCHAR NOT NULL, date DATE NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE, PRIMARY KEY (symbol, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS market_index (symbol VARCHAR NOT NULL, date DATE NOT NULL,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE, PRIMARY KEY (symbol, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hist_ext (symbol VARCHAR NOT NULL, date DATE NOT NULL, is_st BOOLEAN, 
        is_trading BOOLEAN, turnover_rate DOUBLE, pe DOUBLE, pb DOUBLE, market_cap DOUBLE,
        change_pct DOUBLE, PRIMARY KEY (symbol, date))""")
    conn.execute("ALTER TABLE hist_ext ADD COLUMN IF NOT EXISTS change_pct DOUBLE")
    conn.execute("""CREATE TABLE IF NOT EXISTS ex_factors (symbol VARCHAR NOT NULL, date DATE NOT NULL, ex_factor DOUBLE, PRIMARY KEY (symbol, date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS industry (symbol VARCHAR NOT NULL, lv1_code VARCHAR, lv1_name VARCHAR, 
        lv2_code VARCHAR, lv2_name VARCHAR, lv3_code VARCHAR, lv3_name VARCHAR, updated_at TIMESTAMP, PRIMARY KEY (symbol))""")
    _ensure_financial_tables(conn)
    conn.close()
    print(f"✅ 数据库初始化完成: {DB_PATH}")


def load_local_stocks() -> pd.DataFrame:
    conn = get_conn()
    try:
        stocks = conn.execute("SELECT symbol, name, industry, exchange, total_shares, circulating_shares, list_date FROM universe WHERE exchange IN ('SH', 'SZ') ORDER BY symbol").fetchdf()
    finally: conn.close()
    if stocks.empty: raise RuntimeError("universe 表为空，请先更新股票元数据")
    return stocks


def upsert_stocks(df: pd.DataFrame):
    if "update_time" not in df.columns: df["update_time"] = pd.Timestamp.now()
    conn = get_conn(readonly=False)
    try:
        conn.register('_tmp_universe', df)
        conn.execute("""
            INSERT INTO universe (symbol, name, exchange, total_shares, circulating_shares, list_date, update_time)
            SELECT symbol, name, exchange, total_shares, circulating_shares, list_date, update_time
            FROM _tmp_universe
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                total_shares = EXCLUDED.total_shares,
                circulating_shares = EXCLUDED.circulating_shares,
                list_date = EXCLUDED.list_date,
                update_time = EXCLUDED.update_time
        """)
        print(f"✅ 股票元数据重写完成: {len(df)} 行")
    finally:
        conn.close()

def _upsert_df(table: str, df: pd.DataFrame, cols: list, conflict_cols: list):
    df_clean = df[cols].astype(object).where(pd.notna(df[cols]), None)
    conn = get_conn(readonly=False)
    conn.register("_tmp_df", df_clean)
    try:
        updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict_cols)
        conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT * FROM _tmp_df ON CONFLICT({', '.join(conflict_cols)}) DO UPDATE SET {updates}")
    finally: conn.close()
    print(f"✅ {table} 重写完成: {len(df)} 行")

def update_factors(df: pd.DataFrame):
    """增量写入复权因子表。"""
    if df.empty:
        print("⚠️ ex_factors 为空，跳过")
        return
    _upsert_df(
        "ex_factors",
        df,
        ["symbol", "date", "ex_factor"],
        ["symbol", "date"],
    )


def upsert_history(history_df: pd.DataFrame): _upsert_df("history", history_df, HISTORY_COLS, ["symbol", "date"])
def upsert_etf(df: pd.DataFrame): _upsert_df("etf", df, HISTORY_COLS, ["symbol", "date"])
def upsert_market_index(df: pd.DataFrame): _upsert_df("market_index", df, HISTORY_COLS, ["symbol", "date"])


def upsert_realtime_hist_ext(df: pd.DataFrame) -> None:
    """Update only quote-derived fields without clearing Baostock fundamentals."""
    if df.empty:
        return
    cols = ["symbol", "date", "is_trading", "turnover_rate", "change_pct"]
    clean = df[cols].astype(object).where(pd.notna(df[cols]), None)
    conn = get_conn(readonly=False)
    try:
        conn.register("_tmp_realtime_ext", clean)
        conn.execute("""
            INSERT INTO hist_ext (symbol, date, is_trading, turnover_rate, change_pct)
            SELECT symbol, date, is_trading, turnover_rate, change_pct
            FROM _tmp_realtime_ext
            ON CONFLICT (symbol, date) DO UPDATE SET
                is_trading = EXCLUDED.is_trading,
                turnover_rate = EXCLUDED.turnover_rate,
                change_pct = EXCLUDED.change_pct
        """)
    finally:
        conn.close()


def _missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _append_financial_conflicts(conflicts: list[dict]) -> None:
    if not conflicts:
        return
    FINANCIAL_CONFLICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = set()
    if FINANCIAL_CONFLICT_PATH.exists():
        for line in FINANCIAL_CONFLICT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                old.add(json.loads(line)["conflict_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    with FINANCIAL_CONFLICT_PATH.open("a", encoding="utf-8") as handle:
        for item in conflicts:
            identity = {
                key: item[key]
                for key in (
                    "statement", "symbol", "period_end", "announce_date", "rows"
                )
            }
            conflict_id = hashlib.sha256(_dump(identity).encode()).hexdigest()
            if conflict_id in old:
                continue
            handle.write(_dump({"conflict_id": conflict_id, **item}) + "\n")
            old.add(conflict_id)


def _normalize_financial_statement(statement: str, df: pd.DataFrame):
    data = df.copy()
    data.columns = [str(c) for c in data.columns]
    rename = {
        c: c.casefold()
        for c in data.columns
        if c.casefold() in FINANCIAL_KEY_COLS
    }
    data = data.rename(columns=rename)
    missing = [c for c in FINANCIAL_KEY_COLS if c not in data]
    if missing:
        raise ValueError(f"missing financial keys: {missing}")
    data["symbol"] = data["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    for col in ("period_end", "announce_date"):
        data[col] = pd.to_datetime(data[col], errors="coerce").dt.date
    if data[FINANCIAL_KEY_COLS].isna().any(axis=1).any():
        raise ValueError("null financial key")
    data = data.drop_duplicates()

    payload = [c for c in data if c not in FINANCIAL_KEY_COLS + ["fetched_at"]]
    clean, conflicts = [], []
    for key, group in data.groupby(FINANCIAL_KEY_COLS, sort=False):
        merged = dict(zip(FINANCIAL_KEY_COLS, key))
        bad = []
        for col in payload:
            values = {_dump(v): v for v in group[col] if not _missing(v)}
            if len(values) > 1:
                bad.append(col)
                merged[col] = None
            else:
                merged[col] = next(iter(values.values()), None)
        if bad:
            rows = [
                {c: (None if _missing(v) else v) for c, v in row.items()}
                for row in group[FINANCIAL_KEY_COLS + payload].to_dict("records")
            ]
            conflicts.append({
                "statement": statement,
                "symbol": str(key[0]).zfill(6),
                "period_end": str(key[1]),
                "announce_date": str(key[2]),
                "conflicting_fields": bad,
                "rows": rows,
            })
        clean.append(merged)
    result = pd.DataFrame(clean, columns=FINANCIAL_KEY_COLS + payload)
    result["fetched_at"] = pd.Timestamp.now()
    return result, conflicts


def upsert_financial_statement(statement: str, df: pd.DataFrame) -> None:
    if statement not in FINANCIAL_TABLES:
        raise ValueError(f"unsupported financial statement: {statement}")
    if df is None or df.empty:
        return
    data, conflicts = _normalize_financial_statement(statement, df)
    _append_financial_conflicts(conflicts)
    if not data.empty:
        table, temp = FINANCIAL_TABLES[statement], "_tmp_financial"
        conn = get_conn(readonly=False)
        try:
            _ensure_financial_tables(conn)
            conn.register(temp, data)
            types = {
                row[0]: row[1]
                for row in conn.execute(
                    f"DESCRIBE SELECT * FROM {temp}"
                ).fetchall()
            }
            existing = {r[1].casefold() for r in conn.execute(
                f"PRAGMA table_info({_quote_identifier(table)})"
            ).fetchall()}
            for col in data:
                if col.casefold() not in existing:
                    kind = types[col] if data[col].notna().any() else "VARCHAR"
                    conn.execute(
                        f"ALTER TABLE {_quote_identifier(table)} "
                        f"ADD COLUMN {_quote_identifier(col)} {kind}"
                    )
            cols = ", ".join(_quote_identifier(c) for c in data)
            keys = ", ".join(_quote_identifier(c) for c in FINANCIAL_KEY_COLS)
            updates = ", ".join(
                f"{_quote_identifier(c)}=EXCLUDED.{_quote_identifier(c)}"
                for c in data if c not in FINANCIAL_KEY_COLS
            )
            conn.execute(
                f"INSERT INTO {_quote_identifier(table)} ({cols}) "
                f"SELECT {cols} FROM {temp} ON CONFLICT ({keys}) "
                f"DO UPDATE SET {updates}"
            )
        finally:
            conn.close()
