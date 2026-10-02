from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pandas as pd

from research import panel
from utils import loader


class LimitMoveMaskTests(unittest.TestCase):
    def test_shared_cache_contract_and_explicit_sandbox_opt_in(self):
        for flag in (None, "0", "1"):
            for state in ("missing", "stale", "force_refresh", "fresh"):
                with self.subTest(flag=flag, cache=state), tempfile.TemporaryDirectory() as directory:
                    database = str(Path(directory) / "prices.duckdb")
                    cache_dir = Path(directory) / "cache"
                    cache = cache_dir / "cum_factor.parquet"
                    con = duckdb.connect(database)
                    con.execute("CREATE TABLE history(symbol VARCHAR, date DATE, close DOUBLE)")
                    con.execute("INSERT INTO history VALUES ('600000','2024-12-31',5), ('600000','2025-01-02',10)")
                    con.execute("CREATE TABLE ex_factors(symbol VARCHAR, date DATE, ex_factor DOUBLE)")
                    con.execute("INSERT INTO ex_factors VALUES ('600000','2024-01-01',2)")
                    if state != "missing":
                        cache_dir.mkdir()
                        con.execute("COPY (SELECT symbol,date,1.0 cum_factor FROM history) TO ? (FORMAT PARQUET)", [str(cache)])
                    con.close()
                    if cache.exists():
                        timestamp = 0 if state == "stale" else Path(database).stat().st_mtime + 60
                        os.utime(cache, (timestamp, timestamp))
                    before = cache.read_bytes() if cache.exists() else None
                    with patch.object(loader, "get_conn", side_effect=lambda: duckdb.connect(database)), patch.object(
                        loader, "DB_PATH", database
                    ), patch.object(loader, "CACHE_DIR", cache_dir), patch.object(loader, "CACHE_PATH", cache), patch.dict(os.environ):
                        if flag is None:
                            os.environ.pop("TRADAR_EXPERIMENT_SANDBOX", None)
                        else:
                            os.environ["TRADAR_EXPERIMENT_SANDBOX"] = flag
                        result = loader.load_prices("2025-01-02", "2025-01-02", adjust="backward",
                                                    fields=["close"], force_refresh_cache=state == "force_refresh")
                    self.assertEqual(result.iloc[0, 0], 10 if state == "fresh" else 20)
                    if flag == "1":
                        self.assertEqual(cache.read_bytes() if cache.exists() else None, before)
                    else:
                        self.assertTrue(cache.is_file())
                        con = duckdb.connect()
                        factors = con.execute("SELECT date,cum_factor FROM read_parquet(?) ORDER BY date", [str(cache)]).fetchall()
                        con.close()
                        self.assertEqual(len(factors), 2)  # Shared cache includes dates outside the requested window.
                        self.assertEqual([row[1] for row in factors], [1, 1] if state == "fresh" else [2, 2])

    def test_cold_window_adjustment_matches_canonical_math_without_full_history_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "prices.duckdb")
            con = duckdb.connect(database)
            con.execute("CREATE TABLE history(symbol VARCHAR, date DATE, close DOUBLE, volume DOUBLE)")
            con.execute("CREATE TABLE ex_factors(symbol VARCHAR, date DATE, ex_factor DOUBLE)")
            con.execute("""INSERT INTO history VALUES
                ('600000','2026-01-09',10,100), ('600000','2026-01-12',11,110),
                ('600001','2026-01-09',20,200), ('600001','2026-01-12',21,210)
            """)
            # Events before the window/on a weekend must still compound.
            con.execute("""INSERT INTO ex_factors VALUES
                ('600000','2000-01-01',2), ('600000','2026-01-10',1.5),
                ('600001','2026-01-10',NULL)
            """)
            raw = con.execute("SELECT * FROM history ORDER BY date,symbol").df()
            raw["date"] = pd.to_datetime(raw["date"])
            con.close()
            with patch.object(loader, "get_conn", side_effect=lambda: duckdb.connect(database)), patch.object(
                loader, "_cache_is_fresh", return_value=False
            ), patch.object(loader, "CACHE_PATH", Path(directory) / "missing.parquet"), patch.dict(
                os.environ, {"TRADAR_EXPERIMENT_SANDBOX": "1"}
            ):
                factors = loader._compute_full_cum_factor()
                for adjust in ("backward", "forward"):
                    for symbols in (None, ["600000"], ["not_present"]):
                        with self.subTest(adjust=adjust, symbols=symbols):
                            expected_raw = raw if symbols is None else raw[raw.symbol.isin(symbols)]
                            expected = loader._apply_price_adjustment(expected_raw, factors, adjust)
                            expected = expected.set_index(["date", "symbol"])[["close", "volume"]].sort_index()
                            with patch.object(loader, "_compute_full_cum_factor", side_effect=AssertionError("full-history materialization")):
                                actual = loader.load_prices("2026-01-09", "2026-01-12", adjust=adjust,
                                                            fields=["close", "volume"], symbols=symbols)
                            pd.testing.assert_frame_equal(actual, expected)
                self.assertFalse(loader.CACHE_PATH.exists())

    def test_cold_short_window_does_not_materialize_large_history(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "prices.duckdb")
            con = duckdb.connect(database)
            con.execute("""CREATE TABLE history AS
                SELECT lpad((i % 100)::VARCHAR,6,'0') symbol,
                       DATE '2000-01-01' + (i // 100)::INTEGER date,
                       10.0::DOUBLE AS close FROM range(100000) t(i)""")
            con.execute("INSERT INTO history VALUES ('000000','2026-01-05',20)")
            con.execute("CREATE TABLE ex_factors(symbol VARCHAR, date DATE, ex_factor DOUBLE)")
            con.execute("INSERT INTO ex_factors VALUES ('000000','1999-01-01',2)")
            con.close()
            with patch.object(loader, "get_conn", side_effect=lambda: duckdb.connect(database)), patch.object(
                loader, "_cache_is_fresh", return_value=False
            ), patch.object(loader, "_load_full_cum_factor", side_effect=AssertionError("full-history cache")), patch.dict(
                os.environ, {"TRADAR_EXPERIMENT_SANDBOX": "1"}
            ):
                result = loader.load_prices("2026-01-05", "2026-01-05", adjust="backward", fields=["close"])
            self.assertEqual(result.shape, (1, 1))
            self.assertEqual(result.iloc[0, 0], 40)

    def test_forward_cold_anchors_preserve_last_available_trade_and_null_event_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "prices.duckdb")
            con = duckdb.connect(database)
            con.execute("""CREATE TABLE history AS
                SELECT symbol,date,close,close+1 AS open,close+2 AS high,close-1 AS low,
                       100.0 AS volume,1000.0 AS amount FROM (VALUES
                    ('600000',DATE '2026-01-09',10.0),('600000',DATE '2026-01-12',11.0),
                    ('600001',DATE '2026-01-09',20.0),
                    ('600002',DATE '2026-01-09',30.0),('600002',DATE '2026-01-12',31.0),
                    ('600003',DATE '2026-01-09',40.0),('600003',DATE '2026-01-12',41.0)
                ) prices(symbol,date,close)""")
            con.execute("CREATE TABLE ex_factors(symbol VARCHAR,date DATE,ex_factor DOUBLE)")
            con.execute("""INSERT INTO ex_factors VALUES
                ('600000','2000-01-01',2),('600000','2026-01-10',1.5),
                ('600000','2027-01-01',3),
                ('600001','2000-01-01',2),('600001','2026-01-10',3),
                ('600003','2000-01-01',2),('600003','2026-01-10',NULL)
            """)
            raw = con.execute("SELECT * FROM history ORDER BY date,symbol").df()
            raw["date"] = pd.to_datetime(raw["date"])
            con.close()
            with patch.object(loader, "get_conn", side_effect=lambda: duckdb.connect(database)), patch.object(
                loader, "_cache_is_fresh", return_value=False
            ), patch.object(loader, "CACHE_PATH", Path(directory) / "missing.parquet"), patch.dict(
                os.environ, {"TRADAR_EXPERIMENT_SANDBOX": "1"}
            ):
                factors = loader._compute_full_cum_factor()
                expected = loader._apply_price_adjustment(raw, factors, "forward")
                for fields in (["close"], ["open", "high", "low", "close", "volume", "amount"]):
                    with self.subTest(fields=fields):
                        actual = loader.load_prices("2026-01-09", "2026-01-12", fields=fields)
                        pd.testing.assert_frame_equal(actual, expected.set_index(["date", "symbol"])[fields].sort_index())
                self.assertFalse(loader.CACHE_PATH.exists())

    def test_forward_cold_wide_query_has_no_per_price_row_window(self):
        with tempfile.TemporaryDirectory() as directory:
            con = duckdb.connect(config={"threads": 1, "memory_limit": "32MiB",
                                        "temp_directory": directory, "max_temp_directory_size": "32MiB"})
            try:
                con.execute("""CREATE TABLE history AS
                    SELECT lpad(s::VARCHAR,6,'0') symbol, DATE '2021-01-01'+d::INTEGER date,
                           (10+d*0.01)::DOUBLE AS close,11.0::DOUBLE AS open,
                           12.0::DOUBLE AS high,9.0::DOUBLE AS low,1000.0::DOUBLE AS volume
                    FROM range(100) stocks(s),range(1000) days(d)""")
                con.execute("""CREATE TABLE ex_factors AS SELECT DISTINCT symbol,
                    DATE '2000-01-01' date,1.2::DOUBLE ex_factor FROM history""")
                con.execute("INSERT INTO ex_factors SELECT DISTINCT symbol,DATE '2022-01-01',0.5 FROM history")
                connection = Mock(wraps=con)
                result = loader._query_adjusted_prices(connection,
                    "SELECT symbol,date,open,high,low,close,volume FROM history WHERE date>=? AND date<=?",
                    ["2021-01-01", "2023-12-31"], ["open", "high", "low", "close", "volume"], "forward", None)
                query, params = connection.execute.call_args.args
                plan = con.execute("EXPLAIN " + query, params).fetchone()[1]
                self.assertNotIn("WINDOW", plan.upper())
                self.assertEqual(len(result), 100000)
                self.assertAlmostEqual(float(result.loc[result.date.eq(pd.Timestamp("2021-01-01")), "close"].iloc[0]), 20.0)
                self.assertAlmostEqual(float(result.loc[result.date.eq(result.date.max()), "close"].iloc[0]), 19.99)
            finally:
                con.close()

    def test_price_panel_loads_only_requested_field(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-05"), "600000")],
            names=["date", "symbol"],
        )
        close = pd.DataFrame({"close": [10.0]}, index=index)

        with patch.object(loader, "load_prices", return_value=close) as load:
            result = panel.price_panel(
                "2026-01-05", "2026-01-05", field="close", adjust="backward"
            )

        load.assert_called_once_with(
            "2026-01-05", "2026-01-05", adjust="backward", fields=["close"]
        )
        self.assertEqual(result.iloc[0, 0], 10.0)

    def test_fundamental_panels_load_only_requested_fields(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-05"), "600000")],
            names=["date", "symbol"],
        )
        fundamentals = pd.DataFrame(
            {"is_trading": [True], "turnover_rate": [2.0]}, index=index
        )
        panel._load_fundamentals.cache_clear()

        with patch.object(
            loader, "load_fundamentals", return_value=fundamentals
        ) as load:
            result = panel.fundamental_panels(
                "2026-01-05",
                "2026-01-05",
                fields=["is_trading", "turnover_rate"],
            )

        load.assert_called_once_with(
            "2026-01-05",
            "2026-01-05",
            fields=("is_trading", "turnover_rate"),
        )
        self.assertEqual(set(result), {"is_trading", "turnover_rate"})

    def test_cum_factor_cache_filters_dates_before_dataframe_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cum_factor.parquet"
            con = duckdb.connect()
            con.execute("""
                COPY (
                    SELECT * FROM (VALUES
                        (DATE '2024-12-31', '600000', 1.0),
                        (DATE '2025-01-02', '600000', 1.1),
                        (DATE '2025-01-03', '600000', 1.1)
                    ) factors(date, symbol, cum_factor)
                ) TO ? (FORMAT PARQUET)
            """, [str(cache)])
            con.close()

            with patch.object(loader, "CACHE_PATH", cache), patch.object(
                loader, "_cache_is_fresh", return_value=True
            ), patch.object(loader, "get_conn", side_effect=duckdb.connect):
                result = loader._load_full_cum_factor(
                    start="2025-01-02", end="2025-01-02"
                )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2025-01-02"))

    def test_buyable_mask_uses_exchange_change_pct_scale(self) -> None:
        change_pct = pd.DataFrame(
            [[9.79, 9.80, 20.0, -10.0]],
            index=pd.to_datetime(["2026-07-21"]),
            columns=["below", "threshold", "twenty_pct", "limit_down"],
        )
        trading = pd.DataFrame(
            True, index=change_pct.index, columns=change_pct.columns
        )
        limit = pd.DataFrame(
            [[10.0, 10.0, 20.0, 10.0]],
            index=change_pct.index,
            columns=change_pct.columns,
        )

        with patch.object(
            panel,
            "fundamental_panels",
            return_value={
                "is_trading": trading,
                "exchange_change_pct": change_pct,
            },
        ), patch.object(panel, "price_limit_pct_panel", return_value=limit):
            panel._execution_masks.cache_clear()
            result = panel.buyable_mask("2026-07-21", "2026-07-21")
            panel._execution_masks.cache_clear()

        self.assertEqual(
            result.iloc[0].to_dict(),
            {
                "below": True,
                "threshold": False,
                "twenty_pct": False,
                "limit_down": True,
            },
        )

    def test_trading_calendar_sees_new_database_dates(self) -> None:
        class Result:
            def __init__(self, dates):
                self.dates = dates

            def fetchdf(self):
                return pd.DataFrame({"date": self.dates})

        class Connection:
            def __init__(self, dates):
                self.dates = dates

            def execute(self, _query):
                return Result(self.dates)

            def close(self):
                pass

        with patch.object(loader, "get_conn", side_effect=[
            Connection(["2026-07-31"]),
            Connection(["2026-07-31", "2026-08-03"]),
        ]):
            first = loader.load_trading_calendar()
            second = loader.load_trading_calendar()

        self.assertEqual(first[-1], pd.Timestamp("2026-07-31"))
        self.assertEqual(second[-1], pd.Timestamp("2026-08-03"))


if __name__ == "__main__":
    unittest.main()
