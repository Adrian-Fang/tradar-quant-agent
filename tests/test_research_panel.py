from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from research import panel
from utils import loader


class LimitMoveMaskTests(unittest.TestCase):
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
