from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from research import panel
from utils import loader


class LimitMoveMaskTests(unittest.TestCase):
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
