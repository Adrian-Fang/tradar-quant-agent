from __future__ import annotations

import unittest

import pandas as pd

from research.vector_backtest import BacktestConfig, run_backtest


def _prices(index, columns):
    return pd.DataFrame(100.0, index=index, columns=columns)


class VectorBacktestCashConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2026-01-05", periods=4)
        self.columns = ["A", "B", "C"]
        self.no_cost = BacktestConfig(
            buy_cost=0.0, sell_cost=0.0, slippage=0.0, t_plus_1=False
        )

    def test_locked_position_uses_target_budget_before_new_buys(self) -> None:
        target = pd.DataFrame(
            [
                [0.6, 0.4, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.2, 0.0],
                [0.0, 1.0, 0.0],
            ],
            index=self.dates,
            columns=self.columns,
        )
        tradable = pd.DataFrame(True, index=self.dates, columns=self.columns)
        tradable.loc[self.dates[1:3], "A"] = False

        result = run_backtest(
            target,
            _prices(self.dates, self.columns),
            buyable=tradable,
            sellable=tradable,
            config=self.no_cost,
        )

        self.assertListEqual(result.weights.iloc[0].tolist(), [0.6, 0.4, 0.0])
        self.assertListEqual(result.weights.iloc[1].tolist(), [0.6, 0.0, 0.4])
        # 目标预算只有20%，但锁定仓位已经占60%，不得再开新仓。
        self.assertListEqual(result.weights.iloc[2].tolist(), [0.6, 0.0, 0.0])
        self.assertListEqual(result.weights.iloc[3].tolist(), [0.0, 1.0, 0.0])
        self.assertLessEqual(
            float(result.weights.abs().sum(axis=1).max()), 1.0
        )

    def test_regime_budget_is_not_filled_back_to_one(self) -> None:
        target = pd.DataFrame(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0]],
            index=self.dates,
            columns=self.columns,
        )
        tradable = pd.DataFrame(True, index=self.dates, columns=self.columns)
        tradable.loc[self.dates[1], "A"] = False

        result = run_backtest(
            target,
            _prices(self.dates, self.columns),
            buyable=tradable,
            sellable=tradable,
            config=self.no_cost,
        )

        self.assertAlmostEqual(result.weights.loc[self.dates[1], "A"], 0.1)
        self.assertAlmostEqual(result.weights.loc[self.dates[1], "B"], 0.1)
        self.assertAlmostEqual(result.weights.loc[self.dates[1]].sum(), 0.2)

    def test_t_plus_one_uses_execution_day_tradability(self) -> None:
        target = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            index=self.dates,
            columns=self.columns,
        )
        tradable = pd.DataFrame(True, index=self.dates, columns=self.columns)
        tradable.loc[self.dates[1], "A"] = False
        config = BacktestConfig(
            buy_cost=0.0, sell_cost=0.0, slippage=0.0, t_plus_1=True
        )

        result = run_backtest(
            target,
            _prices(self.dates, self.columns),
            buyable=tradable,
            sellable=tradable,
            config=config,
            open_panel=_prices(self.dates, self.columns),
        )

        # d1 想执行 d0 的 A=100%，但 A 在实际执行日不可交易，且此前无持仓。
        self.assertAlmostEqual(result.weights.loc[self.dates[1], "A"], 0.0)
        self.assertAlmostEqual(result.weights.loc[self.dates[2], "B"], 1.0)

    def test_non_rebalance_day_drifts_without_a_free_trade(self) -> None:
        target = pd.DataFrame(
            [[0.5, 0.5, 0.0]] * len(self.dates),
            index=self.dates,
            columns=self.columns,
        )
        target.attrs["rebalance_mask"] = pd.Series(
            [True, False, False, False], index=self.dates
        )
        open_price = _prices(self.dates, self.columns)
        close = open_price.copy()
        close.loc[self.dates[0], "A"] = 110.0
        open_price.loc[self.dates[1]:, "A"] = 110.0
        close.loc[self.dates[1]:, "A"] = 110.0

        result = run_backtest(
            target, close, config=self.no_cost, open_panel=open_price
        )

        self.assertAlmostEqual(result.weights.iloc[0]["A"], 11 / 21)
        self.assertAlmostEqual(result.weights.iloc[1]["A"], 11 / 21)
        self.assertAlmostEqual(result.turnover.iloc[1], 0.0)

    def test_t_plus_one_splits_overnight_and_intraday_and_charges_at_open(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=3)
        target = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
        close = pd.DataFrame({"A": [100.0, 120.0, 120.0]}, index=dates)
        open_price = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=dates)
        config = BacktestConfig(buy_cost=0.01, sell_cost=0.0, t_plus_1=True)

        result = run_backtest(
            target, close, config=config, open_panel=open_price
        )

        self.assertAlmostEqual(result.returns.iloc[0], 0.0)
        self.assertAlmostEqual(result.returns.iloc[1], (120 / 110) / 1.01 - 1)
        self.assertAlmostEqual(result.buys.iloc[1], 1 / 1.01)
        self.assertAlmostEqual(result.cost.iloc[1], 0.01 / 1.01)
        self.assertAlmostEqual(result.weights.iloc[1]["A"], 1.0)

    def test_requested_gross_above_one_is_scaled(self) -> None:
        target = pd.DataFrame(
            [[0.8, 0.8, 0.0]] * len(self.dates),
            index=self.dates,
            columns=self.columns,
        )

        result = run_backtest(
            target, _prices(self.dates, self.columns), config=self.no_cost
        )

        self.assertAlmostEqual(result.weights.iloc[0]["A"], 0.5)
        self.assertAlmostEqual(result.weights.iloc[0]["B"], 0.5)
        self.assertTrue((result.weights.abs().sum(axis=1) <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
