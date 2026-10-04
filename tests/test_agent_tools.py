from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import pytest

from agent.core.contracts import ResearchRun, ToolResult
from agent.tools.calling import TOOL_SCHEMAS, normalize_tool_arguments
from agent.tools.research import ARTIFACT_ROOT, FACTOR_DEF_ROOT, evaluate_factor, inspect_universe, run_backtest
from research.factor_analyzer import FactorAnalyzer
from research.metrics import calc_metrics
from research.vector_backtest import BacktestConfig, run_backtest as canonical_run_backtest


class AgentToolContractTests(unittest.TestCase):
    def test_tool_result_and_research_run_round_trip(self) -> None:
        result = ToolResult(
            tool_name="demo",
            run_id="run-1",
            normalized_args={"date": "2026-01-01"},
            result={"value": 1},
            warnings=[{"code": "notice", "message": "small"}],
            provenance={"source": "test"},
            timing={"elapsed_ms": 1.25},
        )
        restored = ToolResult.from_json(result.to_json())
        self.assertEqual(restored.to_dict(), result.to_dict())

        frame_result = ToolResult(tool_name="frame", result={"frame": pd.DataFrame([[1]])})
        self.assertEqual(frame_result.to_dict()["result"]["frame"]["type"], "dataframe_summary")

        trace = ResearchRun(run_id="run-1", user_request="inspect")
        trace.add_step(result)
        restored_trace = ResearchRun.from_json(trace.to_json())
        self.assertEqual(restored_trace.to_dict(), trace.to_dict())
        self.assertEqual(restored_trace.steps[0]["seq"], 1)

    def test_inspect_universe_normal_path_matches_panel_masks(self) -> None:
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.Index(["A", "B", "C"])
        eligible = pd.DataFrame([[True, False, True], [False, True, True]], index=dates, columns=columns)
        trading = pd.DataFrame([[True, True, True], [True, True, True]], index=dates, columns=columns)
        buyable = pd.DataFrame([[True, False, True], [True, True, False]], index=dates, columns=columns)
        sellable = pd.DataFrame([[True, True, False], [True, True, True]], index=dates, columns=columns)
        limits = pd.DataFrame([[10.0, 10.0, float("nan")], [10.0, 10.0, 10.0]], index=dates, columns=columns)

        with patch("agent.tools.research.panel.eligible_universe_mask", return_value=eligible) as eligible_call, \
             patch("agent.tools.research.panel.buyable_mask", return_value=buyable) as buyable_call, \
             patch("agent.tools.research.panel.sellable_mask", return_value=sellable) as sellable_call, \
             patch("agent.tools.research.panel.price_limit_pct_panel", return_value=limits) as limit_call, \
             patch("agent.tools.research.panel.is_trading_mask", return_value=trading) as trading_call:
            result = inspect_universe(
                "2026-01-02",
                "2026-01-05",
                exclude_st=False,
                min_turnover_rate=2.0,
                min_listed_days=30,
                snapshot_dates=["2026-01-05"],
                run_id="run-1",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "run-1")
        self.assertEqual(result.result["daily_counts"][0]["eligible"], int(eligible.iloc[0].sum()))
        self.assertEqual(result.result["daily_counts"][0]["buyable"], int(buyable.iloc[0].sum()))
        self.assertEqual(result.result["daily_counts"][0]["sellable"], int(sellable.iloc[0].sum()))
        self.assertEqual(result.result["daily_counts"][0]["price_limit_known"], int(limits.iloc[0].notna().sum()))
        self.assertEqual(result.result["snapshots"][0]["membership"]["eligible"], ["B", "C"])
        eligible_call.assert_called_once_with("2026-01-02", "2026-01-05", exclude_st=False, min_turnover_rate=2.0, min_listed_days=30)
        buyable_call.assert_called_once_with("2026-01-02", "2026-01-05")
        sellable_call.assert_called_once_with("2026-01-02", "2026-01-05")
        limit_call.assert_called_once_with("2026-01-02", "2026-01-05")
        trading_call.assert_called_once_with("2026-01-02", "2026-01-05")

    def test_invalid_and_unsupported_args_return_structured_errors(self) -> None:
        invalid = inspect_universe("2026-01-05", "2026-01-02")
        self.assertEqual(invalid.status, "error")
        self.assertEqual(invalid.errors[0]["code"], "invalid_date_range")
        json.loads(invalid.to_json())

        unsupported = inspect_universe("2026-01-02", "2026-01-05", universe="custom")
        self.assertEqual(unsupported.status, "error")
        self.assertEqual(unsupported.errors[0]["code"], "unsupported_argument")

    def test_missing_data_returns_structured_error(self) -> None:
        with patch(
            "agent.tools.research.panel.eligible_universe_mask",
            side_effect=FileNotFoundError("missing panel"),
        ):
            result = inspect_universe("2026-01-02", "2026-01-05")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.errors[0]["code"], "missing_data")

    def test_internal_execution_error_returns_structured_error(self) -> None:
        with patch(
            "agent.tools.research.panel.eligible_universe_mask",
            side_effect=RuntimeError("panel exploded"),
        ):
            result = inspect_universe("2026-01-02", "2026-01-05")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.errors[0]["code"], "internal_execution_error")

    def test_evaluate_factor_matches_factor_analyzer_and_normalizes_args(self) -> None:
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        columns = pd.Index(["A", "B", "C", "D"])
        prices = pd.DataFrame(
            [[100 + day + step * day for step in range(4)] for day in range(10)],
            index=dates,
            columns=columns,
            dtype=float,
        )
        factor = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0] for _ in range(10)],
            index=dates,
            columns=columns,
        )
        eligible = pd.DataFrame(True, index=dates, columns=columns)
        with patch("agent.tools.research.required_warmup_days", return_value=7), \
             patch("agent.tools.research.load_factor_panels", return_value=({"close": prices}, {})), \
             patch("agent.tools.research.evaluate_factor_def_on_panels", return_value=factor), \
             patch("agent.tools.research.panel.eligible_universe_mask", return_value=eligible):
            result = evaluate_factor(
                "high52",
                "2026-01-02",
                "2026-01-08",
                warmup_days=120,
                ic_horizons=[1, 2, 1],
                ic_method="SPEARMAN",
                n_groups=2,
                return_clip=None,
                run_id="run-factor-1",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "run-factor-1")
        self.assertEqual(result.normalized_args["ic_horizons"], [1, 2])
        self.assertEqual(result.normalized_args["ic_method"], "spearman")
        self.assertEqual(result.result["observation"]["effective_warmup_days"], 120)
        self.assertEqual(result.result["observation"]["evaluation_excludes_warmup"], True)
        expected_analyzer = FactorAnalyzer(
            factor.loc["2026-01-02":"2026-01-08"],
            prices.loc["2026-01-02":"2026-01-08"],
            tradable=eligible.loc["2026-01-02":"2026-01-08"],
            return_clip=None,
        )
        expected_ic = expected_analyzer.calc_ic(period=1, method="spearman").mean()
        self.assertAlmostEqual(result.result["ic"]["1"]["ic_mean"], expected_ic)
        self.assertIn("groups", result.result)
        methodology = result.result["methodology"]
        self.assertEqual(methodology["ic_method"], "spearman")
        self.assertIn("spearman correlation", methodology["ic_definition"])
        self.assertIn("inverse rank association", methodology["ic_interpretation"])
        self.assertIn("not causality or a recommendation", methodology["ic_interpretation"])
        json.loads(result.to_json())

    def test_evaluate_factor_errors_are_structured(self) -> None:
        invalid = evaluate_factor("momentum_60d", "2026-01-08", "2026-01-02")
        self.assertEqual(invalid.errors[0]["code"], "invalid_date_range")

        bad_args = evaluate_factor(
            "momentum_60d", "2026-01-02", "2026-01-08", ic_method="bogus"
        )
        self.assertEqual(bad_args.errors[0]["code"], "invalid_eval_args")

        with patch("agent.tools.research.resolve_factor_def", side_effect=FileNotFoundError("no factor")):
            unknown = evaluate_factor("missing_factor", "2026-01-02", "2026-01-08")
        self.assertEqual(unknown.errors[0]["code"], "unknown_factor")

        with patch("agent.tools.research.load_factor_panels", side_effect=RuntimeError("boom")), \
             patch("agent.tools.research.resolve_factor_def", return_value="mock.yaml"), \
             patch("agent.tools.research.load_factor_definition", return_value={"steps": [{"field": "close"}]}), \
             patch("agent.tools.research.required_warmup_days", return_value=7):
            internal = evaluate_factor("mock_factor", "2026-01-02", "2026-01-08")
        self.assertEqual(internal.errors[0]["code"], "internal_execution_error")

    def test_run_backtest_uses_canonical_engine_and_compact_metrics(self) -> None:
        dates = pd.date_range("2026-01-05", periods=4, freq="D")
        columns = pd.Index(["A"])
        weights = pd.DataFrame([[1.0], [1.0], [0.0], [0.0]], index=dates, columns=columns)
        close = pd.DataFrame([[100.0], [110.0], [120.0], [120.0]], index=dates, columns=columns)
        opens = pd.DataFrame([[100.0], [105.0], [115.0], [120.0]], index=dates, columns=columns)
        tradable = pd.DataFrame(True, index=dates, columns=columns)

        result = run_backtest(
            weights,
            close,
            open_panel=opens,
            buyable=tradable,
            sellable=tradable,
            run_id="run-backtest-1",
        )
        expected = canonical_run_backtest(
            weights,
            close,
            buyable=tradable,
            sellable=tradable,
            config=BacktestConfig(),
            open_panel=opens,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_id, "run-backtest-1")
        self.assertEqual(result.result["config"]["buy_cost"], 0.001)
        self.assertEqual(result.result["config"]["sell_cost"], 0.0015)
        self.assertTrue(result.result["config"]["t_plus_1"])
        self.assertEqual(result.result["execution"]["buy_days"], 1)
        self.assertEqual(result.result["execution"]["sell_days"], 1)
        self.assertAlmostEqual(
            result.result["metrics"]["net"]["total_return"],
            calc_metrics(expected.returns)["total_return"],
        )
        self.assertNotIn("returns", result.result)
        json.loads(result.to_json())

    def test_run_backtest_requires_open_panel_and_returns_structured_error(self) -> None:
        dates = pd.date_range("2026-01-05", periods=2, freq="D")
        weights = pd.DataFrame({"A": [1.0, 0.0]}, index=dates)
        close = pd.DataFrame({"A": [100.0, 101.0]}, index=dates)
        result = run_backtest(weights, close)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.errors[0]["code"], "invalid_backtest_args")

    def test_run_backtest_accepts_csv_weight_and_price_artifacts(self) -> None:
        dates = pd.date_range("2026-01-05", periods=2, freq="D")
        weights = pd.DataFrame({"A": [1.0, 0.0]}, index=dates)
        close = pd.DataFrame({"A": [100.0, 101.0]}, index=dates)
        opens = pd.DataFrame({"A": [100.0, 100.5]}, index=dates)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=ARTIFACT_ROOT) as directory:
            paths = {}
            for name, frame in {"weights": weights, "close": close, "open": opens}.items():
                path = f"{directory}/{name}.csv"
                frame.to_csv(path)
                paths[name] = path
            result = run_backtest(
                paths["weights"], paths["close"], open_panel=paths["open"]
            )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.provenance["input_artifacts"]["target_weights"], paths["weights"])


@pytest.mark.parametrize("tool", ["inspect_universe", "evaluate_factor", "run_backtest"])
def test_null_model_options_use_existing_canonical_defaults(tool):
    dates = pd.date_range("2026-01-05", periods=3)
    frame = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=dates)
    required = {
        "inspect_universe": {"start_date": "2026-01-05", "end_date": "2026-01-07"},
        "evaluate_factor": {"factor": "high52", "observe_start": "2026-01-05", "observe_end": "2026-01-07"},
        "run_backtest": {"target_weights": frame, "price_panel": frame + 100, "open_panel": frame + 100},
    }[tool]
    schema = next(row for row in TOOL_SCHEMAS if row["name"] == tool)
    arguments = {key: None for key in schema["parameters"]["properties"]}
    arguments.update(required)
    with patch("agent.tools.research.panel.eligible_universe_mask", side_effect=FileNotFoundError("fixture")), \
            patch("agent.tools.research.load_factor_panels", side_effect=FileNotFoundError("fixture")):
        result = {"inspect_universe": inspect_universe, "evaluate_factor": evaluate_factor,
                  "run_backtest": run_backtest}[tool](**normalize_tool_arguments(tool, arguments))
    expected = {
        "inspect_universe": {"exclude_st": True, "min_turnover_rate": 1.0, "min_listed_days": 20, "snapshot_dates": []},
        "evaluate_factor": {"warmup_days": 120, "ic_horizons": [1, 5, 10, 20], "ic_method": "spearman", "n_groups": 5, "return_clip": 0.5},
        "run_backtest": {"buy_cost": 0.001, "sell_cost": 0.0015, "slippage": 0.0, "t_plus_1": True},
    }[tool]
    assert all(result.normalized_args[key] == value for key, value in expected.items())
    assert result.status == ("success" if tool == "run_backtest" else "error")
    if tool != "run_backtest":
        assert result.errors[0]["code"] == "missing_data"


@pytest.mark.parametrize("attack", ["external", "traversal", "symlink", "bare_traversal"])
def test_factor_paths_cannot_escape_the_canonical_root_before_probing(tmp_path, attack):
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=FACTOR_DEF_ROOT) as directory:
        external = tmp_path / "secret.yaml"
        external.write_text("private fixture", encoding="utf-8")
        alias = Path(directory) / "alias.yaml"
        alias.symlink_to(external)
        factor = {"external": external, "traversal": FACTOR_DEF_ROOT / ".." / "secret.yaml",
                  "symlink": alias, "bare_traversal": "../../secret"}[attack]
        with patch.object(Path, "exists", side_effect=AssertionError("must not probe")), \
                patch("agent.tools.research.load_factor_definition", side_effect=AssertionError("must not read")):
            result = evaluate_factor(factor, "2026-01-05", "2026-01-07")
    assert result.status == "error" and result.errors[0]["code"] == "path_not_allowed"
    assert "private fixture" not in result.to_json()


def test_canonical_explicit_yaml_path_evaluates_successfully():
    dates = pd.date_range("2026-01-05", periods=6)
    prices = pd.DataFrame({"A": [100 + day for day in range(6)], "B": [101 + day * 2 for day in range(6)]}, index=dates)
    factor = prices.rank(axis=1)
    with patch("agent.tools.research.load_factor_panels", return_value=({"close": prices}, {})), \
            patch("agent.tools.research.evaluate_factor_def_on_panels", return_value=factor), \
            patch("agent.tools.research.panel.eligible_universe_mask", return_value=prices.notna()):
        result = evaluate_factor(FACTOR_DEF_ROOT / "high52.yaml", "2026-01-05", "2026-01-10", ic_horizons=[1], n_groups=2)
    assert result.status == "success"
    assert result.result["factor"]["source_path"] == str((FACTOR_DEF_ROOT / "high52.yaml").resolve())


@pytest.mark.parametrize("field", ["target_weights", "price_panel", "open_panel", "buyable", "sellable", "benchmark_returns"])
@pytest.mark.parametrize("attack", ["external", "traversal", "symlink"])
def test_all_backtest_paths_are_confined_before_any_read_or_probe(tmp_path, field, attack):
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ARTIFACT_ROOT) as directory:
        external = tmp_path / "secret.csv"
        external.write_text("private fixture", encoding="utf-8")
        alias = Path(directory) / "alias.csv"
        alias.symlink_to(external)
        value = {"external": external, "traversal": ARTIFACT_ROOT / ".." / "secret.csv", "symlink": alias}[attack]
        frame = pd.DataFrame({"A": [1.0, 0.0]}, index=pd.date_range("2026-01-05", periods=2))
        arguments = {"target_weights": frame, "price_panel": frame + 100, "open_panel": frame + 100, field: value}
        with patch.object(Path, "exists", side_effect=AssertionError("must not probe")), \
                patch.object(Path, "is_file", side_effect=AssertionError("must not probe")), \
                patch("pandas.read_csv", side_effect=AssertionError("must not read")), \
                patch("pandas.read_parquet", side_effect=AssertionError("must not read")):
            result = run_backtest(**arguments)
    assert result.status == "error" and result.errors[0]["code"] == "path_not_allowed"
    assert "private fixture" not in result.to_json()


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_backtest_runtime_artifacts_and_trusted_series_keep_canonical_semantics(extension):
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2026-01-05", periods=3)
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=dates)
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dates)
    benchmark = pd.Series([0.0, 0.01, 0.01], index=dates)
    with TemporaryDirectory(dir=ARTIFACT_ROOT) as directory:
        paths = []
        for name, frame in (("weights", weights), ("close", close), ("open", close)):
            path = Path(directory) / f"{name}.{extension}"
            (frame.to_csv if extension == "csv" else frame.to_parquet)(path)
            paths.append(path)
        from_files = run_backtest(paths[0], str(paths[1]), open_panel=paths[2], benchmark_returns=benchmark)
        from_frames = run_backtest(weights, close, open_panel=close, benchmark_returns=benchmark)
    assert from_files.status == from_frames.status == "success"
    assert from_files.result == from_frames.result
    assert from_files.result["config"] == {"buy_cost": 0.001, "sell_cost": 0.0015, "slippage": 0.0, "t_plus_1": True}


if __name__ == "__main__":
    unittest.main()
