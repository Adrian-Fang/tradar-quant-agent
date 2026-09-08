"""Unified strategy result contract for research scripts and read-only UI consumers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

import pandas as pd


SCHEMA_VERSION = "1.1"
DEFAULT_RESULT_ROOT = Path("data/strategy_runs")

AssetType = Literal["stock", "etf", "index", "mixed", "other"]

METRIC_COLUMNS = [
    "window", "trades", "ann", "cum", "active", "sharpe", "mdd",
    "turnover", "cost", "gross", "mean_return", "win_rate",
]

SIGNAL_COLUMNS = [
    "strategy", "symbol", "signal_date", "signal_type", "status",
]

TRADE_COLUMNS = [
    "strategy", "symbol", "signal_date", "buy_date", "buy_price",
    "sell_date", "sell_price", "return", "exit_reason",
]

DATE_COLUMNS = {"signal_date", "buy_date", "sell_date", "data_asof"}


@dataclass(frozen=True)
class StrategyMeta:
    strategy: str
    version: str
    asset_type: AssetType
    data_asof: str | pd.Timestamp
    display_name: str | None = None
    description: str | None = None
    schema_version: str = SCHEMA_VERSION
    run_time: str | None = None

    def normalized(self) -> dict:
        data = asdict(self)
        data["data_asof"] = _date_string(data["data_asof"])
        data["run_time"] = data["run_time"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
        return data


@dataclass
class StrategyRunResult:
    meta: StrategyMeta
    metrics: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame

    def validate(self) -> None:
        _require_columns(self.metrics, METRIC_COLUMNS, "metrics")
        _require_columns(self.signals, SIGNAL_COLUMNS, "signals")
        _require_columns(self.trades, TRADE_COLUMNS, "trades")

        if not self.meta.strategy.strip():
            raise ValueError("meta.strategy cannot be empty.")

        for name, frame in (("signals", self.signals), ("trades", self.trades)):
            if frame.empty:
                continue
            values = frame["strategy"].dropna().astype(str).unique().tolist()
            invalid = [value for value in values if value != self.meta.strategy]
            if invalid:
                raise ValueError(
                    f"{name}.strategy must match meta.strategy={self.meta.strategy!r}; got {invalid[:5]}."
                )

    def normalized_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.validate()
        return (
            _normalize_frame(self.metrics, METRIC_COLUMNS),
            _normalize_frame(self.signals, SIGNAL_COLUMNS),
            _normalize_frame(self.trades, TRADE_COLUMNS),
        )


def make_empty_metrics(extra_columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS + list(extra_columns or []))


def make_empty_signals(extra_columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS + list(extra_columns or []))


def make_empty_trades(extra_columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS + list(extra_columns or []))


def write_strategy_result(
    result: StrategyRunResult,
    root: str | Path = DEFAULT_RESULT_ROOT,
    slot: str = "latest",
) -> Path:
    metrics, signals, trades = result.normalized_frames()

    output_dir = Path(root) / result.meta.strategy / slot
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(output_dir / "meta.json", result.meta.normalized())
    _write_csv_atomic(output_dir / "metrics.csv", metrics)
    _write_csv_atomic(output_dir / "signals.csv", signals)
    _write_csv_atomic(output_dir / "trades.csv", trades)

    return output_dir



def update_strategy_signals(
    meta: StrategyMeta,
    signals: pd.DataFrame,
    root: str | Path = DEFAULT_RESULT_ROOT,
    slot: str = "latest",
) -> Path:
    """Update the latest operational signals without rerunning the full backtest.

    Existing metrics.csv and trades.csv are preserved. If this is the first run,
    empty schema-valid files are created so a read-only UI can still discover it.
    """
    _require_columns(signals, SIGNAL_COLUMNS, "signals")
    output_dir = Path(root) / meta.strategy / slot
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json_atomic(output_dir / "meta.json", meta.normalized())
    _write_csv_atomic(output_dir / "signals.csv", _normalize_frame(signals, SIGNAL_COLUMNS))

    metrics_path = output_dir / "metrics.csv"
    trades_path = output_dir / "trades.csv"
    if not metrics_path.exists():
        _write_csv_atomic(metrics_path, make_empty_metrics())
    if not trades_path.exists():
        _write_csv_atomic(trades_path, make_empty_trades())

    return output_dir


def load_strategy_result(path: str | Path) -> StrategyRunResult:
    path = Path(path)

    with (path / "meta.json").open("r", encoding="utf-8") as file:
        raw_meta = json.load(file)

    meta = StrategyMeta(
        strategy=raw_meta["strategy"],
        version=raw_meta["version"],
        asset_type=raw_meta["asset_type"],
        data_asof=raw_meta["data_asof"],
        display_name=raw_meta.get("display_name"),
        description=raw_meta.get("description"),
        schema_version=raw_meta.get("schema_version", SCHEMA_VERSION),
        run_time=raw_meta.get("run_time"),
    )

    result = StrategyRunResult(
        meta=meta,
        metrics=pd.read_csv(path / "metrics.csv"),
        signals=pd.read_csv(path / "signals.csv"),
        trades=pd.read_csv(path / "trades.csv"),
    )
    result.validate()
    return result


def discover_strategy_results(
    root: str | Path = DEFAULT_RESULT_ROOT,
    slot: str = "latest",
) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []

    results = []
    for strategy_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidate = strategy_dir / slot
        required = [
            candidate / "meta.json",
            candidate / "metrics.csv",
            candidate / "signals.csv",
            candidate / "trades.csv",
        ]
        if all(path.exists() for path in required):
            results.append(candidate)

    return results


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_frame(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        if column in DATE_COLUMNS or column.lower().endswith("_date"):
            output[column] = output[column].map(_date_string_or_blank)

    extra = [column for column in output.columns if column not in required]
    return output.loc[:, required + extra]


def _date_string(value) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Date value cannot be empty.")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_string_or_blank(value):
    if value is None or pd.isna(value) or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp.replace(path)