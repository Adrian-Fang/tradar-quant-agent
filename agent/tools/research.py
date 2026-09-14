"""Agent-facing wrappers around canonical research capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import time

import pandas as pd

from research import panel
from research.factor_analyzer import FactorAnalyzer
from research.factor_dsl import (
    evaluate_factor_def_on_panels,
    load_factor_definition,
    load_factor_panels,
    required_warmup_days,
    resolve_factor_def,
)
from research.metrics import calc_metrics
from research.vector_backtest import (
    BacktestConfig,
    run_backtest as canonical_run_backtest,
)

from ..core.contracts import ToolResult, ToolStatus


def _normalize_date(value: Any, field_name: str) -> tuple[pd.Timestamp, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field_name} is required")
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp, timestamp.strftime("%Y-%m-%d")


def _normalize_snapshot_dates(value: Any) -> list[tuple[pd.Timestamp, str]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("snapshot_dates must be a list or tuple of dates")
    output = []
    seen = set()
    for item in value:
        timestamp, normalized = _normalize_date(item, "snapshot_dates")
        if normalized not in seen:
            output.append((timestamp, normalized))
            seen.add(normalized)
    return output


def _symbols(row: pd.Series) -> list[str]:
    return sorted(str(symbol) for symbol in row.index[row.fillna(False).astype(bool)])


def inspect_universe(
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    exclude_st: bool = True,
    min_turnover_rate: float = 1.0,
    min_listed_days: int = 20,
    snapshot_dates: list[str] | tuple[str, ...] | None = None,
    run_id: str | None = None,
    **unsupported: Any,
) -> ToolResult:
    """Inspect canonical universe and execution masks without changing them."""
    started = time.perf_counter()
    tool_name = "inspect_universe"
    raw_args = {
        "start_date": start_date,
        "end_date": end_date,
        "exclude_st": exclude_st,
        "min_turnover_rate": min_turnover_rate,
        "min_listed_days": min_listed_days,
        "snapshot_dates": snapshot_dates,
        **unsupported,
    }

    def failure(code: str, message: str, *, warnings=None) -> ToolResult:
        return ToolResult.error(
            tool_name,
            raw_args,
            code,
            message,
            run_id=run_id,
            warnings=warnings,
            provenance={"module": "agent.tools"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    if unsupported:
        names = ", ".join(sorted(unsupported))
        return failure("unsupported_argument", f"unsupported argument(s): {names}")
    if not isinstance(exclude_st, bool):
        return failure("unsupported_argument", "exclude_st must be bool")
    if isinstance(min_turnover_rate, bool):
        return failure("unsupported_argument", "min_turnover_rate must be numeric")
    try:
        normalized_turnover = float(min_turnover_rate)
    except (TypeError, ValueError):
        return failure("unsupported_argument", "min_turnover_rate must be numeric")
    if not math.isfinite(normalized_turnover):
        return failure("unsupported_argument", "min_turnover_rate must be finite")
    if isinstance(min_listed_days, bool) or not isinstance(min_listed_days, int):
        return failure("unsupported_argument", "min_listed_days must be a non-negative integer")
    if min_listed_days < 0:
        return failure("unsupported_argument", "min_listed_days must be a non-negative integer")

    try:
        start_ts, normalized_start = _normalize_date(start_date, "start_date")
        end_ts, normalized_end = _normalize_date(end_date, "end_date")
        if start_ts > end_ts:
            return failure("invalid_date_range", "start_date must be on or before end_date")
        snapshots = _normalize_snapshot_dates(snapshot_dates)
        outside = [
            normalized
            for timestamp, normalized in snapshots
            if timestamp < start_ts or timestamp > end_ts
        ]
        if outside:
            return failure("invalid_date_range", f"snapshot_dates outside requested range: {outside}")
    except ValueError as exc:
        return failure("invalid_date_range", str(exc))

    normalized_args = {
        "start_date": normalized_start,
        "end_date": normalized_end,
        "exclude_st": exclude_st,
        "min_turnover_rate": normalized_turnover,
        "min_listed_days": min_listed_days,
        "snapshot_dates": [normalized for _, normalized in snapshots],
    }

    try:
        frames = {
            "eligible": panel.eligible_universe_mask(
                normalized_start,
                normalized_end,
                exclude_st=exclude_st,
                min_turnover_rate=normalized_turnover,
                min_listed_days=min_listed_days,
            ),
            "buyable": panel.buyable_mask(normalized_start, normalized_end),
            "sellable": panel.sellable_mask(normalized_start, normalized_end),
            "price_limit_pct": panel.price_limit_pct_panel(normalized_start, normalized_end),
            "is_trading": panel.is_trading_mask(normalized_start, normalized_end),
        }
        if any(not isinstance(frame, pd.DataFrame) for frame in frames.values()):
            raise TypeError("canonical panel functions must return pandas DataFrame")
        if any(frame.empty for frame in frames.values()):
            return ToolResult.error(
                tool_name,
                normalized_args,
                "missing_data",
                "one or more canonical universe panels are empty",
                run_id=run_id,
                provenance={"module": "research.panel"},
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        warnings = []
        eligible_raw = frames["eligible"]
        base_index = pd.DatetimeIndex(pd.to_datetime(eligible_raw.index)).sort_values()
        base_columns = pd.Index(eligible_raw.columns)
        aligned = {}
        for name, frame in frames.items():
            index = pd.DatetimeIndex(pd.to_datetime(frame.index)).sort_values()
            if not index.equals(base_index) or not frame.columns.equals(base_columns):
                warnings.append({
                    "code": "mask_alignment_normalized",
                    "message": f"{name} was reindexed to eligible panel shape",
                })
            aligned[name] = frame.copy()
            aligned[name].index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            aligned[name] = aligned[name].reindex(index=base_index, columns=base_columns)

        eligible = aligned["eligible"].fillna(False).astype(bool)
        trading = aligned["is_trading"].fillna(False).astype(bool)
        buyable = aligned["buyable"].fillna(False).astype(bool)
        sellable = aligned["sellable"].fillna(False).astype(bool)
        limit_known = aligned["price_limit_pct"].notna()

        daily_counts = []
        for timestamp in base_index:
            eligible_row = eligible.loc[timestamp]
            trading_row = trading.loc[timestamp]
            daily_counts.append({
                "date": timestamp.strftime("%Y-%m-%d"),
                "eligible": int(eligible_row.sum()),
                "trading": int(trading_row.sum()),
                "buyable": int(buyable.loc[timestamp].sum()),
                "sellable": int(sellable.loc[timestamp].sum()),
                "price_limit_known": int(limit_known.loc[timestamp].sum()),
                "not_trading": int((~trading_row).sum()),
                "eligible_filter_excluded": int((trading_row & ~eligible_row).sum()),
            })

        snapshots_out = []
        for timestamp, normalized in snapshots:
            if timestamp not in eligible.index:
                warnings.append({
                    "code": "snapshot_not_in_data",
                    "message": f"snapshot date is not in the returned trading calendar: {normalized}",
                })
                snapshots_out.append({
                    "date": normalized,
                    "is_trading_day": False,
                    "counts": None,
                    "membership": None,
                })
                continue
            counts = next(row for row in daily_counts if row["date"] == normalized)
            snapshots_out.append({
                "date": normalized,
                "is_trading_day": True,
                "counts": counts,
                "membership": {
                    "eligible": _symbols(eligible.loc[timestamp]),
                    "trading": _symbols(trading.loc[timestamp]),
                    "buyable": _symbols(buyable.loc[timestamp]),
                    "sellable": _symbols(sellable.loc[timestamp]),
                    "price_limit_known": _symbols(limit_known.loc[timestamp]),
                },
            })

        status: ToolStatus = "partial" if any(
            issue["code"] == "snapshot_not_in_data" for issue in warnings
        ) else "success"
        return ToolResult(
            tool_name=tool_name,
            **({"run_id": run_id} if run_id is not None else {}),
            status=status,
            normalized_args=normalized_args,
            result={
                "summary": {
                    "requested_start": normalized_start,
                    "requested_end": normalized_end,
                    "data_start": base_index.min().strftime("%Y-%m-%d"),
                    "data_end": base_index.max().strftime("%Y-%m-%d"),
                    "trading_days": len(base_index),
                    "symbols": len(base_columns),
                    "filters": {
                        "exclude_st": exclude_st,
                        "min_turnover_rate": normalized_turnover,
                        "min_listed_days": min_listed_days,
                    },
                },
                "daily_counts": daily_counts,
                "snapshots": snapshots_out,
            },
            warnings=warnings,
            provenance={
                "module": "research.panel",
                "functions": [
                    "eligible_universe_mask",
                    "buyable_mask",
                    "sellable_mask",
                    "price_limit_pct_panel",
                    "is_trading_mask",
                ],
                "price_limit_source": "research.panel.price_limit_pct_panel",
            },
            timing={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
        return ToolResult.error(
            tool_name,
            normalized_args,
            "missing_data",
            f"canonical universe data unavailable: {type(exc).__name__}: {exc}",
            run_id=run_id,
            provenance={"module": "research.panel"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        return ToolResult.error(
            tool_name,
            normalized_args,
            "internal_execution_error",
            f"inspect_universe failed: {type(exc).__name__}: {exc}",
            run_id=run_id,
            provenance={"module": "agent.tools"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


_IC_METHODS = {"pearson", "kendall", "spearman"}


def _scalar(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _summary(series: pd.Series) -> dict[str, Any]:
    values = series.dropna()
    n = len(values)
    if n == 0:
        return {
            "ic_mean": None,
            "ic_std": None,
            "ir": None,
            "ic_win_rate": None,
            "t_stat": None,
            "n_days": 0,
        }
    mean = values.mean()
    std = values.std()
    ir = mean / std if pd.notna(std) and std != 0 else None
    return {
        "ic_mean": _scalar(mean),
        "ic_std": _scalar(std),
        "ir": _scalar(ir),
        "ic_win_rate": _scalar((values > 0).mean()),
        "t_stat": _scalar(ir * (n ** 0.5) if ir is not None else None),
        "n_days": n,
    }


def _normalize_horizons(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("ic_horizons must be a list or tuple of positive integers")
    horizons = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError("ic_horizons must be a list or tuple of positive integers")
        if item not in horizons:
            horizons.append(item)
    if not horizons:
        raise ValueError("ic_horizons cannot be empty")
    return tuple(horizons)


def _group_summary(
    analyzer: FactorAnalyzer,
    period: int,
    n_groups: int,
) -> dict[str, Any]:
    """Compactly summarize the canonical FactorAnalyzer group returns."""
    daily_returns = analyzer.group_returns(period=period, n_groups=n_groups)
    groups = analyzer.quantile(n_groups)
    forward = analyzer.forward_return(period)
    rows = []
    for group in range(n_groups):
        values = forward.where(groups == group).stack()
        rows.append({
            "group": f"G{group + 1}",
            "n": int(values.notna().sum()),
            "mean": _scalar(daily_returns[f"G{group + 1}"].mean()),
            "median": _scalar(values.median()),
        })
    means = [row["mean"] for row in rows]
    comparisons = [
        means[i + 1] >= means[i]
        for i in range(len(means) - 1)
        if means[i] is not None and means[i + 1] is not None
    ]
    return {
        "period": period,
        "n_groups": n_groups,
        "groups": rows,
        "spread": _scalar(means[-1] - means[0])
        if means[0] is not None and means[-1] is not None
        else None,
        "monotonicity": {
            "adjacent_non_decreasing": int(sum(comparisons)),
            "adjacent_pairs": len(comparisons),
        },
    }


def _yearly_ic(analyzer: FactorAnalyzer, period: int, method: str) -> list[dict[str, Any]]:
    ic = analyzer.calc_ic(period=period, method=method)
    rows = []
    for year, values in ic.groupby(ic.index.year):
        row = _summary(values)
        rows.append({"year": int(year), "period": period, **row})
    return rows


def evaluate_factor(
    factor: str | Path,
    observe_start: str,
    observe_end: str,
    *,
    warmup_days: int = 120,
    ic_horizons: list[int] | tuple[int, ...] = (1, 5, 10, 20),
    ic_method: str = "spearman",
    n_groups: int = 5,
    exclude_st: bool = True,
    min_turnover_rate: float = 1.0,
    min_listed_days: int = 20,
    return_clip: float | None = 0.5,
    run_id: str | None = None,
    **unsupported: Any,
) -> ToolResult:
    """Evaluate one YAML factor using only canonical research modules.

    ``warmup_days`` is a minimum calendar warmup.  The effective load window is
    the maximum of it and the factor DSL's required warmup.
    """
    started = time.perf_counter()
    tool_name = "evaluate_factor"
    raw_args = {
        "factor": factor,
        "observe_start": observe_start,
        "observe_end": observe_end,
        "warmup_days": warmup_days,
        "ic_horizons": ic_horizons,
        "ic_method": ic_method,
        "n_groups": n_groups,
        "exclude_st": exclude_st,
        "min_turnover_rate": min_turnover_rate,
        "min_listed_days": min_listed_days,
        "return_clip": return_clip,
        **unsupported,
    }

    def failure(code: str, message: str, *, normalized_args=None) -> ToolResult:
        return ToolResult.error(
            tool_name,
            normalized_args or raw_args,
            code,
            message,
            run_id=run_id,
            provenance={"module": "agent.tools"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    if unsupported:
        names = ", ".join(sorted(unsupported))
        return failure("unsupported_argument", f"unsupported argument(s): {names}")
    if isinstance(warmup_days, bool) or not isinstance(warmup_days, int) or warmup_days < 0:
        return failure("invalid_eval_args", "warmup_days must be a non-negative integer")
    try:
        horizons = _normalize_horizons(ic_horizons)
    except ValueError as exc:
        return failure("invalid_eval_args", str(exc))
    if not isinstance(ic_method, str) or ic_method.lower() not in _IC_METHODS:
        return failure("invalid_eval_args", f"ic_method must be one of {sorted(_IC_METHODS)}")
    method = ic_method.lower()
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 2:
        return failure("invalid_eval_args", "n_groups must be an integer >= 2")
    if not isinstance(exclude_st, bool):
        return failure("invalid_eval_args", "exclude_st must be bool")
    if isinstance(min_turnover_rate, bool):
        return failure("invalid_eval_args", "min_turnover_rate must be numeric")
    try:
        normalized_turnover = float(min_turnover_rate)
    except (TypeError, ValueError):
        return failure("invalid_eval_args", "min_turnover_rate must be numeric")
    if not math.isfinite(normalized_turnover):
        return failure("invalid_eval_args", "min_turnover_rate must be finite")
    if isinstance(min_listed_days, bool) or not isinstance(min_listed_days, int) or min_listed_days < 0:
        return failure("invalid_eval_args", "min_listed_days must be a non-negative integer")
    if return_clip is not None:
        if isinstance(return_clip, bool):
            return failure("invalid_eval_args", "return_clip must be a finite non-negative number or None")
        try:
            normalized_clip = float(return_clip)
        except (TypeError, ValueError):
            return failure("invalid_eval_args", "return_clip must be a finite non-negative number or None")
        if not math.isfinite(normalized_clip) or normalized_clip < 0:
            return failure("invalid_eval_args", "return_clip must be a finite non-negative number or None")
    else:
        normalized_clip = None

    try:
        start_ts, normalized_start = _normalize_date(observe_start, "observe_start")
        end_ts, normalized_end = _normalize_date(observe_end, "observe_end")
        if start_ts > end_ts:
            return failure("invalid_date_range", "observe_start must be on or before observe_end")
    except ValueError as exc:
        return failure("invalid_date_range", str(exc))

    normalized_args = {
        "factor": str(factor),
        "observe_start": normalized_start,
        "observe_end": normalized_end,
        "warmup_days": warmup_days,
        "ic_horizons": list(horizons),
        "ic_method": method,
        "n_groups": n_groups,
        "exclude_st": exclude_st,
        "min_turnover_rate": normalized_turnover,
        "min_listed_days": min_listed_days,
        "return_clip": normalized_clip,
    }

    try:
        factor_path = resolve_factor_def(factor)
        definition = load_factor_definition(factor_path)
    except FileNotFoundError as exc:
        return failure("unknown_factor", str(exc), normalized_args=normalized_args)
    except (TypeError, ValueError) as exc:
        return failure("invalid_eval_args", f"invalid factor definition: {exc}", normalized_args=normalized_args)

    try:
        required_warmup = required_warmup_days([definition])
        effective_warmup = max(warmup_days, required_warmup)
        load_start = (
            start_ts - pd.Timedelta(days=effective_warmup)
        ).strftime("%Y-%m-%d")
        panels, group_maps = load_factor_panels(
            [definition],
            load_start,
            normalized_end,
            extra_fields={"close"},
        )
        factor_panel = evaluate_factor_def_on_panels(definition, panels, group_maps)
        price_panel = panels.get("close")
        if price_panel is None:
            raise ValueError("canonical factor panels did not include close")
        eligible = panel.eligible_universe_mask(
            load_start,
            normalized_end,
            exclude_st=exclude_st,
            min_turnover_rate=normalized_turnover,
            min_listed_days=min_listed_days,
        )
        factor_panel = factor_panel.reindex_like(price_panel)
        eligible = eligible.reindex_like(factor_panel).fillna(False)
        evaluation_factor = factor_panel.loc[normalized_start:normalized_end]
        evaluation_price = price_panel.reindex_like(factor_panel).loc[normalized_start:normalized_end]
        evaluation_eligible = eligible.loc[normalized_start:normalized_end]
        if evaluation_factor.empty or evaluation_factor.notna().sum().sum() == 0:
            return failure(
                "missing_data",
                "factor has no usable observations in the requested evaluation period",
                normalized_args=normalized_args,
            )
        analyzer = FactorAnalyzer(
            evaluation_factor,
            evaluation_price,
            tradable=evaluation_eligible,
            return_clip=normalized_clip,
            label=str(definition.get("name", Path(factor_path).stem)),
        )
        coverage = analyzer.coverage()
        valid = coverage["valid_names"]
        eligible_counts = coverage["eligible_names"]
        coverage_pct = coverage["coverage"]
        ic = {
            str(period): _summary(analyzer.calc_ic(period=period, method=method))
            for period in horizons
        }
        groups = {
            str(period): _group_summary(analyzer, period, n_groups)
            for period in horizons
        }
        yearly = [
            row
            for period in horizons
            for row in _yearly_ic(analyzer, period, method)
        ]
        loaded_index = pd.DatetimeIndex(pd.to_datetime(factor_panel.index))
        metadata = {
            "name": str(definition.get("name", Path(factor_path).stem)),
            "description": definition.get("description"),
            "source_path": str(Path(factor_path).resolve()),
            "definition": {
                key: value for key, value in definition.items()
                if not str(key).startswith("__")
            },
        }
        return ToolResult(
            tool_name=tool_name,
            **({"run_id": run_id} if run_id is not None else {}),
            status="success",
            normalized_args=normalized_args,
            result={
                "factor": metadata,
                "observation": {
                    "requested_start": normalized_start,
                    "requested_end": normalized_end,
                    "actual_start": evaluation_factor.index.min().strftime("%Y-%m-%d"),
                    "actual_end": evaluation_factor.index.max().strftime("%Y-%m-%d"),
                    "evaluation_days": len(evaluation_factor.index),
                    "load_start": load_start,
                    "load_end": normalized_end,
                    "loaded_data_start": loaded_index.min().strftime("%Y-%m-%d"),
                    "loaded_data_end": loaded_index.max().strftime("%Y-%m-%d"),
                    "required_warmup_days": required_warmup,
                    "requested_warmup_days": warmup_days,
                    "effective_warmup_days": effective_warmup,
                    "evaluation_excludes_warmup": True,
                },
                "coverage": {
                    "days": len(coverage),
                    "valid_names_mean": _scalar(valid.mean()),
                    "valid_names_median": _scalar(valid.median()),
                    "eligible_names_mean": _scalar(eligible_counts.mean()),
                    "eligible_names_median": _scalar(eligible_counts.median()),
                    "coverage_mean": _scalar(coverage_pct.mean()),
                    "coverage_median": _scalar(coverage_pct.median()),
                    "coverage_min": _scalar(coverage_pct.min()),
                    "coverage_max": _scalar(coverage_pct.max()),
                },
                "ic": ic,
                "groups": groups,
                "yearly_ic": yearly,
            },
            provenance={
                "modules": [
                    "research.factor_dsl",
                    "research.factor_analyzer",
                    "research.panel",
                ],
                "functions": [
                    "resolve_factor_def",
                    "load_factor_definition",
                    "required_warmup_days",
                    "load_factor_panels",
                    "evaluate_factor_def_on_panels",
                    "FactorAnalyzer.calc_ic",
                    "FactorAnalyzer.group_returns",
                    "FactorAnalyzer.coverage",
                    "panel.eligible_universe_mask",
                ],
                "evaluation_period_excludes_warmup": True,
            },
            timing={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
        return failure(
            "missing_data",
            f"factor research data unavailable: {type(exc).__name__}: {exc}",
            normalized_args=normalized_args,
        )
    except Exception as exc:
        return failure(
            "internal_execution_error",
            f"evaluate_factor failed: {type(exc).__name__}: {exc}",
            normalized_args=normalized_args,
        )


def _artifact_label(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame rows={value.shape[0]} columns={value.shape[1]}>"
    if isinstance(value, pd.Series):
        return f"<Series rows={len(value)}>"
    return str(value)


def _load_table_artifact(value: Any, field_name: str) -> tuple[pd.DataFrame, str]:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        source = _artifact_label(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{field_name} artifact not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"{field_name} supports only CSV or Parquet artifacts")
        source = str(path.resolve())
    else:
        raise TypeError(f"{field_name} must be a pandas DataFrame or CSV/Parquet path")

    preserve_index = not isinstance(frame.index, pd.RangeIndex)
    if {"date", "symbol"}.issubset(frame.columns):
        candidates = {
            "target_weights": ("weight", "target_weight", "value"),
            "price_panel": ("close", "price", "value"),
            "open_panel": ("open", "price", "value"),
            "buyable": ("buyable", "value"),
            "sellable": ("sellable", "value"),
        }
        value_column = next(
            (column for column in candidates.get(field_name, ("value",)) if column in frame),
            None,
        )
        if value_column is None:
            raise ValueError(f"{field_name} long artifact has no supported value column")
        frame = frame.pivot(index="date", columns="symbol", values=value_column)
    elif not preserve_index:
        date_column = "date" if "date" in frame.columns else frame.columns[0]
        frame = frame.set_index(date_column)

    try:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise")).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} artifact has invalid dates") from exc
    if frame.index.has_duplicates:
        raise ValueError(f"{field_name} artifact has duplicate dates")
    frame = frame.sort_index()
    frame.columns = pd.Index([str(column) for column in frame.columns])
    return frame, source


def _load_benchmark_returns(value: Any) -> tuple[pd.Series | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, pd.Series):
        series = value.copy()
        source = _artifact_label(value)
    else:
        frame, source = _load_table_artifact(value, "benchmark_returns")
        if frame.shape[1] != 1:
            raise ValueError("benchmark_returns must contain exactly one return column")
        series = frame.iloc[:, 0]
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="raise")).normalize()
    if series.index.has_duplicates:
        raise ValueError("benchmark_returns has duplicate dates")
    return pd.to_numeric(series, errors="coerce").sort_index(), source


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            compact[key] = _scalar(value)
        else:
            compact[key] = value
    return compact


def run_backtest(
    target_weights: pd.DataFrame | str | Path,
    price_panel: pd.DataFrame | str | Path,
    *,
    open_panel: pd.DataFrame | str | Path | None = None,
    buyable: pd.DataFrame | str | Path | None = None,
    sellable: pd.DataFrame | str | Path | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | str | Path | None = None,
    buy_cost: float | None = None,
    sell_cost: float | None = None,
    slippage: float = 0.0,
    run_id: str | None = None,
    **unsupported: Any,
) -> ToolResult:
    """Run formed target weights through the canonical T+1 vector engine."""
    started = time.perf_counter()
    tool_name = "run_backtest"
    raw_args = {
        "target_weights": _artifact_label(target_weights),
        "price_panel": _artifact_label(price_panel),
        "open_panel": _artifact_label(open_panel) if open_panel is not None else None,
        "buyable": _artifact_label(buyable) if buyable is not None else None,
        "sellable": _artifact_label(sellable) if sellable is not None else None,
        "benchmark_returns": _artifact_label(benchmark_returns) if benchmark_returns is not None else None,
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "slippage": slippage,
        **unsupported,
    }

    def failure(code: str, message: str, *, normalized_args=None) -> ToolResult:
        return ToolResult.error(
            tool_name,
            normalized_args or raw_args,
            code,
            message,
            run_id=run_id,
            provenance={"module": "agent.tools"},
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    if unsupported:
        names = ", ".join(sorted(unsupported))
        return failure("unsupported_argument", f"unsupported argument(s): {names}")

    def cost_value(value: Any, default: float, name: str) -> float:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite non-negative number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return normalized

    try:
        defaults = BacktestConfig()
        normalized_buy_cost = cost_value(buy_cost, defaults.buy_cost, "buy_cost")
        normalized_sell_cost = cost_value(sell_cost, defaults.sell_cost, "sell_cost")
        normalized_slippage = cost_value(slippage, 0.0, "slippage")
        if open_panel is None:
            raise ValueError("open_panel is required because run_backtest enforces T+1 open execution")
        normalized_args = {
            **raw_args,
            "buy_cost": normalized_buy_cost,
            "sell_cost": normalized_sell_cost,
            "slippage": normalized_slippage,
            "t_plus_1": True,
        }
    except (TypeError, ValueError) as exc:
        return failure("invalid_backtest_args", str(exc))

    try:
        weights, weights_source = _load_table_artifact(target_weights, "target_weights")
        close, close_source = _load_table_artifact(price_panel, "price_panel")
        opens, open_source = _load_table_artifact(open_panel, "open_panel")
        if buyable is not None:
            buyable_frame, buyable_source = _load_table_artifact(buyable, "buyable")
        else:
            buyable_frame, buyable_source = None, None
        if sellable is not None:
            sellable_frame, sellable_source = _load_table_artifact(sellable, "sellable")
        else:
            sellable_frame, sellable_source = None, None
        benchmark, benchmark_source = _load_benchmark_returns(benchmark_returns)
        if weights.empty or close.empty or opens.empty:
            return failure("missing_data", "target weights, close, and open panels must be non-empty", normalized_args=normalized_args)
        if benchmark is not None and benchmark.dropna().empty:
            return failure("missing_data", "benchmark_returns has no usable values", normalized_args=normalized_args)

        config = BacktestConfig(
            buy_cost=normalized_buy_cost,
            sell_cost=normalized_sell_cost,
            slippage=normalized_slippage,
            t_plus_1=True,
        )
        result = canonical_run_backtest(
            weights,
            close,
            buyable=buyable_frame,
            sellable=sellable_frame,
            config=config,
            open_panel=opens,
        )
        if benchmark is not None and result.returns.index.intersection(benchmark.dropna().index).empty:
            return failure("invalid_backtest_args", "benchmark_returns has no dates overlapping the backtest", normalized_args=normalized_args)

        gross_returns = result.returns + result.cost
        net_metrics = _compact_metrics(calc_metrics(result.returns, benchmark=benchmark))
        gross_metrics = _compact_metrics(calc_metrics(gross_returns, benchmark=benchmark))
        return ToolResult(
            tool_name=tool_name,
            **({"run_id": run_id} if run_id is not None else {}),
            status="success",
            normalized_args=normalized_args,
            result={
                "observation": {
                    "actual_start": result.returns.index.min().strftime("%Y-%m-%d"),
                    "actual_end": result.returns.index.max().strftime("%Y-%m-%d"),
                    "trading_days": len(result.returns),
                    "symbols": close.shape[1],
                },
                "config": {
                    "t_plus_1": True,
                    "buy_cost": normalized_buy_cost,
                    "sell_cost": normalized_sell_cost,
                    "slippage": normalized_slippage,
                },
                "metrics": {"net": net_metrics, "gross": gross_metrics},
                "execution": {
                    "buy_days": int((result.buys > 0).sum()),
                    "sell_days": int((result.sells > 0).sum()),
                    "turnover_total": _scalar(result.turnover.sum()),
                    "cost_total": _scalar(result.cost.sum()),
                    "average_exposure": _scalar(result.weights.sum(axis=1).mean()),
                },
            },
            provenance={
                "modules": ["research.vector_backtest", "research.metrics"],
                "functions": ["run_backtest", "BacktestConfig", "calc_metrics"],
                "input_artifacts": {
                    "target_weights": weights_source,
                    "price_panel": close_source,
                    "open_panel": open_source,
                    "buyable": buyable_source,
                    "sellable": sellable_source,
                    "benchmark_returns": benchmark_source,
                },
                "t_plus_1_owned_by": "research.vector_backtest",
            },
            timing={"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
        return failure(
            "missing_data",
            f"backtest artifact/data unavailable: {type(exc).__name__}: {exc}",
            normalized_args=normalized_args,
        )
    except (TypeError, ValueError) as exc:
        return failure("invalid_backtest_args", str(exc), normalized_args=normalized_args)
    except Exception as exc:
        return failure(
            "internal_execution_error",
            f"run_backtest failed: {type(exc).__name__}: {exc}",
            normalized_args=normalized_args,
        )


__all__ = ["evaluate_factor", "inspect_universe", "run_backtest"]
