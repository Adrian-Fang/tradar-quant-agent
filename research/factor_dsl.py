"""Declarative factor evaluator backed by ``data/factor_defs/*.yaml``.

Factor arithmetic lives only in this module. ``evaluate_factor_def_on_panels``
is the shared evaluator used by both research runners; ``run_factor_def`` is a
thin compatibility wrapper which loads the required panels and delegates to it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from research import panel
from utils import loader
from utils.financial_loader import FINANCIAL_FIELD_SPECS


FACTOR_DEF_DIR = Path("data/factor_defs")
ADJUSTED_PRICE_FIELDS = {"close", "_adjusted_open"}
RAW_PRICE_FIELDS = {"open", "high", "low", "volume", "amount"}
FUNDAMENTAL_FIELDS = set(panel.FUNDAMENTAL_FIELDS)
FINANCIAL_FIELDS = set(FINANCIAL_FIELD_SPECS)
GROUP_LEVELS = {"industry_lv1": 1, "industry_lv2": 2, "industry_lv3": 3}
_SAFE_EXPR_BUILTINS = {"abs": abs}
_WINDOW_STEP_KEYS = (
    "rolling_mean",
    "rolling_percentile",
    "rolling_std",
    "rolling_max",
    "rolling_sum",
    "returns",
)
_INTERNAL_DEFINITION_PATH = "__definition_path__"


def resolve_factor_def(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".yaml":
        if not path.exists():
            raise FileNotFoundError(f"factor definition does not exist: {path}")
        return path
    candidate = FACTOR_DEF_DIR / f"{path}.yaml"
    if not candidate.exists():
        available = sorted(item.stem for item in FACTOR_DEF_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"unknown factor {name_or_path!r}; available={available}"
        )
    return candidate


@lru_cache(maxsize=128)
def _load_factor_definition_cached(path_text: str) -> dict:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as stream:
        definition = yaml.safe_load(stream) or {}
    if not isinstance(definition, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    if "steps" not in definition and "combo" not in definition:
        raise ValueError(f"{path} must define steps or combo")
    definition[_INTERNAL_DEFINITION_PATH] = str(path)
    return definition


def load_factor_definition(name_or_path: str | Path) -> dict:
    """Load one factor definition and attach only its source-path metadata."""
    path = resolve_factor_def(name_or_path).resolve()
    return _load_factor_definition_cached(str(path))


def _combo_definition(parent: dict, item: dict) -> dict:
    parent_path = Path(parent[_INTERNAL_DEFINITION_PATH])
    return load_factor_definition(parent_path.parent / f"{item['factor']}.yaml")


def _infer_warmup_days(definition: dict) -> int:
    if "combo" in definition:
        return max(
            (_infer_warmup_days(_combo_definition(definition, item))
             for item in definition["combo"]),
            default=30,
        )
    max_window = 0
    for step in definition["steps"]:
        for key in _WINDOW_STEP_KEYS:
            if key in step:
                max_window = max(max_window, int(step[key]))
    return 30 if max_window == 0 else int(max_window * 7 / 5) + 30


def required_warmup_days(definitions: list[dict] | tuple[dict, ...]) -> int:
    return max((_infer_warmup_days(item) for item in definitions), default=30)


def _definition_dependencies(
    definition: dict,
    fields: set[str],
    groups: set[str],
) -> None:
    if "combo" in definition:
        for item in definition["combo"]:
            _definition_dependencies(
                _combo_definition(definition, item), fields, groups
            )
        return
    for step in definition["steps"]:
        if "field" in step:
            fields.add(str(step["field"]))
        if "expr" in step:
            for name in re.findall(r"[A-Za-z_]\w*", str(step["expr"])):
                if (
                    name in ADJUSTED_PRICE_FIELDS
                    or name in RAW_PRICE_FIELDS
                    or name in FUNDAMENTAL_FIELDS
                    or name in FINANCIAL_FIELDS
                ):
                    fields.add(name)
        if "group_rank_pct" in step:
            groups.add(str(step["group_rank_pct"]["by"]))


def factor_dependencies(
    definitions: list[dict] | tuple[dict, ...],
) -> tuple[set[str], set[str]]:
    fields: set[str] = set()
    groups: set[str] = set()
    for definition in definitions:
        _definition_dependencies(definition, fields, groups)
    return fields, groups


def load_factor_panels(
    definitions: list[dict] | tuple[dict, ...],
    start: str,
    end: str,
    *,
    extra_fields: set[str] | None = None,
    extra_groups: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Load every required market/fundamental/financial/group panel once."""
    fields, groups = factor_dependencies(definitions)
    fields |= set(extra_fields or ())
    groups |= set(extra_groups or ())
    unknown = fields - (
        ADJUSTED_PRICE_FIELDS
        | RAW_PRICE_FIELDS
        | FUNDAMENTAL_FIELDS
        | FINANCIAL_FIELDS
    )
    if unknown:
        raise ValueError(f"unsupported factor fields: {sorted(unknown)}")

    panels: dict[str, pd.DataFrame] = {}
    if fields & ADJUSTED_PRICE_FIELDS:
        adjusted = loader.load_prices(start, end, adjust="backward")
        for field in fields & ADJUSTED_PRICE_FIELDS:
            source = "open" if field == "_adjusted_open" else field
            panels[field] = panel.to_panel(adjusted, source)
    if fields & RAW_PRICE_FIELDS:
        raw = loader.load_prices(start, end, adjust="none")
        panels["_raw_close"] = panel.to_panel(raw, "close")
        for field in fields & RAW_PRICE_FIELDS:
            panels[field] = panel.to_panel(raw, field)

    fundamental_fields = sorted(fields & FUNDAMENTAL_FIELDS)
    if fundamental_fields:
        panels.update(
            panel.fundamental_panels(start, end, fields=fundamental_fields)
        )
    template = panels.get("close")
    if template is None and panels:
        template = next(iter(panels.values()))
    financial_fields = sorted(fields & FINANCIAL_FIELDS)
    if financial_fields:
        panels.update(
            panel.financial_panels(
                start,
                end,
                fields=financial_fields,
                template=template,
            )
        )
        if template is None:
            template = next(iter(panels.values()))
    if template is not None:
        panels = {
            name: values.reindex_like(template)
            for name, values in panels.items()
        }

    group_maps = {}
    for group in groups:
        if group not in GROUP_LEVELS:
            raise ValueError(f"unsupported factor group: {group}")
        values = loader.load_industry_map(level=GROUP_LEVELS[group])
        group_maps[group] = (
            values.reindex(template.columns) if template is not None else values
        )
    return panels, group_maps


def _safe_eval_expr(expr: str, context: dict):
    allowed_ops = set("+-*/() .0123456789")
    tokens = re.findall(r"[A-Za-z_]\w*|\S", expr)
    for token in tokens:
        if token.replace(".", "").replace("-", "").isdigit():
            continue
        if all(character in allowed_ops for character in token):
            continue
        if token in context or token in _SAFE_EXPR_BUILTINS:
            continue
        raise ValueError(
            f"expression contains disallowed identifier {token!r}; "
            f"available={list(context)}"
        )
    return eval(
        expr,
        {"__builtins__": {}},
        {**_SAFE_EXPR_BUILTINS, **context},
    )


def evaluate_factor_def_on_panels(
    definition: dict,
    panels: dict[str, pd.DataFrame],
    group_map: dict[str, pd.Series] | pd.Series | None = None,
) -> pd.DataFrame:
    """Evaluate a loaded factor definition using caller-provided PIT panels."""
    if "combo" in definition:
        total_weight = 0.0
        combined = None
        for item in definition["combo"]:
            component = evaluate_factor_def_on_panels(
                _combo_definition(definition, item), panels, group_map
            )
            component = component.rank(axis=1, pct=True)
            weight = float(item.get("weight", 1.0))
            combined = (
                component * weight
                if combined is None
                else combined + component * weight
            )
            total_weight += weight
        if combined is None or total_weight <= 0:
            raise ValueError("combo factor has no positive components")
        return combined / total_weight

    result = None
    context: dict[str, pd.DataFrame] = {}
    group_maps = (
        group_map
        if isinstance(group_map, dict)
        else {"industry_lv1": group_map}
        if group_map is not None
        else {}
    )

    for step in definition["steps"]:
        if "field" in step:
            field = str(step["field"])
            if field not in panels:
                raise KeyError(f"factor panel {field!r} was not loaded")
            result = panels[field]
            context[field] = result

        elif "expr" in step:
            expression = str(step["expr"])
            for name in set(re.findall(r"[A-Za-z_]\w*", expression)):
                if name in panels and name not in context:
                    context[name] = panels[name]
            result = _safe_eval_expr(expression, context)

        elif "filter" in step:
            condition = str(step["filter"]).strip()
            if condition.startswith(">"):
                result = result.where(result > float(condition[1:]))
            elif condition.startswith("<"):
                result = result.where(result < float(condition[1:]))
            else:
                raise ValueError(f"unsupported filter syntax: {condition}")

        elif "rolling_mean" in step:
            window = int(step["rolling_mean"])
            result = result.rolling(window, min_periods=window).mean()

        elif "rolling_std" in step:
            window = int(step["rolling_std"])
            result = result.rolling(window, min_periods=window).std()

        elif "rolling_max" in step:
            window = int(step["rolling_max"])
            result = result.rolling(window, min_periods=window).max()

        elif "rolling_sum" in step:
            window = int(step["rolling_sum"])
            result = result.rolling(window, min_periods=window).sum()

        elif "rolling_percentile" in step:
            window = int(step["rolling_percentile"])
            result = result.rolling(window, min_periods=window).apply(
                lambda values: (values.iloc[-1] <= values).mean(),
                raw=False,
            )

        elif "rank_pct" in step:
            result = result.rank(axis=1, pct=True)

        elif "group_rank_pct" in step:
            group_key = str(step["group_rank_pct"]["by"])
            groups = group_maps.get(group_key)
            if groups is None:
                raise KeyError(f"factor group map {group_key!r} was not loaded")
            groups = groups.reindex(result.columns)
            result = result.apply(
                lambda row: row.groupby(groups).rank(pct=True), axis=1
            )

        elif "invert" in step and step["invert"]:
            result = 1.0 - result

        elif "returns" in step:
            result = result.pct_change(
                int(step["returns"]), fill_method=None
            )

        elif "abs" in step and step["abs"]:
            result = result.abs()

        elif "log" in step and step["log"]:
            result = np.log(result.clip(lower=1e-10))

        elif "zscore" in step and step["zscore"]:
            mean = result.mean(axis=1)
            std = result.std(axis=1).replace(0, np.nan)
            result = result.sub(mean, axis=0).div(std, axis=0)

        elif "winsorize" in step:
            lower = step["winsorize"].get("lower", 1) / 100
            upper = step["winsorize"].get("upper", 99) / 100
            lo = result.quantile(lower, axis=1)
            hi = result.quantile(upper, axis=1)
            result = result.clip(lower=lo, upper=hi, axis=0)

        elif "clip" in step:
            result = result.clip(
                lower=step["clip"].get("lower"),
                upper=step["clip"].get("upper"),
            )

        elif "save" in step:
            context[str(step["save"])] = result

        else:
            raise ValueError(f"unknown factor step: {step}")

    if result is None or result.empty:
        raise ValueError("factor evaluation returned an empty panel")
    return result


def run_factor_def(
    yaml_path: str,
    observe_start: str,
    observe_end: str,
) -> pd.DataFrame:
    """Load required inputs once, evaluate once, then crop to observation dates."""
    definition = load_factor_definition(yaml_path)
    warmup_days = required_warmup_days([definition])
    load_start = (
        pd.Timestamp(observe_start) - pd.Timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")
    panels, groups = load_factor_panels(
        [definition], load_start, observe_end
    )
    result = evaluate_factor_def_on_panels(definition, panels, groups).loc[
        observe_start:observe_end
    ]
    if result.notna().sum().sum() == 0:
        raise ValueError(
            f"factor {yaml_path} is all-NaN in "
            f"[{observe_start}, {observe_end}] after {warmup_days} warmup days"
        )
    return result
