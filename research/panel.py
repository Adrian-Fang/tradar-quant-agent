# research/panel.py
"""把 loader 长表转换为研究框架使用的宽表，并集中定义研究 mask。"""

from functools import lru_cache

import numpy as np
import pandas as pd

from utils import loader
from utils.financial_loader import (
    FINANCIAL_FIELD_SPECS,
    load_pit_financial_events,
)


FUNDAMENTAL_FIELDS = (
    "is_st",
    "is_trading",
    "turnover_rate",
    "market_cap",
    "pe",
    "pb",
    "exchange_change_pct",
)
LIMIT_HIT_TOLERANCE_PCT = 0.2

# 所有涨跌停数字和生效日期只在这里定义。单一 limit_pct 无法表达旧制 IPO
# 首日的非对称上下限，因此这些首日也按“无可靠统一限制”返回 NaN。
PRICE_LIMIT_RULES = {
    "main": {"normal": 10.0, "st": 5.0},
    "chinext": {
        "reform_date": pd.Timestamp("2020-08-24"),
        "before_normal": 10.0,
        "before_st": 5.0,
        "after": 20.0,
        "registration_ipo_no_limit_days": 5,
    },
    "star": {
        "effective_date": pd.Timestamp("2019-07-22"),
        "limit": 20.0,
        "ipo_no_limit_days": 5,
    },
    "bse": {
        "effective_date": pd.Timestamp("2021-11-15"),
        "limit": 30.0,
        "ipo_no_limit_days": 1,
    },
    "main_registration": {
        "effective_date": pd.Timestamp("2023-04-10"),
        "ipo_no_limit_days": 5,
    },
}


def to_panel(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return df[field].unstack("symbol").sort_index()


def price_panel(start, end, field="close", adjust="backward") -> pd.DataFrame:
    """价格宽表；研究收益默认使用运行时生成的后复权价格。"""
    return to_panel(loader.load_prices(start, end, adjust=adjust), field)


def raw_price_panel(start, end, field="close") -> pd.DataFrame:
    """不复权价格宽表。"""
    return to_panel(loader.load_prices(start, end, adjust="none"), field)


def _financial_events_to_panels(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    fields,
) -> dict[str, pd.DataFrame]:
    panels = {
        field: pd.DataFrame(np.nan, index=dates, columns=symbols)
        for field in fields
    }
    by_statement = {}
    for field in fields:
        by_statement.setdefault(FINANCIAL_FIELD_SPECS[field][0], []).append(field)

    for statement, statement_fields in by_statement.items():
        field_events = events.loc[events["_statement"].eq(statement)]
        for symbol, history in field_events.groupby("symbol", sort=False):
            if symbol not in symbols:
                continue
            history = history.sort_values("available_date").drop_duplicates(
                "available_date", keep="last"
            )
            positions = np.searchsorted(
                history["available_date"].to_numpy(dtype="datetime64[ns]"),
                dates.to_numpy(dtype="datetime64[ns]"),
                side="right",
            ) - 1
            valid = positions >= 0
            for field in statement_fields:
                values = np.full(len(dates), np.nan)
                values[valid] = history[field].to_numpy(dtype=float)[positions[valid]]
                panels[field][symbol] = values
    return panels


def financial_panels(
    start,
    end,
    fields,
    *,
    template: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """公告日 PIT 财务宽表；公告前为空，修订从其可用日起生效。"""
    selected = [fields] if isinstance(fields, str) else list(fields)
    selected = list(dict.fromkeys(selected))
    unknown = sorted(set(selected) - set(FINANCIAL_FIELD_SPECS))
    if unknown:
        raise ValueError(f"unknown financial fields: {unknown}")

    if template is None:
        template = raw_price_panel(start, end)
    events = load_pit_financial_events(start, end, selected)
    return _financial_events_to_panels(
        events, template.index, template.columns, selected
    )


@lru_cache(maxsize=8)
def _load_fundamentals(start, end) -> pd.DataFrame:
    return loader.load_fundamentals(start, end)


def fundamental_panels(start, end, fields=None) -> dict[str, pd.DataFrame]:
    """一次读取基本面长表，并转换指定字段。"""
    selected = list(FUNDAMENTAL_FIELDS if fields is None else fields)
    unknown = sorted(set(selected) - set(FUNDAMENTAL_FIELDS))
    if unknown:
        raise ValueError(f"unknown fundamental fields: {unknown}")
    fundamentals = _load_fundamentals(start, end)
    return {field: to_panel(fundamentals, field) for field in selected}


def field_panel(start, end, field: str) -> pd.DataFrame:
    """因子 DSL 使用的单字段转换辅助。"""
    return fundamental_panels(start, end, fields=[field])[field]


def is_trading_mask(start, end) -> pd.DataFrame:
    """只表示 hist_ext.is_trading，不混入任何选股或涨跌停规则。"""
    return field_panel(start, end, "is_trading").fillna(False).astype(bool)


@lru_cache(maxsize=1)
def _stock_info() -> pd.DataFrame:
    return loader.load_stock_info().set_index("symbol")


def _listing_age_panel(
    index: pd.DatetimeIndex, columns: pd.Index
) -> pd.DataFrame:
    """每个日期距上市日的交易日计数，上市首个交易日为 1。"""
    # hist_ext 可能先于 history 更新；把当前面板日期并入日历，避免新数据短暂
    # 不同步时把合法交易日误判为非法日期。
    calendar = loader.load_trading_calendar().union(pd.DatetimeIndex(index)).sort_values()
    date_ordinals = calendar.get_indexer(pd.DatetimeIndex(index))

    list_dates = _stock_info()["list_date"].reindex(columns)
    list_ordinals = np.searchsorted(
        calendar.to_numpy(), list_dates.to_numpy(dtype="datetime64[ns]"), side="left"
    )
    missing = list_dates.isna().to_numpy()
    # 数据库交易日历起点以前已上市的股票显然已越过新股阶段。
    before_calendar = (
        list_dates.notna().to_numpy()
        & (list_dates.to_numpy(dtype="datetime64[ns]") < calendar[0].to_datetime64())
    )
    list_ordinals = list_ordinals.astype(float)
    list_ordinals[before_calendar] = -100_000
    ages = date_ordinals[:, None] - list_ordinals[None, :] + 1
    ages[:, missing] = np.nan
    return pd.DataFrame(ages, index=index, columns=columns)


def eligible_universe_mask(
    start,
    end,
    exclude_st=True,
    min_turnover_rate=1.0,
    min_listed_days=20,
) -> pd.DataFrame:
    """决定股票能否进入新目标持仓，不代表已有持仓能否买卖。"""
    if min_listed_days < 0:
        raise ValueError("min_listed_days must be non-negative")
    panels = fundamental_panels(
        start, end, fields=["is_trading", "is_st", "turnover_rate"]
    )
    trading = panels["is_trading"].fillna(False).astype(bool)
    eligible = trading & panels["turnover_rate"].ge(min_turnover_rate).fillna(False)
    if exclude_st:
        eligible &= ~panels["is_st"].fillna(False).astype(bool)
    if min_listed_days:
        ages = _listing_age_panel(eligible.index, eligible.columns)
        eligible &= ages.ge(min_listed_days)
    return eligible.fillna(False).astype(bool)


def _board_masks(columns: pd.Index) -> dict[str, np.ndarray]:
    symbols = columns.astype(str)
    return {
        "main": symbols.str.startswith(
            ("000", "001", "002", "003", "600", "601", "603", "605")
        ),
        "chinext": symbols.str.startswith(("300", "301")),
        "star": symbols.str.startswith(("688", "689")),
        "bse": symbols.str.startswith(("4", "8", "920")),
    }


@lru_cache(maxsize=8)
def price_limit_pct_panel(start, end) -> pd.DataFrame:
    """按板块、ST、上市日和历史生效日期生成每日涨跌幅限制百分数。

    NaN 表示该日没有统一涨跌幅限制，或现有字段不能可靠判断。
    """
    panels = fundamental_panels(start, end, fields=["is_st", "is_trading"])
    trading = panels["is_trading"].fillna(False).astype(bool)
    is_st = panels["is_st"]
    dates, columns = trading.index, trading.columns
    board = _board_masks(columns)
    date_grid = dates.to_numpy(dtype="datetime64[ns]")[:, None]
    limits = np.full(trading.shape, np.nan, dtype=float)

    main = np.broadcast_to(board["main"], trading.shape)
    known_st = np.broadcast_to(is_st.notna().to_numpy(), trading.shape)
    st_values = is_st.fillna(False).to_numpy(dtype=bool)
    limits[main & known_st & ~st_values] = PRICE_LIMIT_RULES["main"]["normal"]
    limits[main & known_st & st_values] = PRICE_LIMIT_RULES["main"]["st"]

    chinext = np.broadcast_to(board["chinext"], trading.shape)
    after_chinext_reform = date_grid >= PRICE_LIMIT_RULES["chinext"][
        "reform_date"
    ].to_datetime64()
    limits[chinext & after_chinext_reform] = PRICE_LIMIT_RULES["chinext"]["after"]
    limits[chinext & ~after_chinext_reform & known_st & ~st_values] = (
        PRICE_LIMIT_RULES["chinext"]["before_normal"]
    )
    limits[chinext & ~after_chinext_reform & known_st & st_values] = (
        PRICE_LIMIT_RULES["chinext"]["before_st"]
    )

    star = np.broadcast_to(board["star"], trading.shape)
    star_effective = date_grid >= PRICE_LIMIT_RULES["star"][
        "effective_date"
    ].to_datetime64()
    limits[star & star_effective] = PRICE_LIMIT_RULES["star"]["limit"]

    bse = np.broadcast_to(board["bse"], trading.shape)
    bse_effective = date_grid >= PRICE_LIMIT_RULES["bse"][
        "effective_date"
    ].to_datetime64()
    limits[bse & bse_effective] = PRICE_LIMIT_RULES["bse"]["limit"]

    ages = _listing_age_panel(dates, columns).to_numpy()
    list_dates = _stock_info()["list_date"].reindex(columns)
    list_date_grid = np.broadcast_to(
        list_dates.to_numpy(dtype="datetime64[ns]"), trading.shape
    )

    # 注册制主板、创业板和科创板上市后五个交易日无涨跌幅限制。
    main_registration = (
        main
        & (list_date_grid >= PRICE_LIMIT_RULES["main_registration"][
            "effective_date"
        ].to_datetime64())
        & (ages <= PRICE_LIMIT_RULES["main_registration"]["ipo_no_limit_days"])
    )
    chinext_registration = (
        chinext
        & (list_date_grid >= PRICE_LIMIT_RULES["chinext"][
            "reform_date"
        ].to_datetime64())
        & (ages <= PRICE_LIMIT_RULES["chinext"]["registration_ipo_no_limit_days"])
    )
    star_ipo = star & (ages <= PRICE_LIMIT_RULES["star"]["ipo_no_limit_days"])
    bse_ipo = bse & (ages <= PRICE_LIMIT_RULES["bse"]["ipo_no_limit_days"])

    # 旧制 IPO 首日上下限不对称，不能用一个 limit_pct 可靠表达。
    old_ipo_first_day = (ages == 1) & ~(main_registration | chinext_registration | star_ipo | bse_ipo)
    no_uniform_limit = (
        main_registration
        | chinext_registration
        | star_ipo
        | bse_ipo
        | old_ipo_first_day
    )
    limits[no_uniform_limit] = np.nan
    limits[~trading.to_numpy()] = np.nan
    return pd.DataFrame(limits, index=dates, columns=columns)


@lru_cache(maxsize=8)
def _execution_masks(start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels = fundamental_panels(
        start, end, fields=["is_trading", "exchange_change_pct"]
    )
    trading = panels["is_trading"].fillna(False).astype(bool)
    change = panels["exchange_change_pct"]
    limit = price_limit_pct_panel(start, end)
    at_limit_up = change.ge(limit - LIMIT_HIT_TOLERANCE_PCT) & limit.notna()
    at_limit_down = change.le(-limit + LIMIT_HIT_TOLERANCE_PCT) & limit.notna()
    return trading & ~at_limit_up, trading & ~at_limit_down


def buyable_mask(start, end) -> pd.DataFrame:
    return _execution_masks(start, end)[0]


def sellable_mask(start, end) -> pd.DataFrame:
    return _execution_masks(start, end)[1]


def index_return(code: str, start=None, end=None) -> pd.Series:
    return loader.load_index(code, start, end).pct_change()
