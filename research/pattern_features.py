"""Pure, parameterized price/volume measurements for event patterns.

The module deliberately contains measurements only.  Strategy-specific event
state machines and thresholds stay in their owning scripts.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "close_position",
    "event_window_range",
    "event_window_return",
    "drop_depth",
    "drop_duration",
    "drop_speed",
    "support_distance",
    "nearest_support_distance",
    "relative_volume",
    "rolling_percentile",
    "min_consecutive_mean",
]


def event_window_return(prices, start_i: int, end_i: int) -> float:
    """Return from ``prices[start_i]`` to ``prices[end_i]`` (both included)."""
    start, end = float(np.asarray(prices, dtype=float)[start_i]), float(np.asarray(prices, dtype=float)[end_i])
    return float(end / start - 1.0) if np.isfinite(start) and np.isfinite(end) and start > 0 else np.nan


def event_window_range(high, low, start_i: int, end_i: int, denominator: float | None = None) -> float:
    """High-low range over an inclusive event window, optionally normalized."""
    highs = np.asarray(high, dtype=float)[start_i : end_i + 1]
    lows = np.asarray(low, dtype=float)[start_i : end_i + 1]
    if not np.isfinite(highs).any() or not np.isfinite(lows).any():
        return np.nan
    value = float(np.nanmax(highs) - np.nanmin(lows))
    if denominator is None:
        return value
    return float(value / denominator) if np.isfinite(denominator) and denominator != 0 else np.nan


def drop_depth(peak_price: float, low_price: float) -> float:
    """Return from a peak price to a later low price."""
    return float(low_price / peak_price - 1.0) if np.isfinite(peak_price) and np.isfinite(low_price) and peak_price > 0 else np.nan


def drop_duration(peak_i: int, low_i: int) -> int | None:
    """Trading-session distance between two indexed event points."""
    return int(low_i - peak_i) if peak_i is not None and low_i is not None else None


def drop_speed(depth: float, duration: int | float) -> float:
    """Average per-session drop, using the supplied depth and duration."""
    return float(depth / duration) if np.isfinite(depth) and np.isfinite(duration) and duration > 0 else np.nan


def support_distance(price: float, level: float) -> float:
    """Absolute relative distance of a price from an objective level."""
    return float(abs(price / level - 1.0)) if np.isfinite(price) and np.isfinite(level) and level > 0 else np.nan


def nearest_support_distance(low: float, close: float, level: float) -> float:
    """Nearest relative distance among a bar's low and close to a support line."""
    values = [support_distance(low, level), support_distance(close, level)]
    return float(np.nanmin(values)) if np.isfinite(values).any() else np.nan


def relative_volume(volume: float, reference_volume: float) -> float:
    """Volume divided by a supplied reference volume."""
    return float(volume / reference_volume) if np.isfinite(volume) and np.isfinite(reference_volume) and reference_volume > 0 else np.nan


def close_position(close, high, low):
    """Close location in a bar's high-low range; zero-range bars become NaN."""
    close, high, low = (np.asarray(x, dtype=float) for x in (close, high, low))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (close - low) / (high - low)
    return np.where((high > low) & np.isfinite(result), result, np.nan)


def rolling_percentile(values, window: int, min_periods: int | None = None) -> np.ndarray:
    """Trailing percentile rank of each value within its finite window."""
    values = np.asarray(values, dtype=float)
    minimum = window if min_periods is None else min_periods
    out = np.full(values.shape, np.nan, dtype=float)
    for i, value in enumerate(values):
        left = max(0, i - window + 1)
        sample = values[left : i + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < minimum or not np.isfinite(value) or not len(sample):
            continue
        out[i] = float(np.mean(sample <= value))
    return out


def min_consecutive_mean(values, window: int, start_i: int = 0, end_i: int | None = None) -> tuple[float, int | None]:
    """Minimum finite rolling mean and its ending index in an inclusive window."""
    values = np.asarray(values, dtype=float)
    end_i = len(values) - 1 if end_i is None else end_i
    best, best_i = np.nan, None
    for i in range(start_i + window - 1, end_i + 1):
        chunk = values[i - window + 1 : i + 1]
        if not np.isfinite(chunk).all():
            continue
        current = float(np.mean(chunk))
        if not np.isfinite(best) or current < best:
            best, best_i = current, i
    return best, best_i


if __name__ == "__main__":
    assert np.isclose(event_window_return([10.0, 11.0], 0, 1), 0.1)
    assert np.isclose(drop_speed(drop_depth(10.0, 8.0), 2), -0.1)
    assert np.isclose(nearest_support_distance(9.9, 10.1, 10.0), 0.01)
    print("PATTERN_FEATURES_CHECK=PASS")
