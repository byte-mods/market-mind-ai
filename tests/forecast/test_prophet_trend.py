"""F4 verification: Holt-Winters trend forecaster + linear-fallback path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.prophet_trend import (
    MIN_OBS,
    TrendForecaster,
    get_trend_forecaster,
)


def _ramp_ohlcv(n: int = 200, slope: float = 0.5, noise: float = 0.5,
                seed: int = 17) -> pd.DataFrame:
    """Linear ramp + small Gaussian noise — exact closed-form trend test."""
    rng = np.random.default_rng(seed)
    close = 100.0 + slope * np.arange(n) + rng.normal(0, noise, size=n)
    df = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.full(n, 1_000_000),
    })
    df.attrs["symbol"] = "RAMP"
    return df


def test_prophet_returns_horizon_rows() -> None:
    """Single-row forecast at horizon h ≈ extrapolated slope-line within noise."""
    df = _ramp_ohlcv(n=200, slope=0.5, noise=0.3)
    f = TrendForecaster()
    f.fit(df)
    r = f.predict(horizon=5)
    # On a clean ramp (slope 0.5), the 5-step-ahead point should be roughly
    # last_close + 5 * 0.5 = last_close + 2.5. Tolerate 2σ.
    last_close = df["close"].iloc[-1]
    expected = last_close + 5 * 0.5
    assert abs(r.point - expected) < 4.0, f"point {r.point} far from {expected}"
    assert r.lower_80 < r.point < r.upper_80
    assert r.lower_95 < r.lower_80
    assert r.upper_95 > r.upper_80


def test_prophet_disable_holidays() -> None:
    """The forecaster must not pull a holiday calendar from the system. We
    don't model India holidays anywhere; verify the fit completes without
    requiring locale data by feeding a series whose dates would otherwise
    intersect bank-holiday windows."""
    df = _ramp_ohlcv(n=120, slope=0.0, noise=1.5)
    df.index = pd.date_range("2025-12-15", periods=120, freq="B")  # spans new year
    f = TrendForecaster()
    f.fit(df)
    r = f.predict(horizon=3)
    # Sanity — no NaN bands
    assert not (np.isnan(r.point) or np.isnan(r.upper_80))


def test_prophet_falls_back_to_linear_on_short_series() -> None:
    """Below MIN_OBS, the linear-regression fallback must engage."""
    df = _ramp_ohlcv(n=MIN_OBS - 10, slope=1.0, noise=0.1)
    f = TrendForecaster()
    f.fit(df)
    r = f.predict(horizon=2)
    assert r.components["trend"]["fallback"] is True
    assert r.components["trend"]["method"] == "linear_regression"
    # Linear fallback on a clean ramp recovers slope ≈ 1
    last_close = df["close"].iloc[-1]
    assert abs(r.point - (last_close + 2.0)) < 1.0


def test_prophet_predict_before_fit_raises() -> None:
    f = TrendForecaster()
    with pytest.raises(RuntimeError):
        f.predict(horizon=1)


def test_prophet_horizon_must_be_positive() -> None:
    df = _ramp_ohlcv()
    f = TrendForecaster()
    f.fit(df)
    with pytest.raises(ValueError):
        f.predict(horizon=0)


def test_prophet_bands_widen_with_horizon() -> None:
    df = _ramp_ohlcv(n=300)
    f = TrendForecaster()
    f.fit(df)
    r1 = f.predict(horizon=1)
    r10 = f.predict(horizon=10)
    width1 = r1.upper_80 - r1.lower_80
    width10 = r10.upper_80 - r10.lower_80
    assert width10 > width1


def test_prophet_singleton_returns_same_instance() -> None:
    assert get_trend_forecaster() is get_trend_forecaster()
