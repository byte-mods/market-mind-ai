"""F3 verification: GARCH fits synthetic data and falls back on short series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.garch import GarchForecaster, MIN_OBS, get_garch_forecaster


def _ohlcv_with_garch_signal(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generate a price series whose log-returns have time-varying volatility.

    Two-regime: low-vol (σ=0.5%) for first half, high-vol (σ=2%) for second.
    GARCH should pick this up and the conditional variance should be elevated
    near the end of the series.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    rets_low = rng.normal(0.0, 0.005, size=half)
    rets_high = rng.normal(0.0, 0.02, size=n - half)
    log_rets = np.concatenate([rets_low, rets_high])
    close = 100.0 * np.exp(np.cumsum(log_rets))
    df = pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, size=n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    df.attrs["symbol"] = "SYNTH"
    return df


def test_garch_fits_synthetic() -> None:
    """On a 500-row series with regime-shifting vol, GARCH should:
    - Not fall back (n >> MIN_OBS).
    - Produce a positive sigma forecast.
    - Wider bands on a longer horizon than a shorter one.
    """
    df = _ohlcv_with_garch_signal()
    g = GarchForecaster()
    g.fit(df)
    r1 = g.predict(horizon=1)
    r5 = g.predict(horizon=5)

    assert r1.components["garch"]["fallback"] is False
    assert r1.components["garch"]["sigma_1d_pct"] > 0
    # Bands must straddle the point estimate
    assert r1.lower_80 < r1.point < r1.upper_80
    assert r1.lower_95 < r1.lower_80
    assert r1.upper_95 > r1.upper_80
    # Multi-step variance > 1-step variance — bands wider for h=5
    assert (r5.upper_80 - r5.lower_80) > (r1.upper_80 - r1.lower_80)


def test_garch_fallback_on_short_series() -> None:
    """<MIN_OBS rows must trigger fallback path, still emit a valid result."""
    n = MIN_OBS - 10
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000_000),
    })
    g = GarchForecaster()
    g.fit(df)
    r = g.predict(horizon=1)
    assert r.components["garch"]["fallback"] is True
    assert r.components["garch"]["n_obs"] == n - 1   # one row consumed by diff
    # Result is still valid — bands non-degenerate
    assert r.lower_80 < r.point < r.upper_80


def test_garch_predict_before_fit_raises() -> None:
    g = GarchForecaster()
    with pytest.raises(RuntimeError):
        g.predict(horizon=1)


def test_garch_horizon_must_be_positive() -> None:
    df = _ohlcv_with_garch_signal(n=120)
    g = GarchForecaster()
    g.fit(df)
    with pytest.raises(ValueError):
        g.predict(horizon=0)


def test_garch_emits_forecastresult_with_correct_symbol() -> None:
    df = _ohlcv_with_garch_signal()
    df.attrs["symbol"] = "RELIANCE"
    g = GarchForecaster()
    g.fit(df)
    r = g.predict(horizon=3)
    assert r.symbol == "RELIANCE"
    assert r.horizon_days == 3
    assert r.model == "garch"


def test_garch_singleton_returns_same_instance() -> None:
    assert get_garch_forecaster() is get_garch_forecaster()
