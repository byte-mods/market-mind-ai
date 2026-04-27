"""F2 verification: feature engineering math + schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.features import (
    FEATURE_COLS,
    build_features,
    feature_matrix,
)


def _ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic deterministic OHLCV series."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, size=n))
    high = close + np.abs(rng.normal(0, 0.5, size=n))
    low = close - np.abs(rng.normal(0, 0.5, size=n))
    return pd.DataFrame({
        "open": close, "high": high, "low": low,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, size=n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_build_features_schema() -> None:
    df = _ohlcv()
    out = build_features(df)
    for c in FEATURE_COLS:
        assert c in out.columns, f"missing {c}"
    # First row of every diff/rolling feature must be NaN
    assert pd.isna(out["return"].iloc[0])
    assert pd.isna(out["log_return"].iloc[0])


def test_features_align_against_known_input() -> None:
    """Feed close = [10, 11, 12, ..., 109]; the closed-form return on row 1 = 0.1."""
    n = 100
    close = pd.Series(np.arange(10, 10 + n, dtype=float))
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000_000),
    })
    out = build_features(df)
    # Pure linear ramp → return on row 1 = (11-10)/10 = 0.1
    assert abs(out["return"].iloc[1] - 0.1) < 1e-12
    # log_return on row 1 = log(11/10)
    assert abs(out["log_return"].iloc[1] - np.log(11 / 10)) < 1e-12
    # On a pure up-trend: RSI saturates near 100 by the time the EMA settles
    rsi_late = out["rsi_14"].iloc[60:].dropna()
    assert (rsi_late > 95).all(), f"RSI on monotonic ramp should sit near 100, got {rsi_late.head()}"


def test_feature_matrix_returns_2d_float32_array_no_nan() -> None:
    df = _ohlcv()
    X = feature_matrix(df)
    assert X.ndim == 2
    assert X.shape[1] == len(FEATURE_COLS)
    assert X.dtype == np.float32
    assert not np.isnan(X).any()
    # 50-row warm-up means we lose ~50 rows of the original 200
    assert X.shape[0] >= 140


def test_build_features_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing OHLCV columns"):
        build_features(df)


def test_volume_z_zero_when_volume_constant() -> None:
    """If volume never changes, the z-score should never blow up — division by zero
    handled by NaN propagation, not Inf."""
    n = 100
    close = pd.Series(100.0 + np.arange(n) * 0.1)
    df = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.full(n, 1_000_000),
    })
    out = build_features(df)
    z = out["volume_z"].dropna()
    assert not np.isinf(z).any(), "volume_z must not produce Inf on flat volume"
    # Should be NaN because std=0
    assert pd.isna(out["volume_z"].iloc[-1])
