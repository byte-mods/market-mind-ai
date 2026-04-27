"""F5 verification: PatchTST forward shape + overfit on synthetic signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.patchtst import (
    PATCH,
    PatchTSTForecaster,
    WINDOW,
    get_patchtst_forecaster,
    _build_torch_model,
)
from marketmind.ml.forecast.features import FEATURE_COLS


def _signal_ohlcv(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """Sine + drift + small noise — a deterministic but non-trivial pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100.0 + 0.05 * t + 5.0 * np.sin(t / 10.0) + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, size=n),
    })
    df.attrs["symbol"] = "SIG"
    return df


def test_patchtst_forward_shape() -> None:
    """Pure shape check on the bare model — no fit, just forward."""
    import torch

    Model = _build_torch_model()
    n_features = len(FEATURE_COLS)
    horizon = 5
    m = Model(n_features=n_features, horizon=horizon)
    x = torch.randn(3, WINDOW, n_features)  # batch of 3
    y = m(x)
    assert tuple(y.shape) == (3, horizon)


def test_patchtst_window_divisible_by_patch() -> None:
    assert WINDOW % PATCH == 0


def test_patchtst_overfits_synthetic() -> None:
    """Train on 400 rows of a clean pattern, predict, assert point is in
    the right ballpark and bands non-degenerate."""
    df = _signal_ohlcv()
    f = PatchTSTForecaster(horizon_max=5)
    f.fit(df)
    r = f.predict(horizon=5)
    last_close = float(df["close"].iloc[-1])
    # On a clean signal: point should be within ±15% of last close
    # (loose bound; the test guards "did training do anything", not accuracy).
    assert abs(r.point - last_close) / last_close < 0.15
    assert r.lower_80 < r.point < r.upper_80
    assert r.lower_95 < r.lower_80 and r.upper_95 > r.upper_80
    assert r.components["patchtst"]["fallback"] is False
    # Training must have actually run (≥ 1 epoch)
    assert r.components["patchtst"]["epochs"] >= 1


def test_patchtst_fallback_on_short_series() -> None:
    """With fewer rows than WINDOW + horizon_max + 10, training is skipped."""
    df = _signal_ohlcv(n=WINDOW + 5)
    f = PatchTSTForecaster(horizon_max=5)
    f.fit(df)
    r = f.predict(horizon=3)
    assert r.components["patchtst"]["fallback"] is True
    # Point falls back to last close
    assert abs(r.point - float(df["close"].iloc[-1])) < 0.01


def test_patchtst_predict_before_fit_raises() -> None:
    f = PatchTSTForecaster()
    with pytest.raises(RuntimeError):
        f.predict(horizon=1)


def test_patchtst_horizon_validation() -> None:
    df = _signal_ohlcv()
    f = PatchTSTForecaster(horizon_max=5)
    f.fit(df)
    with pytest.raises(ValueError):
        f.predict(horizon=0)
    with pytest.raises(ValueError):
        f.predict(horizon=10)


def test_patchtst_singleton_returns_same_instance() -> None:
    assert get_patchtst_forecaster() is get_patchtst_forecaster()
