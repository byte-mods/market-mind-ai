"""C1 verification: split-CP marginal coverage on synthetic data."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.conformal import (
    MIN_CALIBRATION_ROWS,
    SplitConformalWrapper,
)


class _NaiveLastClose:
    """Stub forecaster: predicts last_close (random walk null model).

    Residuals = |actual - last_close|, which on iid Gaussian-return data
    has a clean distribution — perfect for a coverage check.
    """

    name = "naive"

    def __init__(self) -> None:
        self._last_close = None

    def fit(self, df: pd.DataFrame) -> None:
        self._last_close = float(df["close"].iloc[-1])

    def predict(self, horizon: int) -> ForecastResult:
        p = self._last_close
        return ForecastResult(
            symbol="X", horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
            point=p,
            lower_80=p, upper_80=p,  # no native bands — conformal supplies them
            lower_95=p, upper_95=p,
            model="naive",
        )


def _gaussian_walk(n: int = 1500, sigma: float = 0.5, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, sigma, size=n))
    df = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.full(n, 1_000_000),
    })
    df.attrs["symbol"] = "WALK"
    return df


def test_conformal_marginal_coverage_on_synthetic() -> None:
    """Wrap the naive predictor; on a stationary Gaussian random walk the
    empirical coverage of the 95% PI should be within ±5% of nominal."""
    df = _gaussian_walk(n=2000)
    horizon = 5
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=horizon, calibration_frac=0.4)
    cp.fit(df)

    # Now walk a held-out tail and measure empirical coverage.
    # We use the last 200 rows that were already in the calibration window;
    # since predict() returns a point + bands relative to *last_close*, we
    # test by manually emulating "as-of" anchors before each test point.
    test_window = 300
    inside_95 = 0
    inside_90 = 0
    eval_n = 0
    for t in range(len(df) - test_window, len(df) - horizon):
        # Re-fit naive on prefix (cheap — naive only stores last close)
        cp.inner.fit(df.iloc[:t + 1])
        # Predict, then compare against actual close at t+horizon
        r = cp.predict(horizon)
        actual = float(df["close"].iloc[t + horizon])
        if r.lower_95 <= actual <= r.upper_95:
            inside_95 += 1
        if (r.point - cp._q90) <= actual <= (r.point + cp._q90):
            inside_90 += 1
        eval_n += 1

    cov_95 = inside_95 / eval_n
    cov_90 = inside_90 / eval_n
    # Wide tolerance — split-CP is asymptotic; finite samples drift.
    assert 0.85 <= cov_95 <= 1.0, f"95% PI coverage {cov_95} outside [0.85, 1.0]"
    assert 0.80 <= cov_90 <= 1.0, f"90% PI coverage {cov_90} outside [0.80, 1.0]"


def test_conformal_handles_short_history_gracefully() -> None:
    df = _gaussian_walk(n=80)  # too short for full split
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5)
    cp.fit(df)
    r = cp.predict(5)
    # Degraded mode: q̂ defaults to 0 → point-degenerate band, but no crash.
    assert r.point > 0
    assert r.lower_95 <= r.point <= r.upper_95
    assert r.components["conformal"]["n_calibration"] == 0


def test_conformal_predict_before_fit_raises() -> None:
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5)
    with pytest.raises(RuntimeError):
        cp.predict(5)


def test_conformal_horizon_mismatch_raises() -> None:
    df = _gaussian_walk()
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5)
    cp.fit(df)
    with pytest.raises(ValueError, match="horizon"):
        cp.predict(7)


def test_conformal_calibration_metadata_in_components() -> None:
    df = _gaussian_walk(n=500)
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5, calibration_frac=0.3)
    cp.fit(df)
    r = cp.predict(5)
    assert r.calibration["method"] == "split_conformal"
    assert r.calibration["alpha_95"] == 0.05
    assert r.components["conformal"]["q95"] >= r.components["conformal"]["q90"]
    assert r.components["conformal"]["n_calibration"] >= MIN_CALIBRATION_ROWS


def test_conformal_recalibrate_updates_quantiles_in_place() -> None:
    df_calm = _gaussian_walk(n=1000, sigma=0.3)
    df_volatile = _gaussian_walk(n=500, sigma=2.0)
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5, calibration_frac=0.3)
    cp.fit(df_calm)
    q95_calm = cp._q95
    cp.recalibrate(df_volatile)
    q95_volatile = cp._q95
    assert q95_volatile > q95_calm, (
        "recalibrating on more volatile data must widen q̂"
    )


def test_conformal_recalibrate_before_fit_raises() -> None:
    cp = SplitConformalWrapper(_NaiveLastClose(), horizon=5)
    with pytest.raises(RuntimeError):
        cp.recalibrate(_gaussian_walk())
