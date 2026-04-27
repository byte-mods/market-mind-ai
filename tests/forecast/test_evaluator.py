"""F8 verification: PI-coverage harness math against deterministic stubs."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.evaluator import evaluate_pi_coverage


def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, size=n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000_000),
    })


class _OracleForecaster:
    """Cheats: the predict reads the actual future close it can see in the df.

    This exists only for the test — it produces 100% PI coverage by definition.
    """

    name = "oracle"

    def __init__(self, df: pd.DataFrame, horizon: int) -> None:
        self._df = df
        self._horizon = horizon
        self._train_rows = 0

    def fit(self, df: pd.DataFrame) -> None:
        self._train_rows = len(df)

    def predict(self, horizon: int) -> ForecastResult:
        actual = float(self._df["close"].iloc[self._train_rows - 1 + horizon])
        return ForecastResult(
            symbol="ORACLE", horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
            point=actual,
            lower_80=actual - 0.01, upper_80=actual + 0.01,
            lower_95=actual - 0.02, upper_95=actual + 0.02,
            model="oracle",
        )


class _PathologicalForecaster:
    """Always wrong by 1000 with band of ±0.001 — guaranteed 0% coverage."""

    name = "bad"

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def predict(self, horizon: int) -> ForecastResult:
        wrong = -1000.0
        return ForecastResult(
            symbol="BAD", horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
            point=wrong,
            lower_80=wrong - 0.001, upper_80=wrong + 0.001,
            lower_95=wrong - 0.002, upper_95=wrong + 0.002,
            model="bad",
        )


def test_evaluator_pi_coverage_calc() -> None:
    """Oracle → 100% coverage. Pathological → 0%."""
    df = _make_df(n=300)
    horizon = 5

    metrics_oracle = evaluate_pi_coverage(
        df=df, forecaster_factory=lambda: _OracleForecaster(df, horizon),
        horizon=horizon, n_slices=20, min_train_rows=60,
    )
    assert metrics_oracle["pi80_coverage"] == 1.0
    assert metrics_oracle["pi95_coverage"] == 1.0
    assert metrics_oracle["calibrated_at_pi80"] is True
    assert metrics_oracle["n_slices_evaluated"] == 20

    metrics_bad = evaluate_pi_coverage(
        df=df, forecaster_factory=_PathologicalForecaster,
        horizon=horizon, n_slices=20, min_train_rows=60,
    )
    assert metrics_bad["pi80_coverage"] == 0.0
    assert metrics_bad["pi95_coverage"] == 0.0
    assert metrics_bad["calibrated_at_pi80"] is False


def test_evaluator_skips_failing_anchors() -> None:
    """A factory that raises on every call is recorded as failed, not crashed."""

    def raising_factory():
        class _R:
            name = "raise"

            def fit(self, df):
                raise RuntimeError("nope")

            def predict(self, horizon):  # not reached
                ...
        return _R()

    df = _make_df()
    metrics = evaluate_pi_coverage(
        df=df, forecaster_factory=raising_factory,
        horizon=5, n_slices=10, min_train_rows=60,
    )
    assert metrics["n_slices_evaluated"] == 0
    assert metrics["n_slices_failed"] == 10


def test_evaluator_validates_short_history() -> None:
    df = _make_df(n=50)
    with pytest.raises(ValueError, match="Not enough rows"):
        evaluate_pi_coverage(
            df=df, forecaster_factory=lambda: _OracleForecaster(df, 5),
            horizon=5, n_slices=10, min_train_rows=60,
        )


def test_evaluator_requires_close_column() -> None:
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing 'close'"):
        evaluate_pi_coverage(
            df=df, forecaster_factory=lambda: _OracleForecaster(df, 1),
            horizon=1, n_slices=1, min_train_rows=1,
        )


def test_evaluator_progress_callback_invoked() -> None:
    df = _make_df(n=300)
    calls: list[tuple[int, int]] = []
    evaluate_pi_coverage(
        df=df, forecaster_factory=lambda: _OracleForecaster(df, 5),
        horizon=5, n_slices=10, min_train_rows=60,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert len(calls) >= 5
    # Total reported consistently
    assert all(c[1] == calls[0][1] for c in calls)
