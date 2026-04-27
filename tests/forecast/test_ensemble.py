"""F6 verification: ensemble blending + regime-conditional output."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import pandas as pd
import pytest

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.ensemble import EnsembleForecaster, get_ensemble_forecaster


class _StubForecaster:
    """Hand-controllable stub. Stamps any field at construction time."""

    def __init__(self, name: str, point: float, lower80: float, upper80: float,
                 fallback: bool = False) -> None:
        self.name = name
        self._point = point
        self._lower80 = lower80
        self._upper80 = upper80
        self._fallback = fallback
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        return ForecastResult(
            symbol="STUB", horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
            point=self._point,
            lower_80=self._lower80, upper_80=self._upper80,
            lower_95=self._lower80 - 5, upper_95=self._upper80 + 5,
            model=self.name, regime_conditional=None,
            components={self.name: {"fallback": self._fallback, "point": self._point}},
            calibration={},
        )


def _df(n: int = 200, last_close: float = 100.0) -> pd.DataFrame:
    df = pd.DataFrame({
        "open":   [last_close] * n,
        "high":   [last_close + 1] * n,
        "low":    [last_close - 1] * n,
        "close":  [last_close] * n,
        "volume": [1_000_000] * n,
    })
    df.attrs["symbol"] = "STUB"
    return df


def test_ensemble_combines_three_components() -> None:
    """Ensemble emits one component dict per sub-forecaster."""
    garch = _StubForecaster("garch", point=100.0, lower80=95.0, upper80=105.0)
    trend = _StubForecaster("trend", point=110.0, lower80=100.0, upper80=120.0)
    patch = _StubForecaster("patchtst", point=105.0, lower80=95.0, upper80=115.0)

    e = EnsembleForecaster(garch=garch, trend=trend, patchtst=patch)
    e.fit(_df(last_close=100.0))
    r = e.predict(horizon=1)

    # Directional point = 0.6*patchtst + 0.4*trend = 0.6*105 + 0.4*110 = 107.0
    assert abs(r.point - 107.0) < 1e-6
    assert "garch" in r.components
    assert "trend" in r.components
    assert "patchtst" in r.components


def test_ensemble_regime_conditional_present() -> None:
    """Bull/bear branches must always be emitted, with shifted points."""
    garch = _StubForecaster("garch", 100, 95, 105)
    trend = _StubForecaster("trend", 100, 95, 105)
    patch = _StubForecaster("patchtst", 100, 95, 105)
    e = EnsembleForecaster(garch=garch, trend=trend, patchtst=patch)
    e.fit(_df())
    r = e.predict(horizon=4)
    assert r.regime_conditional is not None
    bull = r.regime_conditional["bull"]
    bear = r.regime_conditional["bear"]
    # Bull point > ensemble point > bear point
    assert bull.point > r.point > bear.point
    # And bands shift consistently
    assert bull.lower_80 > bear.lower_80
    # Heuristic provenance is labelled
    assert r.regime_conditional["_method"] == "heuristic_drift"


def test_ensemble_falls_back_when_directional_components_unreliable() -> None:
    """If both PatchTST and Trend are in fallback, point pins to last close."""
    garch = _StubForecaster("garch", 100, 95, 105)
    trend = _StubForecaster("trend", 999, 990, 1010, fallback=True)
    patch = _StubForecaster("patchtst", 999, 990, 1010, fallback=True)
    e = EnsembleForecaster(garch=garch, trend=trend, patchtst=patch)
    e.fit(_df(last_close=100.0))
    r = e.predict(horizon=2)
    # Point should be last close, NOT the components' bogus 999
    assert abs(r.point - 100.0) < 1e-6


def test_ensemble_band_anchored_to_ensemble_point_when_garch_ok() -> None:
    """GARCH owns the band logic; bands re-anchor at the ensemble point."""
    # GARCH says: point=100, ±5%. Trend/PatchTST say: point=110.
    garch = _StubForecaster("garch", 100, 95, 105)
    trend = _StubForecaster("trend", 110, 100, 120)
    patch = _StubForecaster("patchtst", 110, 100, 120)
    e = EnsembleForecaster(garch=garch, trend=trend, patchtst=patch)
    e.fit(_df())
    r = e.predict(horizon=1)
    # Ensemble point = 0.6*110 + 0.4*110 = 110.
    # GARCH ratio: lower_80/garch.point = 0.95 → ensemble lower_80 = 110*0.95 = 104.5
    assert abs(r.lower_80 - 104.5) < 1e-6
    assert abs(r.upper_80 - 115.5) < 1e-6


def test_ensemble_falls_back_on_garch_unavailable() -> None:
    """If GARCH is in fallback, bands derived from widest of remaining."""
    garch = _StubForecaster("garch", 100, 99, 101, fallback=True)  # narrow but flagged
    trend = _StubForecaster("trend", 100, 95, 105)
    patch = _StubForecaster("patchtst", 100, 90, 110)  # widest
    e = EnsembleForecaster(garch=garch, trend=trend, patchtst=patch)
    e.fit(_df())
    r = e.predict(horizon=1)
    # Bands should reflect the wider patchtst's 80% PI half-width = 10
    assert abs((r.upper_80 - r.lower_80) - 20.0) < 1e-6


def test_ensemble_fit_validates_close() -> None:
    e = EnsembleForecaster(
        _StubForecaster("garch", 1, 0, 2),
        _StubForecaster("trend", 1, 0, 2),
        _StubForecaster("patchtst", 1, 0, 2),
    )
    with pytest.raises(ValueError):
        e.fit(pd.DataFrame({"open": [1, 2, 3]}))


def test_ensemble_predict_before_fit_raises() -> None:
    e = EnsembleForecaster(
        _StubForecaster("garch", 1, 0, 2),
        _StubForecaster("trend", 1, 0, 2),
        _StubForecaster("patchtst", 1, 0, 2),
    )
    with pytest.raises(RuntimeError):
        e.predict(horizon=1)


def test_ensemble_singleton() -> None:
    assert get_ensemble_forecaster() is get_ensemble_forecaster()
