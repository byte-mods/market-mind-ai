"""F1 verification: ForecastResult schema + Forecaster protocol shape."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from marketmind.ml.forecast.base import Band, Forecaster, ForecastResult


def _make_result(**overrides) -> ForecastResult:
    base = dict(
        symbol="RELIANCE", horizon_days=5,
        as_of=dt.datetime(2026, 4, 27, 12, tzinfo=dt.timezone.utc),
        point=2950.0,
        lower_80=2820.0, upper_80=3080.0,
        lower_95=2750.0, upper_95=3150.0,
        model="ensemble",
    )
    base.update(overrides)
    return ForecastResult(**base)


def test_forecastresult_serializable() -> None:
    fr = _make_result(
        regime_conditional={
            "bull": ForecastResult.make_band(2980, 2870, 3120, 2810, 3180),
            "bear": ForecastResult.make_band(2900, 2790, 3030, 2710, 3110),
        },
        components={"patchtst": {"point": 2945}, "garch_vol_1d": 0.018},
        calibration={"pi80_oos_coverage": 0.78},
    )
    d = fr.to_dict()
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["symbol"] == "RELIANCE"
    assert decoded["horizon_days"] == 5
    assert decoded["regime_conditional"]["bull"]["point"] == 2980
    assert decoded["regime_conditional"]["bear"]["lower_95"] == 2710
    assert decoded["calibration"]["pi80_oos_coverage"] == 0.78
    # tz-aware ISO
    assert decoded["as_of"].endswith("+00:00")


def test_forecastresult_regime_conditional_is_optional() -> None:
    fr = _make_result()
    d = fr.to_dict()
    assert d["regime_conditional"] is None


def test_forecastresult_is_frozen() -> None:
    fr = _make_result()
    with pytest.raises(Exception):
        fr.point = 9999  # type: ignore[misc]


def test_forecaster_protocol_shape() -> None:
    """Anything with `name`, `fit(df)`, `predict(h)` satisfies the Protocol."""

    class _Stub:
        name = "stub"

        def fit(self, df: pd.DataFrame) -> None:
            self._n = len(df)

        def predict(self, horizon: int) -> ForecastResult:
            return _make_result(horizon_days=horizon, model="stub")

    s = _Stub()
    assert isinstance(s, Forecaster)
    s.fit(pd.DataFrame({"close": [1, 2, 3]}))
    out = s.predict(7)
    assert out.horizon_days == 7
    assert out.model == "stub"


def test_band_to_dict_round_trip() -> None:
    b = Band(point=100, lower_80=95, upper_80=105, lower_95=90, upper_95=110)
    d = b.to_dict()
    assert d == {"point": 100, "lower_80": 95, "upper_80": 105,
                 "lower_95": 90, "upper_95": 110}
