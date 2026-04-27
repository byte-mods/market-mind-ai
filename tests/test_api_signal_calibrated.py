"""C3 verification: GET /api/signal/{sym}/calibrated wire contract.

Stand-in pattern (matches test_api_altdata.py / test_api_forecast.py): build
a tiny FastAPI mirror of the live route logic and inject deterministic stubs
for fetcher + RL + regime + sentiment. Exercises the route's wire shape and
error paths without booting the full controller.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.meta_stacker import (
    get_meta_stacker,
    reset_meta_stacker_for_tests,
)


class _StubFetcher:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self.df = df

    def get_historical_data(self, sym: str, days: int = 365):
        return self.df


def _df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000_000),
    })


def _make_app(fetcher: _StubFetcher,
              rl_score: float = 0.0,
              regime: str = "range",
              sentiment: float = 0.0) -> FastAPI:
    """Build a minimal app whose calibrated route mirrors server.py logic."""
    app = FastAPI()

    @app.get("/api/signal/{sym}/calibrated")
    def calibrated(sym: str, horizon: int = 5):
        sym = sym.upper().strip()
        if not (1 <= horizon <= 10):
            return JSONResponse({"error": "bad horizon"}, status_code=400)
        df = fetcher.get_historical_data(sym)
        if df is None or len(df) < 80:
            return JSONResponse(
                {"error": f"insufficient history for {sym}", "symbol": sym},
                status_code=422,
            )
        # Synthetic forecast: pin point to last_close * 1.02 for determinism
        last_close = float(df["close"].iloc[-1])
        fr = ForecastResult(
            symbol=sym, horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
            point=last_close * 1.02,
            lower_80=last_close * 0.98, upper_80=last_close * 1.06,
            lower_95=last_close * 0.95, upper_95=last_close * 1.09,
            model="conformal",
            calibration={"method": "split_conformal", "alpha_95": 0.05},
        )
        forecast_return = fr.point / last_close - 1.0
        forecast_vol = (fr.upper_95 - fr.lower_95) / 2.0 / last_close
        features = {
            "forecast_return": forecast_return,
            "forecast_vol": forecast_vol,
            "rl_signal_score": rl_score,
            "regime_state": regime,
            "sentiment_tilt": sentiment,
        }
        probs = get_meta_stacker().predict_proba(features)
        return JSONResponse({
            "symbol": sym, "horizon_days": horizon, **probs,
            "expected_return": round(forecast_return, 6),
            "return_95ci": [
                round(fr.lower_95 / last_close - 1.0, 6),
                round(fr.upper_95 / last_close - 1.0, 6),
            ],
            "forecast": fr.to_dict(),
            "features": features,
        })

    return app


def setup_function() -> None:
    reset_meta_stacker_for_tests()


def test_api_signal_calibrated_route_returns_200() -> None:
    client = TestClient(_make_app(_StubFetcher(_df()), rl_score=0.6,
                                  regime="trending_bull", sentiment=0.3))
    r = client.get("/api/signal/RELIANCE/calibrated?horizon=5")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "RELIANCE"
    assert body["horizon_days"] == 5
    # Probabilities sum to 1 within numeric tolerance
    total = body["p_buy"] + body["p_sell"] + body["p_hold"]
    assert abs(total - 1.0) < 1e-6
    # 95% CI bracket is in return-space (not price-space)
    lo, hi = body["return_95ci"]
    assert lo < 0 < hi  # synthetic forecast: -5% lower, +9% upper
    # Forecast block carries the calibration metadata
    assert body["forecast"]["calibration"]["method"] == "split_conformal"


def test_signal_calibrated_handles_missing_history() -> None:
    client = TestClient(_make_app(_StubFetcher(None)))
    r = client.get("/api/signal/RELIANCE/calibrated")
    assert r.status_code == 422
    assert "insufficient" in r.json()["error"]


def test_signal_calibrated_rejects_bad_horizon() -> None:
    client = TestClient(_make_app(_StubFetcher(_df())))
    assert client.get("/api/signal/RELIANCE/calibrated?horizon=0").status_code == 400
    assert client.get("/api/signal/RELIANCE/calibrated?horizon=99").status_code == 400


def test_signal_calibrated_strong_bullish_features_yield_high_p_buy() -> None:
    client = TestClient(_make_app(_StubFetcher(_df()), rl_score=0.9,
                                  regime="trending_bull", sentiment=0.7))
    body = client.get("/api/signal/RELIANCE/calibrated").json()
    assert body["p_buy"] > body["p_sell"]


def test_signal_calibrated_features_block_includes_inputs() -> None:
    client = TestClient(_make_app(_StubFetcher(_df()), rl_score=-0.4,
                                  regime="crash", sentiment=-0.5))
    body = client.get("/api/signal/RELIANCE/calibrated").json()
    feat = body["features"]
    assert feat["rl_signal_score"] == -0.4
    assert feat["regime_state"] == "crash"
    assert feat["sentiment_tilt"] == -0.5


def test_signal_calibrated_short_history_rejected() -> None:
    """80-row threshold must be enforced."""
    short = _df(n=70)
    client = TestClient(_make_app(_StubFetcher(short)))
    r = client.get("/api/signal/RELIANCE/calibrated")
    assert r.status_code == 422
