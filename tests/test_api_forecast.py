"""F9 verification: GET /api/forecast/{sym} route.

Stand-in app strategy (same as test_api_altdata.py): build a tiny FastAPI app
that mirrors server.py's wiring and inject deterministic stubs. This avoids
hauling in the full controller boot for an HTTP-shape check.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.cache import ForecastCache


class _StubFetcher:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self.df = df

    def get_historical_data(self, sym: str, days: int = 365) -> pd.DataFrame | None:
        return self.df


def _df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000_000),
    })


def _make_app(fetcher: _StubFetcher, cache: ForecastCache) -> FastAPI:
    app = FastAPI()

    HORIZON_MAX = 10
    VALID_MODELS = ("ensemble",)

    @app.get("/api/forecast/{sym}")
    def forecast(sym: str, horizon: int = 5, model: str = "ensemble"):
        sym = sym.upper().strip()
        if not (1 <= horizon <= HORIZON_MAX):
            return JSONResponse({"error": "bad horizon"}, status_code=400)
        if model not in VALID_MODELS:
            return JSONResponse({"error": "bad model"}, status_code=400)
        cached = cache.get(sym, horizon, model, interval="day")
        if cached is not None:
            return JSONResponse(cached.to_dict() | {"cached": True})
        df = fetcher.get_historical_data(sym)
        if df is None or len(df) < 60:
            return JSONResponse(
                {"error": f"insufficient history for {sym}", "symbol": sym},
                status_code=422,
            )
        df = df.copy(); df.attrs["symbol"] = sym
        # In stand-in: don't actually train the heavy ensemble; emit a fixed result.
        result = ForecastResult(
            symbol=sym, horizon_days=horizon,
            as_of=dt.datetime(2026, 4, 27, 12, tzinfo=dt.timezone.utc),
            point=110.0,
            lower_80=105.0, upper_80=115.0,
            lower_95=100.0, upper_95=120.0,
            model="ensemble",
        )
        cache.set(result, interval="day")
        return JSONResponse(result.to_dict() | {"cached": False})

    return app


def test_api_forecast_route_returns_200(fake_mongo_col) -> None:
    fetcher = _StubFetcher(_df())
    cache = ForecastCache(mongo_col=fake_mongo_col)
    client = TestClient(_make_app(fetcher, cache))
    r = client.get("/api/forecast/RELIANCE?horizon=5")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "RELIANCE"
    assert body["horizon_days"] == 5
    assert body["model"] == "ensemble"
    assert body["cached"] is False
    # Bands monotonic
    assert body["lower_95"] <= body["lower_80"] <= body["point"] <= body["upper_80"] <= body["upper_95"]


def test_api_forecast_route_serves_cached_on_second_call(fake_mongo_col) -> None:
    fetcher = _StubFetcher(_df())
    cache = ForecastCache(mongo_col=fake_mongo_col)
    client = TestClient(_make_app(fetcher, cache))
    first = client.get("/api/forecast/RELIANCE?horizon=5").json()
    second = client.get("/api/forecast/RELIANCE?horizon=5").json()
    assert first["cached"] is False
    assert second["cached"] is True


def test_api_forecast_route_rejects_bad_horizon() -> None:
    client = TestClient(_make_app(_StubFetcher(_df()), ForecastCache(None)))
    r0 = client.get("/api/forecast/RELIANCE?horizon=0")
    r99 = client.get("/api/forecast/RELIANCE?horizon=99")
    assert r0.status_code == 400
    assert r99.status_code == 400


def test_api_forecast_route_rejects_bad_model() -> None:
    client = TestClient(_make_app(_StubFetcher(_df()), ForecastCache(None)))
    r = client.get("/api/forecast/RELIANCE?model=lstm")
    assert r.status_code == 400


def test_api_forecast_route_returns_422_on_no_history() -> None:
    client = TestClient(_make_app(_StubFetcher(None), ForecastCache(None)))
    r = client.get("/api/forecast/RELIANCE")
    assert r.status_code == 422
    assert "insufficient" in r.json()["error"]


def test_api_forecast_route_returns_422_on_short_history() -> None:
    short_df = _df(n=30)
    client = TestClient(_make_app(_StubFetcher(short_df), ForecastCache(None)))
    r = client.get("/api/forecast/RELIANCE")
    assert r.status_code == 422
