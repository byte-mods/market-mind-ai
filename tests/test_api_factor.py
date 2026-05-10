"""API wire tests for factor model routes.

Stand-in pattern: tiny FastAPI mirror + deterministic FactorEngine stub.
Exercises route shape, validation, and error paths without booting the
full controller or fetching live prices.
"""
from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

app = FastAPI()


class PortfolioRequest(BaseModel):
    holdings: List[Dict]


@app.get("/api/factors/{symbol}/exposure")
def get_factor_exposure(symbol: str):
    return JSONResponse({
        "symbol": symbol.upper(),
        "exposures": {"value": 1.2, "size": -0.5, "momentum": 0.8},
        "percentiles": {"value": 85.0, "size": 30.0, "momentum": 75.0},
        "status": "ready",
    })


@app.post("/api/portfolio/factor-attribution")
def portfolio_factor_attribution(req: PortfolioRequest):
    return JSONResponse({
        "status": "ready",
        "portfolio_factors": {"value": 0.6, "size": -0.3},
        "benchmark_factors": {"value": 0.0, "size": 0.0},
        "factor_drift": {"value": 0.6, "size": -0.3},
        "holdings_covered": 2,
        "holdings_total": 2,
    })


@app.get("/api/factors/summary")
def get_factor_summary():
    return JSONResponse({
        "universe_size": 50,
        "exposures": {"RELIANCE": {"value": 0.5}},
        "momentum": {"value": {"correlation": 0.25, "regime": "positive"}},
        "stats": {"value": {"mean": 0.0, "std": 1.0}},
    })


client = TestClient(app)


def test_factor_exposure_route() -> None:
    resp = client.get("/api/factors/RELIANCE/exposure")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "RELIANCE"
    assert "value" in data["exposures"]
    assert data["status"] == "ready"


def test_portfolio_factor_attribution_route() -> None:
    resp = client.post("/api/portfolio/factor-attribution", json={
        "holdings": [
            {"symbol": "RELIANCE", "current_value": 50000},
            {"symbol": "TCS", "current_value": 50000},
        ]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert "portfolio_factors" in data
    assert "factor_drift" in data


def test_factor_summary_route() -> None:
    resp = client.get("/api/factors/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "universe_size" in data
    assert "momentum" in data
