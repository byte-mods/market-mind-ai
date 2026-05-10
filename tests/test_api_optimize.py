"""API wire tests for portfolio optimizer routes.

Stand-in pattern: tiny FastAPI mirror + deterministic optimizer stub.
Exercises request validation, response shape, and BL view passthrough
without booting the full server or fetching live prices.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ─── Stub optimizer ─────────────────────────────────────────────────────────


class _StubOptimizer:
    """Deterministic optimizer that returns predictable shapes."""

    def optimize(self, symbols, objective='max_sharpe', days=252,
                 views=None, market_weights=None):
        n = len(symbols)
        w = np.ones(n) / n
        allocation = {s: round(float(wi) * 100, 2) for s, wi in zip(symbols, w)}
        result = {
            'objective': objective,
            'symbols': symbols,
            'weights': dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True)),
            'expected_annual_return_pct': 12.0,
            'annual_volatility_pct': 15.0,
            'sharpe_ratio': 0.8,
            'interpretation': f'{objective}: Expected return 12.0%/yr, Volatility 15.0%/yr, Sharpe 0.80',
        }
        if views:
            result['_views_received'] = len(views)
        if market_weights:
            result['_market_weights_received'] = True
        return result

    def compare_strategies(self, symbols, days=252):
        return []

    def efficient_frontier(self, symbols, n_points=20, days=252):
        return {
            'frontier': [
                {'return_pct': 10.0 + i, 'vol_pct': 12.0 + i * 0.5, 'sharpe': 0.7 + i * 0.01}
                for i in range(n_points)
            ],
            'max_sharpe_point': {'return_pct': 15.0, 'vol_pct': 14.0, 'sharpe': 1.0},
            'min_variance_point': {'return_pct': 10.0, 'vol_pct': 12.0, 'sharpe': 0.7},
        }


_STUB_OPT = _StubOptimizer()


# ─── Mirror app ─────────────────────────────────────────────────────────────

app = FastAPI()


class OptimizeRequest(BaseModel):
    symbols: List[str]
    objective: str = 'max_sharpe'
    days: int = 252
    views: Optional[List[Dict]] = None
    market_weights: Optional[Dict[str, float]] = None


class BlackLittermanRequest(BaseModel):
    symbols: List[str]
    days: int = 252
    views: List[Dict] = []
    market_weights: Optional[Dict[str, float]] = None


@app.post("/api/optimize")
def optimize_portfolio(req: OptimizeRequest):
    bl_kwargs = {}
    if req.objective == 'black_litterman':
        bl_kwargs = {
            'views': req.views or [],
            'market_weights': req.market_weights,
        }
    result = _STUB_OPT.optimize(
        [s.upper() for s in req.symbols], req.objective, req.days, **bl_kwargs
    )
    compare = _STUB_OPT.compare_strategies([s.upper() for s in req.symbols], req.days)
    result['strategy_comparison'] = compare
    return JSONResponse(result)


@app.post("/api/optimize/black-litterman")
def optimize_black_litterman(req: BlackLittermanRequest):
    result = _STUB_OPT.optimize(
        [s.upper() for s in req.symbols],
        objective='black_litterman',
        days=req.days,
        views=req.views,
        market_weights=req.market_weights,
    )
    return JSONResponse(result)


@app.post("/api/optimize/frontier")
def efficient_frontier(req: OptimizeRequest):
    result = _STUB_OPT.efficient_frontier([s.upper() for s in req.symbols], 20, req.days)
    return JSONResponse(result)


client = TestClient(app)


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_optimize_default_max_sharpe() -> None:
    resp = client.post("/api/optimize", json={"symbols": ["RELIANCE", "TCS"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "max_sharpe"
    assert "weights" in data
    assert sum(data["weights"].values()) == pytest.approx(100.0, abs=0.1)


def test_optimize_hrp_objective() -> None:
    resp = client.post("/api/optimize", json={"symbols": ["RELIANCE", "TCS"], "objective": "hrp"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "hrp"


def test_optimize_black_litterman_via_generic() -> None:
    views = [{"assets": ["TCS"], "type": "absolute", "magnitude": 0.15, "confidence": 0.7}]
    resp = client.post("/api/optimize", json={
        "symbols": ["RELIANCE", "TCS"],
        "objective": "black_litterman",
        "views": views,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "black_litterman"
    assert data.get("_views_received") == 1


def test_optimize_black_litterman_dedicated_endpoint() -> None:
    views = [
        {"assets": ["TCS"], "type": "absolute", "magnitude": 0.15, "confidence": 0.7},
        {"assets": ["TCS", "RELIANCE"], "type": "relative", "magnitude": 0.03, "confidence": 0.6},
    ]
    resp = client.post("/api/optimize/black-litterman", json={
        "symbols": ["RELIANCE", "TCS", "INFY"],
        "views": views,
        "market_weights": {"RELIANCE": 0.4, "TCS": 0.35, "INFY": 0.25},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "black_litterman"
    assert data.get("_views_received") == 2
    assert data.get("_market_weights_received") is True


def test_optimize_frontier() -> None:
    resp = client.post("/api/optimize/frontier", json={"symbols": ["RELIANCE", "TCS"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "frontier" in data
    assert len(data["frontier"]) == 20
    assert "max_sharpe_point" in data
    assert "min_variance_point" in data


def test_optimize_empty_symbols() -> None:
    resp = client.post("/api/optimize", json={"symbols": []})
    # Stub returns empty weights; real optimizer would return error
    assert resp.status_code == 200


def test_black_litterman_no_views() -> None:
    resp = client.post("/api/optimize/black-litterman", json={
        "symbols": ["RELIANCE", "TCS"],
        "views": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "black_litterman"


def test_optimize_black_litterman_omitted_views() -> None:
    """Generic optimize with BL objective and no views key defaults to []."""
    resp = client.post("/api/optimize", json={
        "symbols": ["RELIANCE", "TCS"],
        "objective": "black_litterman",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["objective"] == "black_litterman"
