"""W3.3 verification: POST /api/options/strategy wire contract.

Stand-in pattern (matches test_api_signal_calibrated.py): build a tiny FastAPI
mirror of the live route logic and inject a deterministic option-chain stub.
Exercises wire shape, validation, and unavailable-chain graceful degradation
without booting the full controller.
"""
from __future__ import annotations

import math
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from marketmind.ml.options.builder import analyse
from marketmind.ml.options.strategies import ALL_STRATEGIES, build_default_legs


# ─── stub chain ───────────────────────────────────────────────────────────


def _chain(atm: float = 100.0, step: float = 5.0, vol: float = 20.0) -> dict:
    strikes = [atm + step * k for k in (-2, -1, 0, 1, 2)]
    calls = [
        {"strike": s, "ltp": max(atm - s, 0) + 2.0, "iv": vol, "oi": 1000}
        for s in strikes
    ]
    puts = [
        {"strike": s, "ltp": max(s - atm, 0) + 2.0, "iv": vol, "oi": 1000}
        for s in strikes
    ]
    return {
        "symbol": "TEST",
        "underlying": atm,
        "atm_strike": atm,
        "calls": calls,
        "puts": puts,
        "expiry_dates": ["30-Apr-2026"],
    }


class _StubFetcher:
    def __init__(self, chain: dict) -> None:
        self._chain = chain

    def get_option_chain(self, symbol: str) -> dict:
        return self._chain


def _sanitize(obj):
    """Mirror server.py _sanitize for NaN/Inf scrubbing."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─── route mirror ─────────────────────────────────────────────────────────


class _Leg(BaseModel):
    action: str
    kind: str
    strike: float
    premium: float = 0.0
    iv: float = 0.0
    qty: int = 1
    expiry_days: int = 30


class _Req(BaseModel):
    symbol: str
    strategy: str
    expiry_days: int = 30
    lots: int = 1
    lot_size: int = 1
    legs: Optional[list[_Leg]] = None
    underlying: Optional[float] = None
    back_expiry_days: Optional[int] = None


def _make_app(fetcher: _StubFetcher) -> FastAPI:
    app = FastAPI()

    @app.post("/api/options/strategy")
    def options_strategy(req: _Req):
        if req.strategy not in ALL_STRATEGIES:
            raise HTTPException(status_code=400, detail=f"unknown strategy: {req.strategy}")
        if req.strategy == "calendar_spread" and req.back_expiry_days is None:
            raise HTTPException(
                status_code=400,
                detail="calendar_spread requires back_expiry_days",
            )

        chain = fetcher.get_option_chain(req.symbol.upper())
        underlying = req.underlying or float(chain.get("underlying") or chain.get("atm_strike") or 0)

        if req.legs:
            legs = [leg.dict() for leg in req.legs]
        else:
            if chain.get("unavailable"):
                return JSONResponse(_sanitize({
                    "strategy": req.strategy,
                    "symbol": req.symbol.upper(),
                    "unavailable": True,
                    "reason": chain.get("reason", "options chain unavailable"),
                }))
            legs = build_default_legs(
                req.strategy, chain,
                expiry_days=req.expiry_days,
                lots=req.lots,
                lot_size=req.lot_size,
            )
            if req.strategy == "calendar_spread" and req.back_expiry_days is not None:
                legs[1]["expiry_days"] = int(req.back_expiry_days)

        if underlying <= 0:
            raise HTTPException(status_code=400, detail="could not determine underlying price")

        try:
            result = analyse(legs, underlying, strategy_name=req.strategy,
                              multi_expiry=(req.strategy == "calendar_spread"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        result["symbol"] = req.symbol.upper()
        result["iv_rank_hint"] = None
        return JSONResponse(_sanitize(result))

    return app


# ─── tests ────────────────────────────────────────────────────────────────


def test_api_options_strategy_returns_200_with_default_legs():
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={"symbol": "test", "strategy": "iron_condor"})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "TEST"
    assert body["strategy"] == "iron_condor"
    assert len(body["legs"]) == 4
    assert len(body["payoff"]) > 100
    assert "max_profit" in body and "max_loss" in body
    assert "net_greeks" in body
    assert body["pricing_model"] == "black_scholes_european"


def test_api_options_strategy_accepts_custom_legs():
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={
        "symbol": "test",
        "strategy": "straddle",
        "underlying": 100.0,
        "legs": [
            {"action": "BUY", "kind": "CE", "strike": 100, "premium": 3.0, "iv": 0.20, "qty": 1, "expiry_days": 30},
            {"action": "BUY", "kind": "PE", "strike": 100, "premium": 3.0, "iv": 0.20, "qty": 1, "expiry_days": 30},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    bes = sorted(body["break_evens"])
    assert bes[0] == 94 or abs(bes[0] - 94.0) < 1.0
    assert bes[-1] == 106 or abs(bes[-1] - 106.0) < 1.0


def test_api_options_strategy_rejects_unknown_strategy():
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={"symbol": "test", "strategy": "moonshot"})
    assert r.status_code == 400
    assert "unknown strategy" in r.json()["detail"]


def test_api_options_strategy_calendar_requires_back_expiry():
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={"symbol": "test", "strategy": "calendar_spread"})
    assert r.status_code == 400
    assert "back_expiry_days" in r.json()["detail"]


def test_api_options_strategy_calendar_with_back_expiry_succeeds():
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={
        "symbol": "test",
        "strategy": "calendar_spread",
        "expiry_days": 30,
        "back_expiry_days": 60,
    })
    assert r.status_code == 200
    body = r.json()
    assert any("near expiry" in n for n in body["notes"])
    assert body["legs"][1]["expiry_days"] == 60


def test_api_options_strategy_unavailable_chain_returns_graceful_payload():
    chain = {
        "symbol": "TEST", "underlying": 0, "atm_strike": 0,
        "calls": [], "puts": [], "expiry_dates": [],
        "unavailable": True, "reason": "markets closed",
    }
    app = _make_app(_StubFetcher(chain))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={"symbol": "test", "strategy": "iron_condor"})
    assert r.status_code == 200
    body = r.json()
    assert body["unavailable"] is True
    assert body["reason"] == "markets closed"
    assert "payoff" not in body  # short-circuited before analytics


def test_api_options_strategy_payload_is_json_safe():
    """No NaN/Inf leaks through — _sanitize must scrub them."""
    app = _make_app(_StubFetcher(_chain()))
    client = TestClient(app)
    r = client.post("/api/options/strategy", json={"symbol": "test", "strategy": "bull_call_spread"})
    assert r.status_code == 200
    # raw json text must not contain NaN/Infinity tokens — would break JS clients.
    raw = r.text
    assert "NaN" not in raw
    assert "Infinity" not in raw
