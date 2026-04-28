"""W5.3 T5 verification: SEBI compliance route mirror tests.

Builds a tiny FastAPI mirror of the four `/api/compliance/*` routes and
injects a stub controller. Exercises wire shape and the
`authenticated`/`error` envelope without booting AppController.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─── stub controller ──────────────────────────────────────────────────────


class _StubController:
    """Records call args, returns configured envelopes per method."""

    def __init__(
        self,
        pretrade_response: Optional[Dict] = None,
        audit_response: Optional[Dict] = None,
        insider_response: Optional[Dict] = None,
        designated_response: Optional[Dict] = None,
        is_connected: bool = True,
    ) -> None:
        self._pretrade = pretrade_response or _ok_pretrade_envelope()
        self._audit = audit_response or _ok_audit_envelope()
        self._insider = insider_response or _ok_insider_envelope()
        self._designated = designated_response or _ok_designated_envelope()
        self.last_pretrade_call: Optional[Dict] = None
        self.last_audit_call: Optional[Dict] = None
        self.last_insider_call: Optional[Dict] = None
        self.last_designated_call: Optional[Dict] = None
        self.kite_is_authenticated = is_connected

    def compliance_pretrade_check(self, symbol, transaction_type, quantity, price):
        self.last_pretrade_call = {
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "price": price,
        }
        return dict(self._pretrade)

    def compliance_get_audit_log(self, symbol=None, since=None, limit=100):
        self.last_audit_call = {"symbol": symbol, "since": since, "limit": limit}
        return dict(self._audit)

    def compliance_get_insider_window(self, symbol):
        self.last_insider_call = {"symbol": symbol}
        return dict(self._insider)

    def compliance_set_designated_symbols(self, symbols):
        self.last_designated_call = {"symbols": list(symbols)}
        return dict(self._designated)


def _ok_pretrade_envelope() -> Dict:
    return {
        "authenticated": True, "error": None,
        "decision": "ALLOW", "reasons": [],
        "audit_id": "X:2026-04-28T10:00:00+00:00:abc123",
        "ts": "2026-04-28T10:00:00+00:00",
        "insider_window_open": None, "insider_window_reason": None,
    }


def _ok_audit_envelope() -> Dict:
    return {
        "authenticated": True, "error": None,
        "entries": [{"_id": "X", "ts": "2026-04-28T10:00:00+00:00",
                     "symbol": "RELIANCE", "decision": "ALLOW",
                     "reasons": [], "source": "pretrade",
                     "transaction_type": "BUY", "quantity": 1, "price": 100.0}],
    }


def _ok_insider_envelope() -> Dict:
    return {
        "authenticated": True, "error": None,
        "symbol": "RELIANCE", "is_open": True,
        "closed_until": None, "last_results_date": None,
        "reason": "no announcements available",
    }


def _ok_designated_envelope() -> Dict:
    return {"authenticated": True, "error": None, "symbols": ["RELIANCE"]}


# ─── mirror app ───────────────────────────────────────────────────────────


class PretradeRequest(BaseModel):
    symbol: str
    transaction_type: str
    quantity: float
    price: float


class DesignatedSymbolsRequest(BaseModel):
    symbols: List[str]


def _build_app(stub: _StubController) -> FastAPI:
    app = FastAPI()

    @app.post("/api/compliance/pretrade-check")
    def pretrade(req: PretradeRequest):
        return JSONResponse(_sanitize(
            stub.compliance_pretrade_check(
                req.symbol, req.transaction_type, req.quantity, req.price
            )
        ))

    @app.get("/api/compliance/audit-log")
    def audit_log(symbol: Optional[str] = None, since: Optional[str] = None, limit: int = 100):
        return JSONResponse(_sanitize(
            stub.compliance_get_audit_log(symbol, since, limit)
        ))

    @app.get("/api/compliance/insider-window/{symbol}")
    def insider_window(symbol: str):
        return JSONResponse(_sanitize(stub.compliance_get_insider_window(symbol)))

    @app.post("/api/compliance/designated-symbols")
    def designated_symbols(req: DesignatedSymbolsRequest):
        return JSONResponse(_sanitize(
            stub.compliance_set_designated_symbols(req.symbols)
        ))

    return app


# ─── pretrade-check ───────────────────────────────────────────────────────


def test_compliance_pretrade_check_returns_envelope() -> None:
    stub = _StubController()
    client = TestClient(_build_app(stub))
    r = client.post("/api/compliance/pretrade-check", json={
        "symbol": "RELIANCE", "transaction_type": "BUY",
        "quantity": 1, "price": 100.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["decision"] == "ALLOW"
    assert stub.last_pretrade_call["symbol"] == "RELIANCE"


def test_compliance_pretrade_check_block_envelope_propagates() -> None:
    stub = _StubController(pretrade_response={
        "authenticated": True, "error": None,
        "decision": "BLOCK", "reasons": ["insider window closed: ..."],
        "audit_id": "X:1:abc", "ts": "2026-04-28T10:00:00+00:00",
        "insider_window_open": False, "insider_window_reason": "closed",
    })
    client = TestClient(_build_app(stub))
    r = client.post("/api/compliance/pretrade-check", json={
        "symbol": "RELIANCE", "transaction_type": "BUY",
        "quantity": 1, "price": 100.0,
    })
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert "insider window closed" in body["reasons"][0]


def test_compliance_pretrade_check_missing_field_422() -> None:
    """Missing required body field → FastAPI 422 (Pydantic validation)."""
    client = TestClient(_build_app(_StubController()))
    r = client.post("/api/compliance/pretrade-check", json={
        "symbol": "RELIANCE", "transaction_type": "BUY",
        # missing quantity, price
    })
    assert r.status_code == 422


def test_compliance_pretrade_check_kite_disconnected_envelope() -> None:
    stub = _StubController(
        pretrade_response={
            "authenticated": False, "error": None,
            "decision": "WARN", "reasons": ["concentration"],
            "audit_id": None, "ts": "2026-04-28T10:00:00+00:00",
            "insider_window_open": None, "insider_window_reason": None,
        },
        is_connected=False,
    )
    client = TestClient(_build_app(stub))
    r = client.post("/api/compliance/pretrade-check", json={
        "symbol": "RELIANCE", "transaction_type": "BUY",
        "quantity": 10, "price": 2500.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is False
    assert body["decision"] == "WARN"


# ─── audit-log ────────────────────────────────────────────────────────────


def test_compliance_audit_log_envelope() -> None:
    stub = _StubController()
    client = TestClient(_build_app(stub))
    r = client.get("/api/compliance/audit-log")
    assert r.status_code == 200
    body = r.json()
    assert "entries" in body
    assert body["entries"][0]["symbol"] == "RELIANCE"


def test_compliance_audit_log_query_args_passed_through() -> None:
    stub = _StubController()
    client = TestClient(_build_app(stub))
    r = client.get(
        "/api/compliance/audit-log?symbol=tcs&since=2026-04-27T00:00:00&limit=50"
    )
    assert r.status_code == 200
    assert stub.last_audit_call == {
        "symbol": "tcs", "since": "2026-04-27T00:00:00", "limit": 50,
    }


def test_compliance_audit_log_bad_since_returns_envelope_error() -> None:
    """Mirror controller error envelope (not HTTPException) on bad since."""
    stub = _StubController(audit_response={
        "authenticated": True,
        "error": "since must be ISO datetime, got 'not-a-date'",
        "entries": [],
    })
    client = TestClient(_build_app(stub))
    r = client.get("/api/compliance/audit-log?since=not-a-date")
    assert r.status_code == 200
    body = r.json()
    assert "must be ISO" in body["error"]
    assert body["entries"] == []


# ─── insider-window ───────────────────────────────────────────────────────


def test_compliance_insider_window_envelope() -> None:
    stub = _StubController()
    client = TestClient(_build_app(stub))
    r = client.get("/api/compliance/insider-window/RELIANCE")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "RELIANCE"
    assert body["is_open"] is True
    assert stub.last_insider_call == {"symbol": "RELIANCE"}


def test_compliance_insider_window_path_param_passed_verbatim() -> None:
    """Controller does the upper/strip; route passes through."""
    stub = _StubController()
    client = TestClient(_build_app(stub))
    client.get("/api/compliance/insider-window/reliance")
    assert stub.last_insider_call == {"symbol": "reliance"}


# ─── designated-symbols ───────────────────────────────────────────────────


def test_compliance_designated_symbols_round_trip() -> None:
    stub = _StubController()
    client = TestClient(_build_app(stub))
    r = client.post(
        "/api/compliance/designated-symbols",
        json={"symbols": ["reliance", "tcs"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbols"] == ["RELIANCE"]
    assert stub.last_designated_call == {"symbols": ["reliance", "tcs"]}


def test_compliance_designated_symbols_empty_list_ok() -> None:
    stub = _StubController(designated_response={
        "authenticated": True, "error": None, "symbols": []
    })
    client = TestClient(_build_app(stub))
    r = client.post("/api/compliance/designated-symbols", json={"symbols": []})
    assert r.status_code == 200
    assert r.json()["symbols"] == []


def test_compliance_designated_symbols_no_mongo_envelope() -> None:
    stub = _StubController(designated_response={
        "authenticated": True,
        "error": "mongo not available; designated list not persisted",
        "symbols": ["RELIANCE"],
    })
    client = TestClient(_build_app(stub))
    r = client.post("/api/compliance/designated-symbols",
                    json={"symbols": ["RELIANCE"]})
    body = r.json()
    assert "mongo not available" in body["error"]
    assert body["symbols"] == ["RELIANCE"]


def test_compliance_designated_symbols_missing_field_422() -> None:
    client = TestClient(_build_app(_StubController()))
    r = client.post("/api/compliance/designated-symbols", json={})
    assert r.status_code == 422
