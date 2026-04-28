"""W5.3 T5 real-server route tests.

Boots the actual FastAPI app from `server.py` and monkeypatches the
module-level `controller` binding so we catch any drift between the
mirror's Pydantic schema / wiring logic and the real route. Mirrors the
W4.1 pattern in `tests/test_api_rebalance_tax_optimal_realserver.py`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest


server = pytest.importorskip("server")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


# ─── Stub controller mirroring AppController compliance surface ───────────


class _StubController:
    def __init__(self, is_connected: bool = True) -> None:
        self.kite = type("kite", (), {"is_connected": is_connected})()
        self.kite_is_authenticated = is_connected
        self.last_pretrade_call: Optional[Dict] = None
        self.last_audit_call: Optional[Dict] = None
        self.last_insider_call: Optional[Dict] = None
        self.last_designated_call: Optional[Dict] = None

    def compliance_pretrade_check(self, symbol, transaction_type, quantity, price):
        self.last_pretrade_call = {
            "symbol": symbol, "transaction_type": transaction_type,
            "quantity": quantity, "price": price,
        }
        return {
            "authenticated": True, "error": None,
            "decision": "ALLOW", "reasons": [],
            "audit_id": "X:2026-04-28T10:00:00+00:00:abc",
            "ts": "2026-04-28T10:00:00+00:00",
            "insider_window_open": None, "insider_window_reason": None,
        }

    def compliance_get_audit_log(self, symbol=None, since=None, limit=100):
        self.last_audit_call = {"symbol": symbol, "since": since, "limit": limit}
        return {"authenticated": True, "error": None, "entries": []}

    def compliance_get_insider_window(self, symbol):
        self.last_insider_call = {"symbol": symbol}
        return {
            "authenticated": True, "error": None,
            "symbol": symbol.upper(), "is_open": True,
            "closed_until": None, "last_results_date": None,
            "reason": "no announcements",
        }

    def compliance_set_designated_symbols(self, symbols):
        self.last_designated_call = {"symbols": list(symbols)}
        return {
            "authenticated": True, "error": None,
            "symbols": [s.upper() for s in symbols],
        }


# ─── Route registration ──────────────────────────────────────────────────


def test_real_server_compliance_routes_registered_on_live_app() -> None:
    """All four W5.3 routes are present on the actual FastAPI app."""
    routes = {
        (r.path, tuple(sorted(r.methods)))
        for r in server.app.routes
        if hasattr(r, "path") and hasattr(r, "methods")
    }
    expected = {
        ("/api/compliance/pretrade-check", ("POST",)),
        ("/api/compliance/audit-log", ("GET",)),
        ("/api/compliance/insider-window/{symbol}", ("GET",)),
        ("/api/compliance/designated-symbols", ("POST",)),
    }
    missing = expected - routes
    assert not missing, f"missing routes: {missing}"


# ─── Wiring through the real route → controller ──────────────────────────


def test_real_server_pretrade_check_wires_to_controller(monkeypatch) -> None:
    stub = _StubController()
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)
    r = client.post("/api/compliance/pretrade-check", json={
        "symbol": "RELIANCE", "transaction_type": "BUY",
        "quantity": 1, "price": 100.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["decision"] == "ALLOW"
    # Module-level controller really got swapped — call args land on the stub.
    assert stub.last_pretrade_call["symbol"] == "RELIANCE"
    assert stub.last_pretrade_call["transaction_type"] == "BUY"


def test_real_server_audit_log_wires_to_controller(monkeypatch) -> None:
    stub = _StubController()
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)
    r = client.get("/api/compliance/audit-log?symbol=TCS&limit=25")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert stub.last_audit_call == {
        "symbol": "TCS", "since": None, "limit": 25,
    }


def test_real_server_insider_window_wires_to_controller(monkeypatch) -> None:
    stub = _StubController()
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)
    r = client.get("/api/compliance/insider-window/RELIANCE")
    assert r.status_code == 200
    body = r.json()
    assert body["is_open"] is True
    assert body["symbol"] == "RELIANCE"
    assert stub.last_insider_call == {"symbol": "RELIANCE"}


def test_real_server_designated_symbols_wires_to_controller(monkeypatch) -> None:
    stub = _StubController()
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)
    r = client.post("/api/compliance/designated-symbols",
                    json={"symbols": ["reliance", "tcs"]})
    assert r.status_code == 200
    body = r.json()
    assert body["symbols"] == ["RELIANCE", "TCS"]
    assert stub.last_designated_call == {"symbols": ["reliance", "tcs"]}


def test_real_server_pretrade_check_validation_422(monkeypatch) -> None:
    """Missing required body field → 422, controller not invoked."""
    stub = _StubController()
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)
    r = client.post("/api/compliance/pretrade-check",
                    json={"symbol": "RELIANCE"})  # missing fields
    assert r.status_code == 422
    assert stub.last_pretrade_call is None
