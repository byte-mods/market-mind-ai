"""Route-mirror + real-server tests for /api/options/symbols."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


class _StubFetcher:
    def __init__(self, symbols: List[str] = None) -> None:
        self._symbols = symbols or ["NIFTY", "RELIANCE"]
        self.last_call = ""

    def get_fo_symbols(self) -> List[str]:
        self.last_call = "get_fo_symbols"
        return self._symbols


def _make_app(fetcher: _StubFetcher) -> FastAPI:
    app = FastAPI()

    @app.get("/api/options/symbols")
    async def get_options_symbols():
        symbols = fetcher.get_fo_symbols()
        return JSONResponse({"symbols": symbols})

    return app


def test_options_symbols_returns_list() -> None:
    stub = _StubFetcher(["NIFTY", "BANKNIFTY", "RELIANCE"])
    client = TestClient(_make_app(stub))
    resp = client.get("/api/options/symbols")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == ["NIFTY", "BANKNIFTY", "RELIANCE"]
    assert stub.last_call == "get_fo_symbols"


def test_options_symbols_real_server(monkeypatch: Any) -> None:
    """Schema-drift lock: real server route dispatches to stub fetcher."""
    import pytest
    server = pytest.importorskip("server")
    TestClientRS = pytest.importorskip("fastapi.testclient").TestClient

    stub = _StubFetcher(["FINNIFTY", "TCS"])
    monkeypatch.setattr(
        server, "get_options_fetcher", lambda: stub, raising=False
    )
    client = TestClientRS(server.app)
    resp = client.get("/api/options/symbols")
    assert resp.status_code == 200
    body = resp.json()
    assert "symbols" in body
    assert body["symbols"] == ["FINNIFTY", "TCS"]
