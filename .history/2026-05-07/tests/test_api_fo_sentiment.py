"""Route tests for /api/fo-sentiment."""
from __future__ import annotations

from typing import Any, Dict

import pytest


class _StubFetcher:
    def __init__(self, nifty_chain: Dict = None, banknifty_chain: Dict = None) -> None:
        self.nifty = nifty_chain or {}
        self.banknifty = banknifty_chain or {}

    def get_option_chain(self, symbol: str) -> Dict:
        if symbol == "NIFTY":
            return self.nifty
        if symbol == "BANKNIFTY":
            return self.banknifty
        return {}

    def get_option_chain_from_kite(self, symbol: str, kite: Any) -> Any:
        return None


def test_fo_sentiment_unavailable_when_both_chains_empty(monkeypatch: Any) -> None:
    """When NSE returns empty for both indices and Kite is offline, surface unavailable."""
    server = pytest.importorskip("server")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    unavailable = {
        "unavailable": True,
        "calls": [],
        "puts": [],
        "pcr": 1.0,
        "max_pain": 0.0,
    }
    stub = _StubFetcher(nifty_chain=unavailable, banknifty_chain=unavailable)
    monkeypatch.setattr(server, "get_options_fetcher", lambda: stub, raising=False)
    monkeypatch.setattr(server.controller.kite, "is_connected", False, raising=False)

    client = TestClient(server.app)
    resp = client.get("/api/fo-sentiment")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("unavailable") is True
    assert "reason" in body
    assert body.get("gauge") is None
    assert body.get("sentiment") is None
    assert body.get("nifty_pcr") is None
    assert body.get("banknifty_pcr") is None
    assert body.get("avg_pcr") is None
    assert body.get("nifty_max_pain") is None
    assert body.get("banknifty_max_pain") is None
    assert body.get("nifty_call_oi") is None
    assert body.get("nifty_put_oi") is None


def test_fo_sentiment_mixed_availability_uses_available_chain_only(monkeypatch: Any) -> None:
    """When only one chain is available, compute sentiment from that chain only."""
    server = pytest.importorskip("server")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    available = {
        "calls": [{"strike": 25000, "oi": 100000}],
        "puts": [{"strike": 25000, "oi": 150000}],
        "pcr": 1.5,
        "max_pain": 25000.0,
        "total_call_oi": 100000,
        "total_put_oi": 150000,
    }
    unavailable = {"unavailable": True, "calls": [], "puts": [], "pcr": 1.0}
    stub = _StubFetcher(nifty_chain=available, banknifty_chain=unavailable)
    monkeypatch.setattr(server, "get_options_fetcher", lambda: stub, raising=False)
    monkeypatch.setattr(server.controller.kite, "is_connected", False, raising=False)

    client = TestClient(server.app)
    resp = client.get("/api/fo-sentiment")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("unavailable") is None
    # Only NIFTY available → avg_pcr should be 1.5 (Fear/30), not diluted with 1.0
    assert body["nifty_pcr"] == 1.5
    assert body["banknifty_pcr"] is None
    assert body["avg_pcr"] == 1.5
    assert body["gauge"] == 30
    assert body["sentiment"] == "Fear"
    assert body["nifty_max_pain"] == 25000.0
    assert body["banknifty_max_pain"] is None


def test_fo_sentiment_mixed_availability_reversed(monkeypatch: Any) -> None:
    """When only BANKNIFTY is available, compute sentiment from BANKNIFTY only."""
    server = pytest.importorskip("server")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    available = {
        "calls": [{"strike": 48000, "oi": 200000}],
        "puts": [{"strike": 48000, "oi": 100000}],
        "pcr": 0.5,
        "max_pain": 48000.0,
        "total_call_oi": 200000,
        "total_put_oi": 100000,
    }
    unavailable = {"unavailable": True, "calls": [], "puts": [], "pcr": 1.0}
    stub = _StubFetcher(nifty_chain=unavailable, banknifty_chain=available)
    monkeypatch.setattr(server, "get_options_fetcher", lambda: stub, raising=False)
    monkeypatch.setattr(server.controller.kite, "is_connected", False, raising=False)

    client = TestClient(server.app)
    resp = client.get("/api/fo-sentiment")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("unavailable") is None
    assert body["nifty_pcr"] is None
    assert body["banknifty_pcr"] == 0.5
    assert body["avg_pcr"] == 0.5
    assert body["gauge"] == 90  # Extreme Greed
    assert body["sentiment"] == "Extreme Greed"


def test_fo_sentiment_computes_gauge_when_data_present(monkeypatch: Any) -> None:
    """When chains have real data, compute PCR-based sentiment gauge."""
    server = pytest.importorskip("server")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    chain = {
        "calls": [{"strike": 25000, "oi": 100000}],
        "puts": [{"strike": 25000, "oi": 150000}],
        "pcr": 1.5,
        "max_pain": 25000.0,
        "total_call_oi": 100000,
        "total_put_oi": 150000,
    }
    stub = _StubFetcher(nifty_chain=chain, banknifty_chain=chain)
    monkeypatch.setattr(server, "get_options_fetcher", lambda: stub, raising=False)
    monkeypatch.setattr(server.controller.kite, "is_connected", False, raising=False)

    client = TestClient(server.app)
    resp = client.get("/api/fo-sentiment")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("unavailable") is None
    assert body["gauge"] == 30  # Fear (avg_pcr 1.5, strict > 1.5 is False)
    assert body["sentiment"] == "Fear"
    assert body["nifty_max_pain"] == 25000.0
