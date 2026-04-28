"""W4.1 deferred MINOR: real-server route test.

Mirror tests in `test_api_rebalance_tax_optimal.py` exercise a hand-rolled
copy of the route logic. This file boots the *actual* FastAPI app from
`server.py` and monkeypatches the module-level `controller` binding so we
catch any drift between the mirror's Pydantic schema / validation logic
and the real route's wiring.

Anything import-time (Kite init, Mongo, RL engine) that fails in a
sandbox skips the file with a clear message rather than failing red.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pytest


# Skip the whole module if the real server can't be imported in this env
# (e.g. CI without local.json / kiteconnect / RL deps installed).
server = pytest.importorskip("server")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


# ─── Stub controller (mirrors AppController.recommend_tax_optimal_rebalance) ──


class _StubController:
    """Records call args, returns a configured envelope. Mirrors the keyword
    signature of `AppController.recommend_tax_optimal_rebalance` exactly."""

    def __init__(self, response: Dict, is_connected: bool = True) -> None:
        self._response = response
        self.last_call: Optional[Dict] = None
        # Server route reads `controller.kite.is_connected` for the 400 path.
        self.kite = type("kite", (), {"is_connected": is_connected})()

    def recommend_tax_optimal_rebalance(
        self,
        target_weights,
        ltcg_used_inr=0.0,
        new_symbol_prices=None,
        harvest_losses=False,
        harvest_min_loss_inr=1_000.0,
        harvest_min_loss_pct=5.0,
        as_of=None,
        lots_override=None,
    ) -> Dict:
        self.last_call = {
            "target_weights": target_weights,
            "ltcg_used_inr": ltcg_used_inr,
            "new_symbol_prices": new_symbol_prices,
            "harvest_losses": harvest_losses,
            "harvest_min_loss_inr": harvest_min_loss_inr,
            "harvest_min_loss_pct": harvest_min_loss_pct,
            "as_of": as_of,
            "lots_override": lots_override,
        }
        return dict(self._response)


def _ok_envelope() -> Dict:
    return {
        "trades": [],
        "realized_gains": [],
        "tax_summary": {},
        "naive_tax_summary": {},
        "savings_inr": 0.0,
        "savings_pct": 0.0,
        "tracking_error_pct": 0.0,
        "harvest_candidates": [],
        "warnings": [],
        "authenticated": True,
        "error": None,
    }


# ─── Tests against the real `server.app` ──────────────────────────────────────


def test_real_server_rebalance_route_registered_on_live_app() -> None:
    """The W4.1 POST route must be present on the actual FastAPI app
    (catches a server.py import-order bug or accidental route deletion)."""
    paths = {(r.path, tuple(sorted(r.methods))) for r in server.app.routes
             if hasattr(r, "path") and hasattr(r, "methods")}
    assert ("/api/portfolio/rebalance/tax-optimal", ("POST",)) in paths, (
        "POST /api/portfolio/rebalance/tax-optimal not registered."
    )


def test_real_server_rebalance_tax_optimal_uses_module_controller(
    monkeypatch,
) -> None:
    """A request to the real route must dispatch into whatever object is
    bound to `server.controller` — proves the route does not capture the
    original controller reference at import time."""
    stub = _StubController(_ok_envelope())
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Envelope keys we depend on.
    for k in ("trades", "tax_summary", "naive_tax_summary", "savings_inr",
              "savings_pct", "tracking_error_pct", "warnings",
              "harvest_candidates", "realized_gains", "authenticated", "error"):
        assert k in body, f"missing key: {k}"
    assert stub.last_call is not None
    assert stub.last_call["target_weights"] == {"RELIANCE": 1.0}


def test_real_server_rebalance_tax_optimal_invalid_as_of_returns_400_envelope(
    monkeypatch,
) -> None:
    """Malformed `as_of` → real route returns 400 + flat envelope (NOT a
    Pydantic 422). Controller must not be called."""
    stub = _StubController(_ok_envelope())
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}, "as_of": "31-12-2025"},
    )
    assert resp.status_code == 400
    body = resp.json()
    # Same flat envelope shape as the 200 path.
    assert "YYYY-MM-DD" in body["error"]
    assert body["trades"] == []
    assert "authenticated" in body
    # Short-circuited before any controller method.
    assert stub.last_call is None


def test_real_server_rebalance_tax_optimal_pydantic_422_on_missing_weights(
    monkeypatch,
) -> None:
    """Missing required `target_weights` → Pydantic 422 from FastAPI before
    the route function ever runs. Controller untouched."""
    stub = _StubController(_ok_envelope())
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"ltcg_used_inr": 1000.0},
    )
    assert resp.status_code == 422
    # Pydantic's default body shape — explicitly NOT the flat envelope.
    body = resp.json()
    assert "detail" in body
    assert stub.last_call is None


def test_real_server_rebalance_tax_optimal_lots_override_round_trips(
    monkeypatch,
) -> None:
    """`lots_override` body field is parsed by Pydantic and forwarded into
    the controller as a plain dict-of-list-of-dict (controller does the
    date parsing). Catches schema drift between the mirror tests and the
    real `server.LotPayload` / `server.TaxRebalanceRequest`."""
    stub = _StubController(_ok_envelope())
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={
            "target_weights": {"RELIANCE": 1.0},
            "lots_override": {
                "RELIANCE": [
                    {"quantity": 10.0, "cost_basis": 2400.0,
                     "acquisition_date": "2024-06-01"},
                ],
            },
        },
    )
    assert resp.status_code == 200
    forwarded = stub.last_call["lots_override"]
    assert forwarded is not None
    assert "RELIANCE" in forwarded
    assert forwarded["RELIANCE"][0]["quantity"] == 10.0
    assert forwarded["RELIANCE"][0]["cost_basis"] == 2400.0
    assert forwarded["RELIANCE"][0]["acquisition_date"] == "2024-06-01"


def test_real_server_rebalance_tax_optimal_as_of_parsed_to_date_instance(
    monkeypatch,
) -> None:
    """A valid ISO `as_of` is parsed to a `datetime.date` instance before
    the controller call — locks the route's parsing contract."""
    stub = _StubController(_ok_envelope())
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}, "as_of": "2026-04-28"},
    )
    assert resp.status_code == 200
    assert stub.last_call["as_of"] == date(2026, 4, 28)


def test_real_server_rebalance_tax_optimal_disconnected_envelope_round_trips(
    monkeypatch,
) -> None:
    """Controller returns `authenticated=False` → real route echoes the
    envelope through with status 200 (NOT a 401/503). This is the
    portfolio-cluster contract."""
    stub = _StubController(
        {**_ok_envelope(),
         "authenticated": False,
         "error": "Kite not connected. Click Login to authenticate."},
        is_connected=False,
    )
    monkeypatch.setattr(server, "controller", stub)
    client = TestClient(server.app)

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is False
    assert "Login" in body["error"]
