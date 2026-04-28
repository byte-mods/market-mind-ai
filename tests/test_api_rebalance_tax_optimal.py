"""W4.1 verification: POST /api/portfolio/rebalance/tax-optimal wire contract.

Stand-in pattern (matches `test_api_options_strategy.py`): build a tiny FastAPI
mirror of the live route logic and inject a stub controller. Exercises wire
shape, validation, and the `authenticated`/`error` envelope without booting
the full AppController stack.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ─── helpers ──────────────────────────────────────────────────────────────


def _sanitize(obj):
    """Mirror server.py _sanitize for NaN/Inf scrubbing."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ─── stub controller ──────────────────────────────────────────────────────


class _StubController:
    """Records the call args, returns a configured envelope."""

    def __init__(self, response: Dict) -> None:
        self._response = response
        self.last_call: Optional[Dict] = None
        self.kite = type("kite", (), {"is_connected": True})()

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
    ):
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


# ─── route mirror ─────────────────────────────────────────────────────────


class _LotPayload(BaseModel):
    quantity: float
    cost_basis: float
    acquisition_date: Optional[str] = None


class _TaxRebalanceRequest(BaseModel):
    target_weights: Dict[str, float]
    ltcg_used_inr: float = 0.0
    new_symbol_prices: Optional[Dict[str, float]] = None
    harvest_losses: bool = False
    harvest_min_loss_inr: float = 1_000.0
    harvest_min_loss_pct: float = 5.0
    as_of: Optional[str] = None
    lots_override: Optional[Dict[str, List[_LotPayload]]] = None


def _make_app(controller: _StubController) -> FastAPI:
    app = FastAPI()

    @app.post("/api/portfolio/rebalance/tax-optimal")
    def rebalance_tax_optimal(req: _TaxRebalanceRequest):
        as_of_date: Optional[date] = None
        if req.as_of:
            try:
                as_of_date = datetime.strptime(req.as_of, "%Y-%m-%d").date()
            except ValueError:
                return JSONResponse(
                    _sanitize({
                        'authenticated': controller.kite.is_connected,
                        'error': f"as_of must be YYYY-MM-DD, got {req.as_of!r}",
                        'trades': [], 'realized_gains': [],
                        'tax_summary': {}, 'naive_tax_summary': {},
                        'savings_inr': 0.0, 'savings_pct': 0.0,
                        'tracking_error_pct': 0.0,
                        'harvest_candidates': [], 'warnings': [],
                    }),
                    status_code=400,
                )
        lots_override_raw: Optional[Dict[str, List[Dict]]] = None
        if req.lots_override is not None:
            lots_override_raw = {
                sym: [lot.model_dump() for lot in lots]
                for sym, lots in req.lots_override.items()
            }
        result = controller.recommend_tax_optimal_rebalance(
            req.target_weights,
            req.ltcg_used_inr,
            req.new_symbol_prices,
            req.harvest_losses,
            req.harvest_min_loss_inr,
            req.harvest_min_loss_pct,
            as_of_date,
            lots_override_raw,
        )
        return JSONResponse(_sanitize(result))

    return app


def _ok_envelope(**overrides) -> Dict:
    base = {
        'trades': [{'symbol': 'RELIANCE', 'action': 'SELL', 'quantity': 2,
                    'est_price': 2700.0, 'notional_inr': 5400.0, 'lots_sold': []}],
        'realized_gains': [],
        'tax_summary': {'stcg_realized_inr': 400.0, 'ltcg_realized_inr': 0.0,
                        'ltcg_taxable_inr': 0.0, 'stcg_tax_inr': 60.0,
                        'ltcg_tax_inr': 0.0, 'total_tax_inr': 60.0,
                        'ltcg_exemption_consumed_inr': 0.0},
        'naive_tax_summary': {'stcg_realized_inr': 400.0, 'ltcg_realized_inr': 0.0,
                              'ltcg_taxable_inr': 0.0, 'stcg_tax_inr': 60.0,
                              'ltcg_tax_inr': 0.0, 'total_tax_inr': 60.0,
                              'ltcg_exemption_consumed_inr': 0.0},
        'savings_inr': 0.0,
        'savings_pct': 0.0,
        'tracking_error_pct': 0.5,
        'harvest_candidates': [],
        'warnings': ['RELIANCE: no lot ledger supplied — taxed as STCG worst-case'],
        'authenticated': True,
        'error': None,
    }
    base.update(overrides)
    return base


# ─── tests ────────────────────────────────────────────────────────────────


def test_api_rebalance_tax_optimal_happy_path() -> None:
    """200 + full envelope + all canonical keys present."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 0.6, "INFY": 0.4}},
    )
    assert resp.status_code == 200
    body = resp.json()
    for k in ("trades", "tax_summary", "naive_tax_summary", "savings_inr",
              "savings_pct", "tracking_error_pct", "warnings",
              "harvest_candidates", "realized_gains", "authenticated", "error"):
        assert k in body, f"missing key: {k}"
    assert body["authenticated"] is True
    assert body["error"] is None
    # Defaults round-tripped to controller call.
    assert ctrl.last_call["ltcg_used_inr"] == 0.0
    assert ctrl.last_call["harvest_losses"] is False
    assert ctrl.last_call["as_of"] is None  # untouched -> defaults to today() inside controller


def test_api_rebalance_tax_optimal_validates_weight_sum() -> None:
    """Weight-sum drift must surface in `error` (envelope, not 500)."""
    ctrl = _StubController(_ok_envelope(
        error="target_weights must sum to ~1.0 (±0.01), got 0.8000",
        trades=[], warnings=[],
    ))
    client = TestClient(_make_app(ctrl))

    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 0.4, "INFY": 0.4}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is True
    assert body["error"] is not None
    assert "sum" in body["error"].lower() or "1.0" in body["error"]


def test_api_rebalance_tax_optimal_authenticated_flag_round_trips() -> None:
    """When the controller signals disconnected Kite, the route must echo it through."""
    ctrl = _StubController(_ok_envelope(
        authenticated=False,
        error="Kite not connected. Click Login to authenticate.",
        trades=[],
    ))
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is False
    assert "Login" in body["error"]


def test_api_rebalance_tax_optimal_warnings_surfaced_when_no_lot_ledger() -> None:
    """The lot-ledger fallback warning must reach the client untouched."""
    expected_warn = "RELIANCE: no lot ledger supplied — taxed as STCG worst-case"
    ctrl = _StubController(_ok_envelope(warnings=[expected_warn]))
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}},
    )
    assert resp.status_code == 200
    assert expected_warn in resp.json()["warnings"]


def test_api_rebalance_tax_optimal_harvest_candidates_returned_when_flag_set() -> None:
    """`harvest_losses=True` propagates to the controller and the response carries the list."""
    candidate = {
        "symbol": "INFY", "quantity": 20.0, "cost_basis": 1500.0,
        "current_price": 1300.0, "est_loss_inr": 4000.0,
        "holding_period_days": 100, "gain_type": "STCG",
        "advisory": "Avoid re-purchase within 30 days",
    }
    ctrl = _StubController(_ok_envelope(harvest_candidates=[candidate]))
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={
            "target_weights": {"RELIANCE": 0.5, "INFY": 0.5},
            "harvest_losses": True,
            "harvest_min_loss_inr": 500.0,
            "harvest_min_loss_pct": 3.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["harvest_candidates"] == [candidate]
    # And the route forwarded the flags.
    assert ctrl.last_call["harvest_losses"] is True
    assert ctrl.last_call["harvest_min_loss_inr"] == 500.0
    assert ctrl.last_call["harvest_min_loss_pct"] == 3.0


def test_api_rebalance_tax_optimal_invalid_as_of_returns_400() -> None:
    """Malformed `as_of` is rejected with 400; controller is NOT called."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}, "as_of": "not-a-date"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "YYYY-MM-DD" in body["error"]
    assert ctrl.last_call is None  # short-circuited before controller


def test_api_rebalance_tax_optimal_as_of_parsed_to_date() -> None:
    """Valid ISO date is parsed and forwarded as a `date` instance."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}, "as_of": "2026-04-28"},
    )
    assert resp.status_code == 200
    assert ctrl.last_call["as_of"] == date(2026, 4, 28)


def test_api_rebalance_tax_optimal_pydantic_rejects_missing_target_weights() -> None:
    """No `target_weights` → 422 from Pydantic; controller untouched."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"ltcg_used_inr": 50000.0},
    )
    assert resp.status_code == 422
    assert ctrl.last_call is None


def test_api_rebalance_tax_optimal_lots_override_threads_to_controller() -> None:
    """`lots_override` body field round-trips into the controller call as a
    plain dict-of-list-of-dict (controller parses the dates internally)."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={
            "target_weights": {"RELIANCE": 1.0},
            "lots_override": {
                "RELIANCE": [
                    {"quantity": 5.0, "cost_basis": 2400.0,
                     "acquisition_date": "2024-06-01"},
                    {"quantity": 5.0, "cost_basis": 2600.0,
                     "acquisition_date": "2025-09-15"},
                ]
            },
        },
    )
    assert resp.status_code == 200
    forwarded = ctrl.last_call["lots_override"]
    assert forwarded is not None
    assert "RELIANCE" in forwarded
    assert len(forwarded["RELIANCE"]) == 2
    assert forwarded["RELIANCE"][0]["quantity"] == 5.0
    assert forwarded["RELIANCE"][0]["cost_basis"] == 2400.0
    assert forwarded["RELIANCE"][0]["acquisition_date"] == "2024-06-01"


def test_api_rebalance_tax_optimal_lots_override_omitted_is_none() -> None:
    """When `lots_override` is omitted from the body, the controller call
    receives `None` (not an empty dict) — preserves the legacy code path."""
    ctrl = _StubController(_ok_envelope())
    client = TestClient(_make_app(ctrl))
    resp = client.post(
        "/api/portfolio/rebalance/tax-optimal",
        json={"target_weights": {"RELIANCE": 1.0}},
    )
    assert resp.status_code == 200
    assert ctrl.last_call["lots_override"] is None
