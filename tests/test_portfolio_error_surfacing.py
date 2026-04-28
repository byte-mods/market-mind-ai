"""Verify that Kite holdings errors surface through the controller payload
instead of being swallowed into an empty list.

These tests stub `KiteClient.holdings()` directly via a fake; they do NOT hit
the real Kite API and do not require kiteconnect to be installed.
"""
from __future__ import annotations

from typing import List, Dict, Optional

import pytest

from marketmind.core.kite_client import KiteClient


class _FakeKiteRaisesToken:
    """Stand-in for the kiteconnect client that always throws TokenException."""

    class TokenException(Exception):
        pass

    def holdings(self) -> List[Dict]:
        raise self.TokenException("Token is invalid or has expired")


class _FakeKiteRaisesNetwork:
    def holdings(self) -> List[Dict]:
        raise ConnectionError("name resolution temporarily failed")


class _FakeKiteOk:
    def __init__(self, payload: List[Dict]) -> None:
        self._payload = payload

    def holdings(self) -> List[Dict]:
        return list(self._payload)


def _make_client_with_inner(inner) -> KiteClient:
    """Construct a KiteClient with a stubbed `kite` attribute, bypassing
    the kiteconnect import path so tests don't need the package."""
    # Build a minimal config that is_authenticated=False so __init__ skips
    # the real KiteConnect import.
    from marketmind.core.kite_client import KiteConfig
    cfg = KiteConfig.__new__(KiteConfig)
    cfg.config_path = ""
    cfg.config = {}
    client = KiteClient(cfg)
    client.kite = inner
    client.is_connected = True
    return client


# ─── KiteClient ──────────────────────────────────────────────────────────


def test_get_holdings_with_diagnostic_token_exception_returns_relogin_message() -> None:
    client = _make_client_with_inner(_FakeKiteRaisesToken())
    holdings, err = client.get_holdings_with_diagnostic()
    assert holdings == []
    assert err is not None
    assert "re-login" in err.lower() or "token" in err.lower()


def test_get_holdings_with_diagnostic_unknown_exception_returns_classname() -> None:
    client = _make_client_with_inner(_FakeKiteRaisesNetwork())
    holdings, err = client.get_holdings_with_diagnostic()
    assert holdings == []
    assert err is not None
    assert "ConnectionError" in err


def test_get_holdings_with_diagnostic_success_returns_none_error() -> None:
    payload = [{"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500,
                "last_price": 2700, "day_change": 5, "exchange": "NSE", "product": "CNC"}]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    holdings, err = client.get_holdings_with_diagnostic()
    assert holdings == payload
    assert err is None


def test_get_holdings_with_diagnostic_uninitialised_client_returns_message() -> None:
    """When .kite is None, error is non-empty and explains the cause."""
    from marketmind.core.kite_client import KiteConfig
    cfg = KiteConfig.__new__(KiteConfig)
    cfg.config_path = ""
    cfg.config = {}
    client = KiteClient(cfg)
    client.kite = None  # explicitly uninitialised
    client.is_connected = False

    holdings, err = client.get_holdings_with_diagnostic()
    assert holdings == []
    assert err is not None
    assert "not initialised" in err.lower() or "local.json" in err


def test_get_holdings_legacy_silent_path_still_returns_list() -> None:
    """Legacy callers using get_holdings() (no tuple) still get back []."""
    client = _make_client_with_inner(_FakeKiteRaisesToken())
    result = client.get_holdings()
    assert result == []


# ─── Controller payload ──────────────────────────────────────────────────


def _make_controller_with_kite(kite_client: KiteClient):
    """Build a real AppController but swap its KiteClient for our fake."""
    from marketmind.app_controller import AppController
    ctrl = AppController.__new__(AppController)
    ctrl.kite = kite_client
    return ctrl


def test_portfolio_summary_token_expired_returns_authenticated_true_with_error() -> None:
    client = _make_client_with_inner(_FakeKiteRaisesToken())
    ctrl = _make_controller_with_kite(client)
    summary = ctrl.get_portfolio_summary()
    assert summary["authenticated"] is True
    assert summary["error"] is not None
    assert summary["holdings"] == []


def test_portfolio_summary_disconnected_returns_authenticated_false() -> None:
    """When .is_connected is False, the controller short-circuits with a
    'not connected' error — does not call kite.holdings()."""
    client = _make_client_with_inner(_FakeKiteOk([]))
    client.is_connected = False
    ctrl = _make_controller_with_kite(client)
    summary = ctrl.get_portfolio_summary()
    assert summary["authenticated"] is False
    assert summary["error"] is not None
    assert "Login" in summary["error"]


def test_portfolio_summary_success_no_error_field_or_none_error() -> None:
    payload = [{"tradingsymbol": "INFY", "quantity": 5, "average_price": 1500,
                "last_price": 1600, "day_change": 10, "exchange": "NSE", "product": "CNC"}]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)
    summary = ctrl.get_portfolio_summary()
    assert summary["error"] is None
    assert summary["authenticated"] is True
    assert summary["holdings"] == payload
    assert summary["total_value"] == pytest.approx(1600 * 5)
    assert summary["invested"] == pytest.approx(1500 * 5)


def test_portfolio_summary_empty_holdings_no_error() -> None:
    """Empty holdings (Kite reports nothing) is a successful response with no error."""
    client = _make_client_with_inner(_FakeKiteOk([]))
    ctrl = _make_controller_with_kite(client)
    summary = ctrl.get_portfolio_summary()
    assert summary["holdings"] == []
    assert summary["error"] is None
    assert summary["authenticated"] is True


# ─── Tax-aware rebalance (W4.1) ──────────────────────────────────────────


def test_app_controller_tax_rebalance_returns_recommendation() -> None:
    """Happy path: connected Kite + valid weights → recommendation envelope.

    Holdings have no per-lot dates (the production-realistic case), so the
    rebalancer's UNKNOWN-lot fallback fires and surfaces a warning.
    """
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
        {"tradingsymbol": "INFY", "quantity": 20, "average_price": 1500.0,
         "last_price": 1450.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 0.6, "INFY": 0.4},
    )

    assert rec["authenticated"] is True
    assert rec["error"] is None
    # Envelope keys present.
    for k in ("trades", "tax_summary", "naive_tax_summary", "savings_inr",
              "savings_pct", "tracking_error_pct", "warnings",
              "harvest_candidates", "realized_gains"):
        assert k in rec, f"missing key: {k}"
    # Lot-ledger absent → at least one warning surfaced.
    assert any("lot ledger" in w or "STCG" in w for w in rec["warnings"])


def test_app_controller_tax_rebalance_kite_disconnected_surfaces_error() -> None:
    """When Kite is not connected: authenticated=False + login prompt, no work done."""
    client = _make_client_with_inner(_FakeKiteOk([]))
    client.is_connected = False
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 1.0},
    )
    assert rec["authenticated"] is False
    assert rec["error"] is not None
    assert "Login" in rec["error"]
    assert rec["trades"] == []


def test_app_controller_tax_rebalance_kite_token_expired_surfaces_message() -> None:
    """TokenException on holdings fetch → authenticated=True + relogin error."""
    client = _make_client_with_inner(_FakeKiteRaisesToken())
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 1.0},
    )
    assert rec["authenticated"] is True
    assert rec["error"] is not None
    assert "re-login" in rec["error"].lower() or "token" in rec["error"].lower()
    assert rec["trades"] == []


def test_app_controller_tax_rebalance_invalid_weights_returns_error_not_raises() -> None:
    """Weight sum drift > 1% → returns error envelope (not propagated as 500)."""
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 0.4, "INFY": 0.4},  # sums to 0.8
        new_symbol_prices={"INFY": 1500.0},
    )
    assert rec["authenticated"] is True
    assert rec["error"] is not None
    assert "sum" in rec["error"].lower() or "1.0" in rec["error"]
    assert rec["trades"] == []


def test_app_controller_tax_rebalance_malformed_kite_row_surfaces_warning() -> None:
    """A Kite holdings row missing tradingsymbol must surface as a warning,
    not be silently dropped — the caller needs to know Kite returned junk."""
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
        # malformed row — missing tradingsymbol
        {"quantity": 99, "average_price": 100.0, "last_price": 110.0,
         "exchange": "NSE", "product": "CNC"},
        # also malformed — empty tradingsymbol
        {"tradingsymbol": "", "quantity": 5, "average_price": 50.0,
         "last_price": 55.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 1.0},
    )

    assert rec["authenticated"] is True
    assert rec["error"] is None
    # The malformed rows must be visible in warnings.
    malformed_warns = [w for w in rec["warnings"]
                       if "missing tradingsymbol" in w]
    assert len(malformed_warns) == 2, (
        f"expected 2 malformed-row warnings, got {malformed_warns}"
    )
    # Indexes preserved so user can debug from raw Kite payload.
    assert any("row 1" in w for w in malformed_warns)
    assert any("row 2" in w for w in malformed_warns)
    # Valid row still made it through to the rebalancer.
    assert rec["trades"] != [] or rec["tax_summary"] != {}


def test_app_controller_tax_rebalance_lots_override_skips_unknown_fallback() -> None:
    """When caller supplies a reconcilable per-lot ledger AND a sell is
    planned (which triggers _materialise_lots), the rebalancer uses real
    dates and the UNKNOWN-STCG-worst-case warning does NOT fire for that
    symbol — it only would fire if `lots` were None."""
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        # Force a partial sell so _materialise_lots fires for RELIANCE.
        target_weights={"RELIANCE": 0.5, "INFY": 0.5},
        new_symbol_prices={"INFY": 1500.0},
        lots_override={
            "RELIANCE": [
                {"quantity": 6.0, "cost_basis": 2400.0,
                 "acquisition_date": "2024-06-01"},
                {"quantity": 4.0, "cost_basis": 2600.0,
                 "acquisition_date": "2025-09-15"},
            ],
        },
    )

    assert rec["authenticated"] is True
    assert rec["error"] is None
    # No UNKNOWN-STCG fallback warning for RELIANCE — ledger reconciled.
    assert not any(
        "RELIANCE" in w and ("STCG worst-case" in w or "no lot ledger" in w)
        for w in rec["warnings"]
    ), f"unexpected fallback warning: {rec['warnings']}"


def test_app_controller_tax_rebalance_lots_override_bad_date_falls_back_with_warning() -> None:
    """A malformed `acquisition_date` string surfaces as a warning and the
    symbol falls back to UNKNOWN — never raises."""
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 1.0},
        lots_override={
            "RELIANCE": [
                {"quantity": 10.0, "cost_basis": 2400.0,
                 "acquisition_date": "not-a-date"},
            ],
        },
    )
    assert rec["authenticated"] is True
    assert rec["error"] is None
    assert any("lots_override parse error" in w for w in rec["warnings"])


def test_app_controller_tax_rebalance_lots_override_quantity_mismatch_falls_back() -> None:
    """When supplied lot quantities don't reconcile with Kite holding qty,
    the rebalancer's existing reconciliation logic warns + falls back to
    UNKNOWN single-lot. Reconciliation only fires when a SELL is planned for
    the symbol, so the target weights here force a partial sell of RELIANCE."""
    payload = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2500.0,
         "last_price": 2700.0, "exchange": "NSE", "product": "CNC"},
    ]
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        # Force a partial sell of RELIANCE so _materialise_lots fires.
        target_weights={"RELIANCE": 0.5, "INFY": 0.5},
        new_symbol_prices={"INFY": 1500.0},
        lots_override={
            "RELIANCE": [
                # qty sums to 7 but Kite reports 10 → mismatch
                {"quantity": 4.0, "cost_basis": 2400.0,
                 "acquisition_date": "2024-06-01"},
                {"quantity": 3.0, "cost_basis": 2600.0,
                 "acquisition_date": "2025-09-15"},
            ],
        },
    )
    assert rec["authenticated"] is True
    assert rec["error"] is None
    # tax_rebalancer._materialise_lots emits a "supplied lots sum to ..." warning.
    assert any("supplied lots sum to" in w for w in rec["warnings"]), (
        f"expected reconciliation warning, got: {rec['warnings']}"
    )


def test_app_controller_tax_rebalance_malformed_warnings_capped_at_five() -> None:
    """If Kite returns more than 5 malformed rows, individual warnings are
    capped at 5 and a summary line replaces the rest — bounded payload size."""
    # 8 malformed + 1 valid
    payload = [{"quantity": 1, "average_price": 1.0, "last_price": 1.0}
               for _ in range(8)]
    payload.append({"tradingsymbol": "RELIANCE", "quantity": 10,
                    "average_price": 2500.0, "last_price": 2700.0,
                    "exchange": "NSE", "product": "CNC"})
    client = _make_client_with_inner(_FakeKiteOk(payload))
    ctrl = _make_controller_with_kite(client)

    rec = ctrl.recommend_tax_optimal_rebalance(
        target_weights={"RELIANCE": 1.0},
    )

    individual = [w for w in rec["warnings"] if "missing tradingsymbol" in w]
    summaries = [w for w in rec["warnings"]
                 if "more malformed Kite rows" in w]
    assert len(individual) == 5, individual
    assert len(summaries) == 1
    assert "3 more" in summaries[0]  # 8 total - 5 reported = 3 elided
