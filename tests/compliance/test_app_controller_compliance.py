"""W5.3 T4 verification: AppController compliance integration + place_order audit hook.

Uses ``AppController.__new__`` to bypass the heavy ``__init__`` (kite, RL,
mongo connect) and manually wires only the fields each test needs. Mirrors
the pattern used in ``tests/test_portfolio_error_surfacing.py``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

import pytest

from marketmind.app_controller import AppController
from marketmind.compliance.audit_log import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    SOURCE_ORDER_ATTEMPT,
    SOURCE_PRETRADE,
    AuditLogStore,
)
from marketmind.compliance.pretrade_check import PretradeChecker
from tests.conftest import FakeMongoCol


# ─── Test helpers ────────────────────────────────────────────────────────


class _FakeKite:
    """Minimal stand-in mirroring the AppController.kite surface used here."""

    def __init__(
        self,
        is_connected: bool = True,
        holdings: Optional[List[Dict]] = None,
        order_id: Optional[str] = "ORDER123",
        raise_on_place: bool = False,
    ) -> None:
        self.is_connected = is_connected
        self._holdings = holdings or []
        self._order_id = order_id
        self.last_place_kwargs: Optional[Dict] = None
        self._raise_on_place = raise_on_place

    def get_holdings(self) -> List[Dict]:
        return list(self._holdings)

    def place_order(self, **kwargs):
        if self._raise_on_place:
            raise RuntimeError("kite blew up")
        self.last_place_kwargs = kwargs
        return self._order_id


def _build_controller(
    kite: _FakeKite,
    *,
    audit_col: Optional[FakeMongoCol] = None,
    designated_col: Optional[FakeMongoCol] = None,
) -> AppController:
    """Bypass __init__ and inject only what the compliance methods touch."""
    ctrl = AppController.__new__(AppController)
    ctrl.kite = kite
    ctrl._mongo_db = None
    ctrl._compliance_designated_cache = None
    audit = AuditLogStore(mongo_col=audit_col)
    ctrl._compliance_audit_store = audit

    if designated_col is not None:
        # Wire a real Mongo-col-backed lookup by overriding _mongo_col.
        def _col(name, _audit=audit_col, _des=designated_col):
            if name == "compliance_audit_log":
                return _audit
            if name == "compliance_designated":
                return _des
            return None
        ctrl._mongo_col = _col  # type: ignore[method-assign]
    else:
        ctrl._mongo_col = lambda name: None  # type: ignore[method-assign]

    ctrl.compliance = PretradeChecker(
        audit_log=audit,
        designated_symbols_provider=ctrl._compliance_designated_symbols,
    )
    return ctrl


def _h(sym: str, qty: float, last: float) -> Dict:
    return {"tradingsymbol": sym, "quantity": qty, "last_price": last, "average_price": last}


# ─── compliance_pretrade_check ───────────────────────────────────────────


def test_app_controller_compliance_pretrade_returns_envelope() -> None:
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(holdings=[_h("HDFC", 1000, 1000)]), audit_col=audit)
    out = ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    assert out["authenticated"] is True
    assert out["error"] is None
    assert out["decision"] == DECISION_ALLOW
    assert isinstance(out["reasons"], list)
    # Audit entry written.
    assert len(list(audit)) == 1
    assert list(audit)[0]["source"] == SOURCE_PRETRADE


def test_app_controller_compliance_pretrade_empty_symbol_blocks() -> None:
    ctrl = _build_controller(_FakeKite(), audit_col=FakeMongoCol())
    out = ctrl.compliance_pretrade_check("", "BUY", 1, 100.0)
    assert out["decision"] == DECISION_BLOCK
    assert "non-empty" in out["error"]


def test_app_controller_compliance_pretrade_kite_disconnected_proceeds() -> None:
    """Compliance must work even when Kite is not authenticated — holdings
    fall back to []. authenticated flag reflects state."""
    ctrl = _build_controller(
        _FakeKite(is_connected=False), audit_col=FakeMongoCol()
    )
    out = ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    assert out["authenticated"] is False
    # Empty holdings + buy → 100% concentration → WARN (not BLOCK).
    assert out["decision"] in {"WARN", "ALLOW"}


# ─── place_order audit hook ──────────────────────────────────────────────


def test_app_controller_place_order_writes_audit_when_kite_disconnected() -> None:
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(is_connected=False), audit_col=audit)
    result = ctrl.place_order(
        tradingsymbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        order_type="MARKET",
        price=100.0,
    )
    assert result is None
    rows = list(audit)
    assert len(rows) == 1
    assert rows[0]["decision"] == DECISION_BLOCK
    assert rows[0]["source"] == SOURCE_ORDER_ATTEMPT
    assert "kite not authenticated" in rows[0]["reasons"][0]


def test_app_controller_place_order_writes_audit_on_success() -> None:
    audit = FakeMongoCol()
    ctrl = _build_controller(
        _FakeKite(order_id="ORDER42"), audit_col=audit
    )
    result = ctrl.place_order(
        tradingsymbol="reliance",  # lower-case input
        exchange="NSE",
        transaction_type="buy",
        quantity=5,
        order_type="MARKET",
        price=2500.0,
    )
    assert result == "ORDER42"
    rows = list(audit)
    assert len(rows) == 1
    assert rows[0]["decision"] == DECISION_ALLOW
    assert rows[0]["symbol"] == "RELIANCE"  # normalised
    assert rows[0]["transaction_type"] == "BUY"  # normalised
    assert rows[0]["source"] == SOURCE_ORDER_ATTEMPT
    assert "ORDER42" in rows[0]["reasons"][0]


def test_app_controller_place_order_writes_audit_on_kite_rejection() -> None:
    """Kite returning None (rejected order) is logged as BLOCK."""
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(order_id=None), audit_col=audit)
    result = ctrl.place_order(
        tradingsymbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        order_type="MARKET",
        price=100.0,
    )
    assert result is None
    rows = list(audit)
    assert len(rows) == 1
    assert rows[0]["decision"] == DECISION_BLOCK
    assert "rejected" in rows[0]["reasons"][0]


def test_app_controller_place_order_audit_failure_does_not_break_order() -> None:
    """Audit-log Mongo down must not break the trading path."""
    ctrl = _build_controller(_FakeKite(order_id="OK"), audit_col=None)
    # No mongo; AuditLogStore.append no-ops; place_order still returns the id.
    assert ctrl.place_order(
        tradingsymbol="X", exchange="NSE", transaction_type="BUY",
        quantity=1, order_type="MARKET", price=100.0,
    ) == "OK"


def test_app_controller_place_order_kite_raise_writes_audit_and_returns_none() -> None:
    """If Kite leaks an exception, audit row is still written and the order
    silently fails (return None) — preserves regulatory audit guarantee."""
    audit = FakeMongoCol()
    ctrl = _build_controller(
        _FakeKite(raise_on_place=True), audit_col=audit
    )
    result = ctrl.place_order(
        tradingsymbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        order_type="MARKET",
        price=100.0,
    )
    assert result is None
    rows = list(audit)
    assert len(rows) == 1
    assert rows[0]["decision"] == DECISION_BLOCK
    assert "kite raised" in rows[0]["reasons"][0]
    assert "RuntimeError" in rows[0]["reasons"][0]


def test_app_controller_designated_cache_isolation() -> None:
    """Mutating the returned set MUST NOT contaminate the controller cache."""
    audit = FakeMongoCol()
    designated = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(), audit_col=audit, designated_col=designated)
    ctrl.compliance_set_designated_symbols(["RELIANCE"])
    s1 = ctrl._compliance_designated_symbols()
    s1.add("PWNED")  # caller mutation
    s2 = ctrl._compliance_designated_symbols()
    assert "PWNED" not in s2
    assert s2 == {"RELIANCE"}


# ─── designated symbols round-trip ───────────────────────────────────────


def test_app_controller_designated_symbols_round_trip() -> None:
    audit = FakeMongoCol()
    designated = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(), audit_col=audit, designated_col=designated)

    out = ctrl.compliance_set_designated_symbols(["reliance", "TCS", "  infy  ", ""])
    assert out["error"] is None
    assert out["symbols"] == ["RELIANCE", "TCS", "INFY"]

    # Cache invalidated; new lookup returns the persisted set.
    assert ctrl._compliance_designated_symbols() == {"RELIANCE", "TCS", "INFY"}

    # Re-set with deduplication
    out2 = ctrl.compliance_set_designated_symbols(["TCS", "TCS", "WIPRO"])
    assert out2["symbols"] == ["TCS", "WIPRO"]
    assert ctrl._compliance_designated_symbols() == {"TCS", "WIPRO"}


def test_app_controller_designated_no_mongo_returns_error_but_holds_in_memory() -> None:
    ctrl = _build_controller(_FakeKite(), audit_col=FakeMongoCol(), designated_col=None)
    out = ctrl.compliance_set_designated_symbols(["RELIANCE"])
    assert "mongo not available" in (out["error"] or "")
    assert out["symbols"] == ["RELIANCE"]
    # Cache holds the value in-process even without persistence.
    assert ctrl._compliance_designated_symbols() == {"RELIANCE"}


# ─── compliance_get_audit_log ────────────────────────────────────────────


def test_app_controller_get_audit_log_returns_entries_with_iso_ts() -> None:
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(), audit_col=audit)
    # Trigger one entry via pretrade
    ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    out = ctrl.compliance_get_audit_log()
    assert out["error"] is None
    assert len(out["entries"]) == 1
    assert isinstance(out["entries"][0]["ts"], str)  # ISO string


def test_app_controller_get_audit_log_bad_since_returns_400_envelope() -> None:
    ctrl = _build_controller(_FakeKite(), audit_col=FakeMongoCol())
    out = ctrl.compliance_get_audit_log(since="not-a-date")
    assert "must be ISO" in out["error"]
    assert out["entries"] == []


def test_app_controller_get_audit_log_filters_by_symbol() -> None:
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(), audit_col=audit)
    ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    ctrl.compliance_pretrade_check("TCS", "BUY", 1, 100.0)
    out = ctrl.compliance_get_audit_log(symbol="reliance")
    assert len(out["entries"]) == 1
    assert out["entries"][0]["symbol"] == "RELIANCE"


def test_app_controller_pretrade_then_place_order_round_trip_audit_log() -> None:
    """Two source-typed audit rows in one trade flow: pretrade gate writes
    `source=pretrade`, the actual order writes `source=order_attempt`.
    Both must surface via `compliance_get_audit_log`, newest-first, with
    `_id` stripped and `ts` ISO-formatted."""
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(order_id="ORDER77"), audit_col=audit)

    # 1) pretrade gate
    pre = ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    assert pre["error"] is None
    assert pre["audit_id"] is not None  # exactly one pretrade row written

    # 2) the actual order — separate audit row, source=order_attempt
    order_id = ctrl.place_order(
        tradingsymbol="RELIANCE",
        exchange="NSE",
        transaction_type="BUY",
        quantity=1,
        order_type="MARKET",
        price=100.0,
    )
    assert order_id == "ORDER77"

    # Underlying store has BOTH rows.
    raw = list(audit)
    assert len(raw) == 2
    sources = {r["source"] for r in raw}
    assert sources == {SOURCE_PRETRADE, SOURCE_ORDER_ATTEMPT}

    # API returns both, newest-first, no _id leak, ts as ISO string.
    out = ctrl.compliance_get_audit_log(symbol="RELIANCE")
    assert out["error"] is None
    assert len(out["entries"]) == 2
    for e in out["entries"]:
        assert "_id" not in e
        assert isinstance(e["ts"], str)
        assert e["symbol"] == "RELIANCE"
    api_sources = [e["source"] for e in out["entries"]]
    assert set(api_sources) == {SOURCE_PRETRADE, SOURCE_ORDER_ATTEMPT}
    # Newest-first ordering: order_attempt was appended after pretrade,
    # so it should be index 0.
    assert api_sources[0] == SOURCE_ORDER_ATTEMPT
    assert api_sources[1] == SOURCE_PRETRADE


def test_app_controller_get_audit_log_strips_internal_id_field() -> None:
    """Mongo `_id` is a write-collision-safe internal key, not a regulatory
    contract field. The audit-log payload must never leak it; consumers
    use `audit_id` from the pretrade response to reference an entry."""
    audit = FakeMongoCol()
    ctrl = _build_controller(_FakeKite(), audit_col=audit)
    pre = ctrl.compliance_pretrade_check("RELIANCE", "BUY", 1, 100.0)
    # Underlying store DOES persist `_id` (write-collision safety).
    raw = list(audit)
    assert raw and "_id" in raw[0]
    # API surface must NOT expose it.
    out = ctrl.compliance_get_audit_log()
    assert out["error"] is None
    assert len(out["entries"]) == 1
    assert "_id" not in out["entries"][0]
    # `audit_id` from the pretrade response is the canonical reference.
    assert pre["audit_id"] == raw[0]["_id"]


# ─── compliance_get_insider_window ───────────────────────────────────────


def test_app_controller_get_insider_window_handles_fetch_failure(monkeypatch) -> None:
    ctrl = _build_controller(_FakeKite(), audit_col=FakeMongoCol())

    class _BoomIngester:
        def fetch_announcements(self, symbol):
            raise RuntimeError("nse down")

    import marketmind.core.filings_ingest as fi
    monkeypatch.setattr(fi, "get_filings_ingester", lambda: _BoomIngester())

    out = ctrl.compliance_get_insider_window("RELIANCE")
    assert "nse down" in out["error"]
    # Empty announcements → window open with "no announcements" reason.
    assert out["is_open"] is True
    assert "no announcements" in out["reason"]


def test_app_controller_get_insider_window_empty_symbol() -> None:
    ctrl = _build_controller(_FakeKite(), audit_col=FakeMongoCol())
    out = ctrl.compliance_get_insider_window("")
    assert "non-empty" in out["error"]
    assert out["is_open"] is None
