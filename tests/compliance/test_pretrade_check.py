"""W5.3 T3 verification: PretradeChecker orchestration."""
from __future__ import annotations

import datetime as _dt
from typing import Iterable, List

import pytest

from marketmind.compliance.audit_log import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_WARN,
    SOURCE_PRETRADE,
    AuditLogStore,
)
from marketmind.compliance.pretrade_check import PretradeChecker, PretradeDecision
from tests.conftest import FakeMongoCol


def _ann(date_str: str) -> dict:
    return {"date": date_str, "category": "results", "desc": "Quarterly Financial Results"}


def _h(sym: str, qty: float, last: float) -> dict:
    return {"tradingsymbol": sym, "quantity": qty, "last_price": last}


def _checker(
    designated: Iterable[str] = (),
    audit_col: FakeMongoCol | None = None,
) -> PretradeChecker:
    col = audit_col if audit_col is not None else FakeMongoCol()
    return PretradeChecker(
        audit_log=AuditLogStore(mongo_col=col),
        designated_symbols_provider=lambda: list(designated),
    )


# ─── Designated symbol in closed window → BLOCK ──────────────────────────


def test_pretrade_designated_symbol_in_closed_window_blocks() -> None:
    col = FakeMongoCol()
    checker = _checker(designated=["RELIANCE"], audit_col=col)
    decision = checker.check(
        symbol="RELIANCE",
        transaction_type="BUY",
        quantity=1,
        price=100.0,
        announcements=[_ann("2026-04-27")],
        holdings=[],
        today=_dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_BLOCK
    assert any("insider window closed" in r for r in decision.reasons)
    assert decision.insider_window_open is False
    # Audit entry written.
    rows = list(col)
    assert len(rows) == 1
    assert rows[0]["decision"] == DECISION_BLOCK
    assert rows[0]["source"] == SOURCE_PRETRADE


# ─── Non-designated symbol → insider check skipped ───────────────────────


def test_pretrade_non_designated_symbol_skips_insider_check() -> None:
    col = FakeMongoCol()
    checker = _checker(designated=["TCS"], audit_col=col)
    decision = checker.check(
        symbol="RELIANCE",
        transaction_type="BUY",
        quantity=1,
        price=100.0,
        announcements=[_ann("2026-04-27")],
        holdings=[_h("HDFC", 100, 100)],
        today=_dt.date(2026, 4, 28),
    )
    # No insider check fired.
    assert decision.insider_window_open is None
    assert decision.insider_window_reason is None
    # Concentration: 100/(10000+100) ≈ 1% → no warn.
    assert decision.decision == DECISION_ALLOW


# ─── Concentration warning → WARN (not BLOCK) ────────────────────────────


def test_pretrade_concentration_breach_warns_not_blocks() -> None:
    decision = _checker().check(
        symbol="RELIANCE",
        transaction_type="BUY",
        quantity=10,
        price=2500.0,
        announcements=[],
        holdings=[],
        today=_dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_WARN
    assert any("concentration" in r for r in decision.reasons)


# ─── Clean trade → ALLOW ─────────────────────────────────────────────────


def test_pretrade_clean_trade_allows() -> None:
    holdings = [_h("HDFC", 1000, 1000)]  # ₹1M portfolio
    decision = _checker().check(
        symbol="RELIANCE",
        transaction_type="BUY",
        quantity=1,
        price=100.0,
        announcements=[],
        holdings=holdings,
        today=_dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_ALLOW
    assert decision.reasons == []


# ─── Errors → BLOCK ──────────────────────────────────────────────────────


def test_pretrade_invalid_input_blocks() -> None:
    decision = _checker().check(
        symbol="RELIANCE",
        transaction_type="BUY",
        quantity=0,
        price=100.0,
        announcements=[],
        holdings=[],
    )
    assert decision.decision == DECISION_BLOCK
    assert any("quantity must be > 0" in r for r in decision.reasons)


# ─── Audit entry written for every check (ALLOW + WARN + BLOCK) ──────────


def test_pretrade_writes_audit_for_every_decision_class() -> None:
    col = FakeMongoCol()
    checker = _checker(audit_col=col)
    # 1 ALLOW
    checker.check(
        "X", "BUY", 1, 100.0, [], [_h("Y", 1000, 1000)], _dt.date(2026, 4, 28)
    )
    # 1 WARN (concentration)
    checker.check(
        "X", "BUY", 10, 2500.0, [], [], _dt.date(2026, 4, 28)
    )
    # 1 BLOCK (bad qty)
    checker.check(
        "X", "BUY", 0, 100.0, [], [], _dt.date(2026, 4, 28)
    )
    decisions = sorted(d["decision"] for d in col)
    assert decisions == [DECISION_ALLOW, DECISION_BLOCK, DECISION_WARN]


# ─── Designated provider failure → safe degrade ──────────────────────────


def test_pretrade_designated_provider_failure_degrades_safely() -> None:
    def boom() -> List[str]:
        raise RuntimeError("provider down")

    checker = PretradeChecker(
        audit_log=AuditLogStore(mongo_col=FakeMongoCol()),
        designated_symbols_provider=boom,
    )
    decision = checker.check(
        "RELIANCE", "BUY", 1, 100.0, [_ann("2026-04-27")], [_h("Y", 1, 1000)],
        today=_dt.date(2026, 4, 28),
    )
    # Insider check skipped because designated set falls back to empty.
    assert decision.insider_window_open is None
    assert decision.decision in {DECISION_ALLOW, DECISION_WARN}


# ─── Symbol normalisation in audit entry ─────────────────────────────────


def test_pretrade_symbol_normalised_uppercase_in_audit() -> None:
    col = FakeMongoCol()
    _checker(audit_col=col).check(
        "  reliance  ",
        "buy",
        1,
        100.0,
        [],
        [_h("Y", 100, 100)],
    )
    rows = list(col)
    assert rows[0]["symbol"] == "RELIANCE"
    assert rows[0]["transaction_type"] == "BUY"


# ─── Decision is frozen + serialisable ───────────────────────────────────


def test_pretrade_decision_frozen() -> None:
    decision = _checker().check(
        "X", "BUY", 1, 100.0, [], [], _dt.date(2026, 4, 28)
    )
    with pytest.raises(Exception):
        decision.decision = DECISION_BLOCK  # type: ignore[misc]


def test_pretrade_decision_to_dict_serialisable() -> None:
    decision = _checker().check(
        "X", "BUY", 1, 100.0, [], [_h("Y", 100, 100)], _dt.date(2026, 4, 28)
    )
    d = decision.to_dict()
    assert isinstance(d["ts"], str)  # ISO string
    assert d["decision"] in {DECISION_ALLOW, DECISION_WARN, DECISION_BLOCK}
    assert isinstance(d["reasons"], list)


# ─── Audit failure does not block decision ───────────────────────────────


class _ExplodingStore(AuditLogStore):
    def append(self, entry):  # noqa: ANN001
        raise RuntimeError("audit down")


def test_pretrade_audit_failure_does_not_block_decision() -> None:
    checker = PretradeChecker(
        audit_log=_ExplodingStore(mongo_col=FakeMongoCol()),
        designated_symbols_provider=lambda: [],
    )
    decision = checker.check(
        "X", "BUY", 1, 100.0, [], [_h("Y", 100, 100)], _dt.date(2026, 4, 28)
    )
    assert decision.decision == DECISION_ALLOW  # not derailed
    assert decision.audit_id is None


# ─── Designated symbol case + whitespace tolerated ───────────────────────


def test_pretrade_designated_symbol_lookup_normalised() -> None:
    col = FakeMongoCol()
    checker = _checker(designated=["  reliance  "], audit_col=col)
    decision = checker.check(
        "RELIANCE", "BUY", 1, 100.0, [_ann("2026-04-27")],
        [_h("Y", 100, 100)], _dt.date(2026, 4, 28),
    )
    assert decision.insider_window_open is False
    assert decision.decision == DECISION_BLOCK


# ─── Empty announcements + designated → still proceeds ───────────────────


def test_pretrade_decision_uses_structured_insider_blocked_not_substring(monkeypatch) -> None:
    """Decision must NOT depend on substring-matching the reason text.

    Repro: monkeypatch insider_window's reason to a string that does NOT
    contain the historic prefix 'insider window closed'. Decision must
    still be BLOCK because the structured flag drives it, not the text.
    """
    from marketmind.compliance import insider_window as iw_mod
    from marketmind.compliance.insider_window import InsiderWindowStatus

    def _fake(symbol, announcements, today):  # noqa: ARG001
        return InsiderWindowStatus(
            symbol=symbol,
            is_open=False,
            closed_until=_dt.date(2026, 4, 30),
            last_results_date=_dt.date(2026, 4, 27),
            reason="quarterly trading restriction in effect",  # NO old prefix
        )

    import marketmind.compliance.pretrade_check as pc_mod
    monkeypatch.setattr(pc_mod, "compute_insider_window", _fake)

    decision = _checker(designated=["RELIANCE"]).check(
        "RELIANCE", "BUY", 1, 100.0, [_ann("2026-04-27")],
        [_h("Y", 100, 100)], _dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_BLOCK
    assert decision.insider_window_open is False


def test_pretrade_oversell_blocks() -> None:
    """Position-limit over-sell error must escalate to BLOCK."""
    decision = _checker().check(
        "RELIANCE", "SELL", 100, 100.0, [],
        holdings=[_h("RELIANCE", 10, 100)],
        today=_dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_BLOCK
    assert any("only 10" in r for r in decision.reasons)


def test_pretrade_with_no_mongo_audit_log_returns_decision_with_none_audit_id() -> None:
    """PretradeChecker constructed with mongo_col=None still returns decisions;
    audit_id is None."""
    checker = PretradeChecker(
        audit_log=AuditLogStore(mongo_col=None),
        designated_symbols_provider=lambda: [],
    )
    decision = checker.check(
        "RELIANCE", "BUY", 1, 100.0, [], [_h("Y", 100, 100)],
        today=_dt.date(2026, 4, 28),
    )
    assert decision.decision == DECISION_ALLOW
    assert decision.audit_id is None


def test_pretrade_designated_with_empty_announcements_allows_with_warning_in_reason() -> None:
    decision = _checker(designated=["RELIANCE"]).check(
        "RELIANCE", "BUY", 1, 100.0, [], [_h("Y", 100, 100)],
        today=_dt.date(2026, 4, 28),
    )
    # insider_window returns is_open=True with "no announcements" reason → allow.
    assert decision.insider_window_open is True
    assert decision.decision == DECISION_ALLOW
    assert "no announcements" in (decision.insider_window_reason or "")
