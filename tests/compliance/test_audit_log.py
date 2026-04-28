"""W5.3 T1 verification: AuditLogStore round-trip, filters, no-mongo path."""
from __future__ import annotations

import datetime as _dt
from typing import List

import pytest

from marketmind.compliance.audit_log import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_WARN,
    MAX_QUERY_LIMIT,
    SOURCE_ORDER_ATTEMPT,
    SOURCE_PRETRADE,
    AuditLogEntry,
    AuditLogStore,
)
from tests.conftest import FakeMongoCol


def _entry(
    symbol: str = "RELIANCE",
    decision: str = DECISION_ALLOW,
    *,
    ts: _dt.datetime | None = None,
    reasons: List[str] | None = None,
    source: str = SOURCE_PRETRADE,
) -> AuditLogEntry:
    return AuditLogEntry(
        ts=ts or _dt.datetime(2026, 4, 28, 10, 0, 0, tzinfo=_dt.timezone.utc),
        symbol=symbol,
        transaction_type="BUY",
        quantity=10,
        price=2500.0,
        decision=decision,
        reasons=reasons or [],
        source=source,
    )


# ─── Entry validation ────────────────────────────────────────────────────


def test_audit_log_entry_rejects_invalid_decision() -> None:
    with pytest.raises(ValueError, match="decision must be one of"):
        AuditLogEntry(
            ts=_dt.datetime.now(_dt.timezone.utc),
            symbol="X",
            transaction_type="BUY",
            quantity=1,
            price=1.0,
            decision="MAYBE",
        )


def test_audit_log_entry_rejects_naive_ts() -> None:
    with pytest.raises(ValueError, match="ts must be tz-aware"):
        AuditLogEntry(
            ts=_dt.datetime(2026, 4, 28),
            symbol="X",
            transaction_type="BUY",
            quantity=1,
            price=1.0,
            decision=DECISION_ALLOW,
        )


def test_audit_log_entry_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        _entry(source="hand_typed")


# ─── Append + query round-trip ───────────────────────────────────────────


def test_audit_log_append_and_query_round_trip(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    eid = store.append(_entry())
    assert eid is not None and eid.startswith("RELIANCE:")
    rows = store.query()
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "RELIANCE"
    assert r["decision"] == DECISION_ALLOW
    assert r["transaction_type"] == "BUY"
    assert r["source"] == SOURCE_PRETRADE
    assert r["reasons"] == []


def test_audit_log_query_orders_newest_first(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    store.append(_entry(ts=_dt.datetime(2026, 4, 27, 9, 0, 0, tzinfo=_dt.timezone.utc)))
    store.append(_entry(ts=_dt.datetime(2026, 4, 28, 9, 0, 0, tzinfo=_dt.timezone.utc)))
    store.append(_entry(ts=_dt.datetime(2026, 4, 26, 9, 0, 0, tzinfo=_dt.timezone.utc)))
    rows = store.query()
    assert [r["ts"].day for r in rows] == [28, 27, 26]


def test_audit_log_query_filters_by_symbol(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    store.append(_entry(symbol="RELIANCE"))
    store.append(_entry(symbol="TCS"))
    store.append(_entry(symbol="INFY"))
    rows = store.query(symbol="tcs")  # case-insensitive
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TCS"


def test_audit_log_query_filters_since(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    store.append(_entry(ts=_dt.datetime(2026, 4, 26, 9, 0, 0, tzinfo=_dt.timezone.utc)))
    store.append(_entry(ts=_dt.datetime(2026, 4, 28, 9, 0, 0, tzinfo=_dt.timezone.utc)))
    cutoff = _dt.datetime(2026, 4, 27, 0, 0, 0, tzinfo=_dt.timezone.utc)
    rows = store.query(since=cutoff)
    assert len(rows) == 1
    assert rows[0]["ts"].day == 28


def test_audit_log_query_since_requires_tzaware(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    with pytest.raises(ValueError, match="since must be tz-aware"):
        store.query(since=_dt.datetime(2026, 4, 28))


def test_audit_log_query_caps_limit(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    base = _dt.datetime(2026, 4, 28, 0, 0, 0, tzinfo=_dt.timezone.utc)
    for i in range(5):
        store.append(_entry(ts=base + _dt.timedelta(seconds=i)))
    assert len(store.query(limit=2)) == 2
    assert len(store.query(limit=MAX_QUERY_LIMIT + 50)) == 5
    assert store.query(limit=0) == []


# ─── No-mongo graceful degrade ───────────────────────────────────────────


def test_audit_log_no_mongo_append_is_noop_returns_none() -> None:
    store = AuditLogStore(mongo_col=None)
    assert store.append(_entry()) is None
    assert store.query() == []
    assert store.query(symbol="X", limit=10) == []


# ─── Index ensure ────────────────────────────────────────────────────────


def test_audit_log_indexes_set_on_first_write(fake_mongo_col: FakeMongoCol) -> None:
    store = AuditLogStore(mongo_col=fake_mongo_col)
    assert fake_mongo_col.indexes == []
    store.append(_entry())
    fields = {idx[0] for idx in fake_mongo_col.indexes}
    assert "symbol" in fields and "ts" in fields
    # No TTL — audit log must persist forever.
    assert all(idx[1] is None for idx in fake_mongo_col.indexes)
    # Idempotent — second write doesn't re-index.
    n_before = len(fake_mongo_col.indexes)
    store.append(_entry())
    assert len(fake_mongo_col.indexes) == n_before


# ─── Mongo throws → graceful ─────────────────────────────────────────────


class _ExplodingCol:
    def __init__(self) -> None:
        self.indexes: List[tuple] = []

    def create_index(self, field, **_):  # noqa: ANN001, ARG002
        self.indexes.append((field, None))

    def replace_one(self, *_, **__):  # noqa: ANN002, ANN003
        raise RuntimeError("mongo down")

    def find(self, *_):  # noqa: ANN002
        raise RuntimeError("mongo down")


def test_audit_log_mongo_explosions_logged_not_raised() -> None:
    store = AuditLogStore(mongo_col=_ExplodingCol())
    assert store.append(_entry()) is None
    assert store.query() == []


# ─── Decision sentinels exported ─────────────────────────────────────────


def test_audit_log_decision_and_source_sentinels_exported() -> None:
    # Smoke check that the module exposes the sentinel set callers depend on.
    assert {DECISION_ALLOW, DECISION_BLOCK, DECISION_WARN} == {"ALLOW", "BLOCK", "WARN"}
    assert {SOURCE_PRETRADE, SOURCE_ORDER_ATTEMPT} == {"pretrade", "order_attempt"}
