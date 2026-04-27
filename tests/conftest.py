"""
Shared pytest fixtures for the MarketMind test suite.

Design notes:
- `fake_mongo_col` is a duck-typed in-memory stand-in for a `pymongo.Collection`.
  It implements only the surface used in this codebase: `find_one`, `find`,
  `replace_one(upsert=True)`, `insert_many`, `create_index`, and iteration.
  Tests should NOT pull in `mongomock` — every real-mongo behaviour we depend
  on (TTL, upsert-by-key, find-by-_id) is trivially modelled with a dict.
- Network-touching fetchers MUST be tested through the `requests_mock` fixture
  provided by the `requests-mock` plugin. No live HTTP in tests.
- The `frozen_now` fixture pins `datetime.now()` for the macro-cadence
  alt-data sources whose values depend on staleness windows.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pytest


class FakeMongoCol:
    """In-memory dict-backed pymongo.Collection stand-in.

    Indexes documents by the value of `_id` (or the `key_field` passed in for
    collections that don't use `_id` as their primary key — e.g. `news` keys
    on `url`, `prices` on `symbol`).
    """

    def __init__(self, key_field: str = "_id") -> None:
        self._key_field = key_field
        self._docs: Dict[Any, Dict[str, Any]] = {}
        self.indexes: List[tuple] = []
        # Test introspection helpers — not on the real pymongo API.
        self.write_count = 0
        self.read_count = 0

    # ─── pymongo surface ──────────────────────────────────────────────
    def create_index(self, field, expireAfterSeconds: Optional[int] = None, **_: Any) -> None:
        self.indexes.append((field, expireAfterSeconds))

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.read_count += 1
        if self._key_field in query:
            return self._docs.get(query[self._key_field])
        # naive linear scan fallback for non-key queries
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    def find(self, query: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        self.read_count += 1
        query = query or {}
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                yield doc

    def replace_one(
        self,
        query: Dict[str, Any],
        doc: Dict[str, Any],
        upsert: bool = False,
    ) -> None:
        self.write_count += 1
        # Resolve key: prefer query[key_field], fall back to doc[key_field]
        key = query.get(self._key_field, doc.get(self._key_field))
        if key is None:
            raise ValueError(f"replace_one needs {self._key_field} in query or doc")
        if key in self._docs or upsert:
            stored = dict(doc)
            stored.setdefault(self._key_field, key)
            self._docs[key] = stored

    def insert_many(self, docs: Iterable[Dict[str, Any]]) -> None:
        for d in docs:
            self.write_count += 1
            key = d.get(self._key_field)
            if key is None:
                raise ValueError(f"insert_many doc missing {self._key_field}")
            self._docs[key] = dict(d)

    # ─── test helpers (not pymongo) ───────────────────────────────────
    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._docs.values())

    def clear(self) -> None:
        self._docs.clear()
        self.write_count = 0
        self.read_count = 0


@pytest.fixture
def fake_mongo_col() -> FakeMongoCol:
    """Default _id-keyed fake collection."""
    return FakeMongoCol(key_field="_id")


@pytest.fixture
def fake_mongo_alt_signals() -> FakeMongoCol:
    """`alt_signals` collection — keyed by composite `source:key`."""
    return FakeMongoCol(key_field="_id")


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> _dt.datetime:
    """Pin `datetime.now()` and `datetime.utcnow()` to a fixed point.

    Useful for tests that compute YoY/MoM deltas off maintained history
    tables, where 'now' must be deterministic.
    """
    fixed = _dt.datetime(2026, 4, 27, 12, 0, 0, tzinfo=_dt.timezone.utc)

    class _FrozenDT(_dt.datetime):
        @classmethod
        def now(cls, tz: Optional[_dt.tzinfo] = None) -> _dt.datetime:
            return fixed if tz else fixed.replace(tzinfo=None)

        @classmethod
        def utcnow(cls) -> _dt.datetime:
            return fixed.replace(tzinfo=None)

    monkeypatch.setattr(_dt, "datetime", _FrozenDT)
    return fixed
