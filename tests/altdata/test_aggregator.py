"""T9 verification: aggregator fan-out, Mongo persist, graceful degrade."""
from __future__ import annotations

import datetime as dt
import time
from typing import List

from marketmind.core.altdata.aggregator import AltDataAggregator, ALT_SIGNALS_TTL_S
from marketmind.core.altdata.base import AltDataSource, AltSignal


class _StaticSource(AltDataSource):
    def __init__(self, slug: str, signals: List[AltSignal]) -> None:
        self.SLUG = slug
        self._signals = signals

    def fetch(self) -> List[AltSignal]:
        return list(self._signals)


class _SlowSource(AltDataSource):
    SLUG = "slow"

    def fetch(self) -> List[AltSignal]:
        time.sleep(15)  # exceeds PER_SOURCE_TIMEOUT_S
        return []


class _RaisingSource(AltDataSource):
    SLUG = "raising"

    def fetch(self) -> List[AltSignal]:
        raise RuntimeError("upstream blew up")


def _sig(source: str, key: str, value: float = 1.0) -> AltSignal:
    return AltSignal(
        source=source, key=key, value=value, unit="x",
        as_of=dt.datetime(2026, 4, 27, 12, 0, tzinfo=dt.timezone.utc),
        confidence=0.8, raw={},
    )


def test_aggregator_writes_to_mongo(fake_mongo_col) -> None:
    src_a = _StaticSource("a", [_sig("a", "k1", 1.0), _sig("a", "k2", 2.0)])
    src_b = _StaticSource("b", [_sig("b", "k1", 3.0)])
    agg = AltDataAggregator(sources=[src_a, src_b], mongo_col=fake_mongo_col)

    out = agg.get_all()
    # Flat-dict response shape
    assert out["a"]["k1"]["value"] == 1.0
    assert out["b"]["k1"]["value"] == 3.0
    assert out["_meta"]["signal_count"] == 3
    # Mongo got 3 upserts keyed by "{source}:{key}"
    assert len(fake_mongo_col) == 3
    assert fake_mongo_col.find_one({"_id": "a:k1"})["value"] == 1.0
    assert fake_mongo_col.find_one({"_id": "b:k1"})["value"] == 3.0
    # TTL index ensured exactly once
    assert ("as_of", ALT_SIGNALS_TTL_S) in fake_mongo_col.indexes


def test_aggregator_skips_when_mongo_down() -> None:
    src = _StaticSource("a", [_sig("a", "k1")])
    agg = AltDataAggregator(sources=[src], mongo_col=None)
    out = agg.get_all()
    # Response still produced
    assert out["a"]["k1"]["value"] == 1.0
    # No persistence path attempted — nothing to assert beyond no-crash


def test_aggregator_quarantines_raising_source(fake_mongo_col) -> None:
    """One source that raises must not poison the others' results."""
    good = _StaticSource("good", [_sig("good", "ok")])
    bad = _RaisingSource()
    agg = AltDataAggregator(sources=[good, bad], mongo_col=fake_mongo_col)
    out = agg.get_all()
    assert "good" in out
    assert out["good"]["ok"]["value"] == 1.0
    # `raising` source returned [] — it must not appear as a source bucket
    assert "raising" not in out


def test_aggregator_persists_as_of_as_datetime(fake_mongo_col) -> None:
    """Mongo TTL needs a real datetime in the field, not an ISO string."""
    src = _StaticSource("a", [_sig("a", "k")])
    AltDataAggregator(sources=[src], mongo_col=fake_mongo_col).get_all()
    doc = fake_mongo_col.find_one({"_id": "a:k"})
    assert isinstance(doc["as_of"], dt.datetime), \
        f"as_of must be datetime for TTL, got {type(doc['as_of']).__name__}"


def test_aggregator_re_emit_overwrites_previous_doc(fake_mongo_col) -> None:
    src1 = _StaticSource("a", [_sig("a", "k", value=1.0)])
    AltDataAggregator(sources=[src1], mongo_col=fake_mongo_col).get_all()

    src2 = _StaticSource("a", [_sig("a", "k", value=99.0)])
    AltDataAggregator(sources=[src2], mongo_col=fake_mongo_col).get_all()

    assert len(fake_mongo_col) == 1, "re-emit should replace, not duplicate"
    assert fake_mongo_col.find_one({"_id": "a:k"})["value"] == 99.0


def test_aggregator_empty_sources_returns_minimal_response(fake_mongo_col) -> None:
    agg = AltDataAggregator(sources=[], mongo_col=fake_mongo_col)
    out = agg.get_all()
    assert out["_meta"]["signal_count"] == 0
    assert "as_of" in out
    assert len(fake_mongo_col) == 0


def test_aggregator_response_includes_meta_source_count(fake_mongo_col) -> None:
    sources = [
        _StaticSource("a", [_sig("a", "k")]),
        _StaticSource("b", [_sig("b", "k1"), _sig("b", "k2")]),
    ]
    out = AltDataAggregator(sources=sources, mongo_col=fake_mongo_col).get_all()
    assert out["_meta"]["source_count"] == 2
    assert out["_meta"]["signal_count"] == 3
