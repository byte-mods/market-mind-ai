"""T2 verification: AltSignal serialises, safe_fetch swallows exceptions."""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import List

import pytest

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch


def test_altsignal_serializable() -> None:
    sig = AltSignal(
        source="reddit",
        key="sentiment_tilt",
        value=0.42,
        unit="pct",
        as_of=dt.datetime(2026, 4, 27, 12, 0, tzinfo=dt.timezone.utc),
        confidence=0.8,
        raw={"posts": 100},
    )
    d = sig.to_dict()
    # Round-trips through json without TypeError
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["source"] == "reddit"
    assert decoded["value"] == 0.42
    assert decoded["as_of"] == "2026-04-27T12:00:00+00:00"
    assert decoded["raw"] == {"posts": 100}


def test_altsignal_is_frozen() -> None:
    sig = AltSignal(
        source="x", key="y", value=1, unit="", as_of=dt.datetime.now(dt.timezone.utc),
        confidence=1.0,
    )
    with pytest.raises(dataclasses_FrozenInstanceError()):
        sig.value = 2  # type: ignore[misc]


def dataclasses_FrozenInstanceError():
    import dataclasses
    return dataclasses.FrozenInstanceError


class _ExplodingSource(AltDataSource):
    SLUG = "boom"

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        raise RuntimeError("upstream 503")


class _NoneReturningSource(AltDataSource):
    SLUG = "none"

    @safe_fetch
    def fetch(self) -> List[AltSignal]:  # type: ignore[return-value]
        return None  # type: ignore[return-value]


class _GoodSource(AltDataSource):
    SLUG = "good"

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        return [
            AltSignal(
                source=self.SLUG,
                key="ok",
                value=1,
                unit="",
                as_of=dt.datetime(2026, 4, 27, tzinfo=dt.timezone.utc),
                confidence=1.0,
            )
        ]


def test_safe_fetch_swallows_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    src = _ExplodingSource()
    with caplog.at_level(logging.WARNING, logger="marketmind.core.altdata.base"):
        out = src.fetch()
    assert out == []
    assert any("upstream 503" in r.message for r in caplog.records)


def test_safe_fetch_normalises_none_to_empty_list() -> None:
    """A source that returns None must be coerced to [] so callers can iterate."""
    assert _NoneReturningSource().fetch() == []


def test_safe_fetch_passes_through_good_results() -> None:
    out = _GoodSource().fetch()
    assert len(out) == 1
    assert out[0].source == "good"
    assert out[0].confidence == 1.0
