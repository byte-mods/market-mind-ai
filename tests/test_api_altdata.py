"""T10 verification: GET /api/altdata returns 200 with the aggregator's flat dict.

Why we don't import server.py directly: importing it triggers FastAPI app
construction, which pulls KiteClient/AppController and a full controller
init — heavy and brittle for tests. Instead we mount a tiny FastAPI app
with the same route shape and stub the aggregator. This verifies the
*wire contract* the route is supposed to honour without coupling the test
to the controller's lifecycle.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from marketmind.core.altdata.aggregator import AltDataAggregator
from marketmind.core.altdata.base import AltDataSource, AltSignal


class _StaticSource(AltDataSource):
    def __init__(self, slug: str, signals: List[AltSignal]) -> None:
        self.SLUG = slug
        self._signals = signals

    def fetch(self) -> List[AltSignal]:
        return list(self._signals)


def _sig(source: str, key: str, value: float = 1.0) -> AltSignal:
    return AltSignal(
        source=source, key=key, value=value, unit="x",
        as_of=dt.datetime(2026, 4, 27, 12, 0, tzinfo=dt.timezone.utc),
        confidence=0.8, raw={},
    )


def _make_app(agg: AltDataAggregator) -> FastAPI:
    """Build a minimal app mirroring server.py's /api/altdata wiring."""
    app = FastAPI()

    @app.get("/api/altdata")
    def altdata() -> JSONResponse:
        return JSONResponse(agg.get_all())

    return app


def test_api_altdata_route_returns_200(fake_mongo_col) -> None:
    sources = [
        _StaticSource("reddit", [_sig("reddit", "sentiment_tilt", 0.42)]),
        _StaticSource("siam", [_sig("siam", "passenger_vehicles_yoy", 8.1)]),
    ]
    app = _make_app(AltDataAggregator(sources=sources, mongo_col=fake_mongo_col))
    client = TestClient(app)

    r = client.get("/api/altdata")
    assert r.status_code == 200
    body = r.json()
    assert "reddit" in body
    assert body["reddit"]["sentiment_tilt"]["value"] == 0.42
    assert body["siam"]["passenger_vehicles_yoy"]["value"] == 8.1
    assert body["_meta"]["source_count"] == 2
    assert body["_meta"]["signal_count"] == 2


def test_api_altdata_route_handles_empty_sources() -> None:
    app = _make_app(AltDataAggregator(sources=[], mongo_col=None))
    client = TestClient(app)
    r = client.get("/api/altdata")
    assert r.status_code == 200
    body = r.json()
    assert body["_meta"]["signal_count"] == 0
    assert "as_of" in body


def test_api_altdata_route_does_not_500_when_one_source_blows_up(fake_mongo_col) -> None:
    """A misbehaving source must not propagate to a 500."""

    class _Boom(AltDataSource):
        SLUG = "boom"

        def fetch(self) -> List[AltSignal]:
            raise RuntimeError("upstream gone")

    sources = [
        _Boom(),
        _StaticSource("ok", [_sig("ok", "k", 1.0)]),
    ]
    app = _make_app(AltDataAggregator(sources=sources, mongo_col=fake_mongo_col))
    client = TestClient(app)
    r = client.get("/api/altdata")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body
    assert "boom" not in body
