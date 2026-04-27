"""T1 verification: the test harness boots and the conftest fixtures resolve."""
from __future__ import annotations

import pytest

from tests.conftest import FakeMongoCol


def test_conftest_fixtures_available(fake_mongo_col: FakeMongoCol) -> None:
    """fake_mongo_col injects a working in-memory collection."""
    assert isinstance(fake_mongo_col, FakeMongoCol)
    assert len(fake_mongo_col) == 0


def test_fake_mongo_upsert_round_trip(fake_mongo_col: FakeMongoCol) -> None:
    fake_mongo_col.replace_one(
        {"_id": "k1"}, {"_id": "k1", "value": 42}, upsert=True
    )
    assert fake_mongo_col.find_one({"_id": "k1"}) == {"_id": "k1", "value": 42}
    assert fake_mongo_col.write_count == 1
    # Re-upsert overwrites
    fake_mongo_col.replace_one(
        {"_id": "k1"}, {"_id": "k1", "value": 99}, upsert=True
    )
    assert fake_mongo_col.find_one({"_id": "k1"})["value"] == 99


def test_fake_mongo_create_index_records_ttl(fake_mongo_col: FakeMongoCol) -> None:
    fake_mongo_col.create_index("as_of", expireAfterSeconds=604_800)
    assert fake_mongo_col.indexes == [("as_of", 604_800)]


def test_requests_mock_plugin_loaded(requests_mock) -> None:
    """The requests-mock plugin must be installed and auto-importable."""
    requests_mock.get("https://example.invalid/x", json={"ok": True})
    import requests

    r = requests.get("https://example.invalid/x")
    assert r.json() == {"ok": True}


def test_frozen_now_pins_datetime(frozen_now) -> None:
    import datetime as dt

    assert dt.datetime.now(dt.timezone.utc) == frozen_now
