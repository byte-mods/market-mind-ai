"""T4 verification: ValuePickr Discourse parser + ticker tally."""
from __future__ import annotations

from typing import Any, Dict, List

import requests_mock as rm_module

from marketmind.core.altdata.valuepickr import (
    ValuePickrSource,
    get_valuepickr_source,
)


def _topic(title: str, replies: int = 5, views: int = 200) -> Dict[str, Any]:
    return {"title": title, "reply_count": replies, "views": views}


def _payload(topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"topic_list": {"topics": topics}}


def test_valuepickr_parses_latest_json(requests_mock: rm_module.Mocker) -> None:
    topics = [
        _topic("RELIANCE Industries — Capex cycle thesis", replies=42, views=5000),
        _topic("TCS Q3 results discussion", replies=15, views=2000),
        _topic("Banking sector outlook 2026", replies=10, views=1500),
        _topic("RELIANCE Jio + Reliance Retail bull case", replies=20, views=3000),
    ]
    requests_mock.get(
        "https://forum.valuepickr.com/top/weekly.json", json=_payload(topics)
    )

    src = ValuePickrSource(timeout_s=1.0)
    signals = src.fetch()
    by_key = {s.key: s for s in signals}

    assert by_key["weekly_thread_count"].value == 4
    assert by_key["weekly_total_replies"].value == 42 + 15 + 10 + 20
    assert by_key["weekly_total_views"].value == 5000 + 2000 + 1500 + 3000


def test_valuepickr_extracts_ticker_mentions(requests_mock: rm_module.Mocker) -> None:
    topics = [
        _topic("RELIANCE: capex cycle"),
        _topic("RELIANCE Jio + Retail"),
        _topic("TCS — large deals"),
        _topic("Banking sector outlook"),  # no ticker — no signal
    ]
    requests_mock.get(
        "https://forum.valuepickr.com/top/weekly.json", json=_payload(topics)
    )

    by_key = {s.key: s for s in ValuePickrSource(timeout_s=1.0).fetch()}
    assert by_key["top_thread_RELIANCE"].value == 2
    assert by_key["top_thread_TCS"].value == 1


def test_valuepickr_falls_back_on_503(requests_mock: rm_module.Mocker) -> None:
    requests_mock.get(
        "https://forum.valuepickr.com/top/weekly.json", status_code=503
    )
    assert ValuePickrSource(timeout_s=1.0).fetch() == []


def test_valuepickr_falls_back_on_malformed_json(requests_mock: rm_module.Mocker) -> None:
    requests_mock.get(
        "https://forum.valuepickr.com/top/weekly.json", text="<html>nope</html>"
    )
    assert ValuePickrSource(timeout_s=1.0).fetch() == []


def test_valuepickr_handles_empty_topic_list(requests_mock: rm_module.Mocker) -> None:
    requests_mock.get(
        "https://forum.valuepickr.com/top/weekly.json",
        json={"topic_list": {"topics": []}},
    )
    assert ValuePickrSource(timeout_s=1.0).fetch() == []


def test_valuepickr_singleton_returns_same_instance() -> None:
    a = get_valuepickr_source()
    b = get_valuepickr_source()
    assert a is b
