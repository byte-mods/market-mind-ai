"""T3 verification: reddit fetcher parses top.json, falls back on 429."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
import requests_mock as rm_module

from marketmind.core.altdata.reddit_sentiment import (
    RedditSentimentSource,
    get_reddit_source,
)


def _post(title: str, ups: int = 100, selftext: str = "") -> Dict[str, Any]:
    return {"data": {"title": title, "ups": ups, "selftext": selftext}}


def _top_payload(posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"data": {"children": posts}}


def test_reddit_parses_top_json(requests_mock: rm_module.Mocker) -> None:
    """Happy path: bullish/bearish split + ticker tally."""
    posts = [
        _post("RELIANCE breakout — bullish target ₹3000", ups=500),
        _post("TCS multibagger rally", ups=300),
        _post("Why I'm shorting INFY — bearish breakdown", ups=200),
        _post("RELIANCE long term hold", ups=150, selftext="Accumulating RELIANCE"),
        _post("Random post about GST policy", ups=80),
    ]
    requests_mock.get(
        "https://www.reddit.com/r/IndianStockMarket/top.json",
        json=_top_payload(posts),
    )
    requests_mock.get(
        "https://www.reddit.com/r/IndiaInvestments/top.json",
        json=_top_payload([]),
    )

    src = RedditSentimentSource(timeout_s=1.0)
    signals = src.fetch()
    by_key = {s.key: s for s in signals}

    assert by_key["weekly_post_count"].value == 5
    assert by_key["weekly_upvotes"].value == 500 + 300 + 200 + 150 + 80
    # 3 bullish (breakout, multibagger, long), 1 bearish (shorting/breakdown counted once)
    assert by_key["sentiment_tilt"].value > 0
    # RELIANCE mentioned in 2 distinct posts (and selftext) → at least 3 hits
    assert by_key["top_ticker_RELIANCE"].value >= 2
    assert "top_ticker_TCS" in by_key
    # GST is in stop-words — must not surface as a ticker
    assert "top_ticker_GST" not in by_key
    # All signals share a single as_of
    timestamps = {s.as_of for s in signals}
    assert len(timestamps) == 1


def test_reddit_falls_back_on_429(requests_mock: rm_module.Mocker) -> None:
    """429 returns [] without raising. Logged at INFO not WARNING."""
    requests_mock.get(
        "https://www.reddit.com/r/IndianStockMarket/top.json",
        status_code=429,
        text="rate limited",
    )
    requests_mock.get(
        "https://www.reddit.com/r/IndiaInvestments/top.json",
        status_code=429,
        text="rate limited",
    )

    src = RedditSentimentSource(timeout_s=1.0)
    assert src.fetch() == []


def test_reddit_falls_back_on_network_error(requests_mock: rm_module.Mocker) -> None:
    requests_mock.get(
        "https://www.reddit.com/r/IndianStockMarket/top.json",
        exc=ConnectionError("dns"),
    )
    requests_mock.get(
        "https://www.reddit.com/r/IndiaInvestments/top.json",
        exc=ConnectionError("dns"),
    )

    src = RedditSentimentSource(timeout_s=1.0)
    assert src.fetch() == []


def test_reddit_handles_empty_response(requests_mock: rm_module.Mocker) -> None:
    requests_mock.get(
        "https://www.reddit.com/r/IndianStockMarket/top.json",
        json={"data": {"children": []}},
    )
    requests_mock.get(
        "https://www.reddit.com/r/IndiaInvestments/top.json",
        json={"data": {"children": []}},
    )

    src = RedditSentimentSource(timeout_s=1.0)
    # Empty corpus → no signals (not zero-valued ones)
    assert src.fetch() == []


def test_reddit_singleton_returns_same_instance() -> None:
    a = get_reddit_source()
    b = get_reddit_source()
    assert a is b


def test_reddit_no_brotli_in_accept_encoding(requests_mock: rm_module.Mocker) -> None:
    """Hard rule: never add 'br' — requests can't decode brotli without
    the brotli package, and we got burned by this on NSE before."""
    captured: list[str] = []

    def _capture(request, _context):
        captured.append(request.headers.get("Accept-Encoding", ""))
        return {"data": {"children": []}}

    requests_mock.get(
        "https://www.reddit.com/r/IndianStockMarket/top.json", json=_capture
    )
    requests_mock.get(
        "https://www.reddit.com/r/IndiaInvestments/top.json", json=_capture
    )

    RedditSentimentSource(timeout_s=1.0).fetch()
    assert captured, "no requests captured"
    for ae in captured:
        assert "br" not in ae.split(","), f"Accept-Encoding contains br: {ae}"
