"""Unit tests for OptionsFetcher — F&O symbol list + option chain parsing."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from marketmind.core.options_fetcher import OptionsFetcher, get_options_fetcher


class _StubSession:
    """Minimal stub for requests.Session."""

    def __init__(self, json_response: Dict[str, Any] = None, status_code: int = 200) -> None:
        self._json = json_response or {}
        self._status = status_code
        self.calls: list[str] = []

    def get(self, url: str, timeout: float = 10) -> "_StubSession":
        self.calls.append(url)
        return self

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise Exception(f"HTTP {self._status}")

    def json(self) -> Dict[str, Any]:
        return self._json


def test_get_fo_symbols_parses_nse_response() -> None:
    """FO symbols are extracted from NSE equity-stockIndices response."""
    fetcher = OptionsFetcher()
    fetcher._session = _StubSession(json_response={
        "data": [
            {"symbol": "RELIANCE"},
            {"symbol": "TCS"},
            {"symbol": "INFY"},
        ]
    })
    symbols = fetcher.get_fo_symbols()
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "INFY" in symbols
    assert "NIFTY" in symbols  # indices injected
    assert symbols == sorted(symbols)


def test_get_fo_symbols_fallback_on_error() -> None:
    """On NSE failure, returns the hardcoded fallback list so UI isn't bricked."""
    fetcher = OptionsFetcher()
    fetcher._session = _StubSession(status_code=500)
    symbols = fetcher.get_fo_symbols()
    assert len(symbols) > 20
    assert "RELIANCE" in symbols
    assert "NIFTY" in symbols


def test_get_option_chain_returns_unavailable_on_error() -> None:
    """When NSE is down, the response carries `unavailable: True` + human reason."""
    fetcher = OptionsFetcher()
    fetcher._session = _StubSession(status_code=503)
    result = fetcher.get_option_chain("RELIANCE")
    assert result.get("unavailable") is True
    assert "symbol" in result
    assert result["symbol"] == "RELIANCE"
    assert "reason" in result


def test_parse_chain_empty_records() -> None:
    """Empty NSE payload → unavailable dict with all required keys."""
    fetcher = OptionsFetcher()
    result = fetcher._parse_chain({}, "RELIANCE")
    assert result.get("unavailable") is True
    assert result["symbol"] == "RELIANCE"
    assert result["calls"] == []
    assert result["puts"] == []
    assert "timestamp" in result


def test_parse_chain_computes_metrics() -> None:
    """Correct OI aggregation, PCR, max-pain, ATM strike."""
    fetcher = OptionsFetcher()
    raw = {
        "records": {
            "underlyingValue": 2500,
            "expiryDates": ["2026-05-15", "2026-05-22"],
            "data": [
                {
                    "strikePrice": 2500,
                    "CE": {
                        "openInterest": 100000,
                        "changeinOpenInterest": 5000,
                        "totalTradedVolume": 20000,
                        "impliedVolatility": 18.5,
                        "lastPrice": 45.0,
                        "bidprice": 44.5,
                        "askPrice": 45.5,
                    },
                    "PE": {
                        "openInterest": 120000,
                        "changeinOpenInterest": 8000,
                        "totalTradedVolume": 25000,
                        "impliedVolatility": 19.0,
                        "lastPrice": 38.0,
                        "bidprice": 37.5,
                        "askPrice": 38.5,
                    },
                },
                {
                    "strikePrice": 2600,
                    "CE": {
                        "openInterest": 80000,
                        "changeinOpenInterest": 3000,
                        "totalTradedVolume": 15000,
                        "impliedVolatility": 17.0,
                        "lastPrice": 20.0,
                        "bidprice": 19.5,
                        "askPrice": 20.5,
                    },
                    "PE": {
                        "openInterest": 60000,
                        "changeinOpenInterest": 2000,
                        "totalTradedVolume": 10000,
                        "impliedVolatility": 16.5,
                        "lastPrice": 65.0,
                        "bidprice": 64.5,
                        "askPrice": 65.5,
                    },
                },
            ],
        }
    }
    result = fetcher._parse_chain(raw, "RELIANCE")
    assert result["symbol"] == "RELIANCE"
    assert result["underlying"] == 2500
    assert result["atm_strike"] == 2500
    assert result["total_call_oi"] == 180000
    assert result["total_put_oi"] == 180000
    assert result["pcr"] == 1.0
    assert result["max_call_oi_strike"] == 2500
    assert result["max_put_oi_strike"] == 2500
    assert len(result["calls"]) == 2
    assert len(result["puts"]) == 2
    assert result["calls"][0]["ltp"] == 45.0
    assert result["puts"][0]["ltp"] == 38.0
    assert result["sentiment"] == "Neutral"
