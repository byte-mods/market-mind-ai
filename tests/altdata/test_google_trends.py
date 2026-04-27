"""T8 verification: Google Trends source — pytrends optional, safe-degrade."""
from __future__ import annotations

import sys
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from marketmind.core.altdata.google_trends import (
    GoogleTrendsSource,
    get_trends_source,
)


class _FakePyTrendsClient:
    """Minimal duck-typed stand-in for pytrends.TrendReq."""

    def __init__(self, frame: Dict[str, List[float]]) -> None:
        self._frame = frame
        self.last_payload: Dict[str, object] = {}

    def build_payload(self, kw_list, timeframe, geo) -> None:
        self.last_payload = {"kw_list": kw_list, "timeframe": timeframe, "geo": geo}

    def interest_over_time(self) -> object:
        return _FakeFrame(self._frame)


class _FakeFrame:
    """Mimics the pandas DataFrame surface GoogleTrendsSource uses:
    ``len(df)``, ``sym in df.columns``, ``df[sym].mean()``."""

    def __init__(self, data: Dict[str, List[float]]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(next(iter(self._data.values()), []))

    @property
    def columns(self):
        return list(self._data.keys())

    def __getitem__(self, key: str) -> "_FakeSeries":
        return _FakeSeries(self._data[key])


class _FakeSeries:
    def __init__(self, vals: List[float]) -> None:
        self._vals = vals

    def mean(self) -> float:
        return sum(self._vals) / len(self._vals) if self._vals else 0.0


def test_google_trends_returns_empty_when_pytrends_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate pytrends not being importable. Source must degrade silently."""
    # Block the import — sys.modules sentinel + ImportError on lookup
    monkeypatch.setitem(sys.modules, "pytrends", None)
    monkeypatch.setitem(sys.modules, "pytrends.request", None)

    src = GoogleTrendsSource()
    # No factory injected → real import path → ImportError → []
    assert src.fetch() == []


def test_google_trends_emits_signals_when_data_available() -> None:
    fake_data = {
        "RELIANCE":  [50, 55, 60, 58, 62, 65, 70],
        "TCS":       [40, 42, 38, 41, 39, 40, 42],
        "INFY":      [30, 28, 32, 35, 30, 29, 31],
        "HDFCBANK":  [60, 58, 62, 65, 60, 58, 55],
        "ICICIBANK": [50, 52, 51, 49, 50, 52, 53],
    }
    factory = lambda: _FakePyTrendsClient(fake_data)
    src = GoogleTrendsSource(pytrends_factory=factory)

    signals = src.fetch()
    by_key = {s.key: s for s in signals}
    assert "trend_RELIANCE_score" in by_key
    assert "trend_TCS_score" in by_key
    assert by_key["trend_RELIANCE_score"].value == round(sum(fake_data["RELIANCE"]) / 7, 1)
    # All trend signals share the fixed-confidence rule
    assert all(s.confidence == 0.6 for s in signals)


def test_google_trends_falls_back_on_exception() -> None:
    """If pytrends client raises, return [] without propagating."""
    bad_client = MagicMock()
    bad_client.build_payload.side_effect = RuntimeError("429 rate limited")
    src = GoogleTrendsSource(pytrends_factory=lambda: bad_client)
    assert src.fetch() == []


def test_google_trends_handles_empty_frame() -> None:
    src = GoogleTrendsSource(pytrends_factory=lambda: _FakePyTrendsClient({}))
    assert src.fetch() == []


def test_google_trends_caps_tickers_at_five() -> None:
    """Google's API limits batches to 5; constructor must enforce this silently."""
    too_many = ("A", "B", "C", "D", "E", "F", "G")
    src = GoogleTrendsSource(tickers=too_many)
    assert len(src.tickers) == 5


def test_google_trends_singleton() -> None:
    assert get_trends_source() is get_trends_source()
