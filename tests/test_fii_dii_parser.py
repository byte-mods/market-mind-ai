"""Tests for FIIDIIFetcher._parse_fiidii — NSE category-tagged + legacy shapes."""
from __future__ import annotations

from marketmind.core.fii_dii_fetcher import FIIDIIFetcher


def test_parse_groups_by_date_and_splits_by_category():
    raw = [
        {"category": "DII", "date": "08-May-2026",
         "buyValue": "21296.87", "sellValue": "14548.74", "netValue": "6748.13"},
        {"category": "FII/FPI", "date": "08-May-2026",
         "buyValue": "15083.49", "sellValue": "19194.09", "netValue": "-4110.6"},
        {"category": "DII", "date": "07-May-2026",
         "buyValue": "10000", "sellValue": "8000", "netValue": "2000"},
        {"category": "FII/FPI", "date": "07-May-2026",
         "buyValue": "12000", "sellValue": "13000", "netValue": "-1000"},
    ]
    out = FIIDIIFetcher()._parse_fiidii(raw)
    assert len(out) == 2
    # items are reversed → chronological (oldest first)
    earliest, latest = out[0], out[1]
    assert earliest["date"] == "07-May-2026"
    assert earliest["fii_net"] == -1000.0
    assert earliest["dii_net"] == 2000.0
    assert earliest["combined_net"] == 1000.0
    assert latest["date"] == "08-May-2026"
    assert latest["fii_net"] == -4110.6
    assert latest["dii_net"] == 6748.13


def test_parse_handles_legacy_single_row_shape():
    raw = [{
        "date": "07-May-2026",
        "fiiBuy": 1000, "fiiSell": 1500, "fiiNet": -500,
        "diiBuy": 800, "diiSell": 600, "diiNet": 200,
    }]
    out = FIIDIIFetcher()._parse_fiidii(raw)
    assert len(out) == 1
    assert out[0]["fii_net"] == -500.0
    assert out[0]["dii_net"] == 200.0
    assert out[0]["combined_net"] == -300.0


def test_parse_skips_rows_without_date():
    raw = [
        {"category": "FII/FPI", "buyValue": "100", "sellValue": "50", "netValue": "50"},
        {"category": "DII", "date": "01-Jan-2026",
         "buyValue": "10", "sellValue": "5", "netValue": "5"},
    ]
    out = FIIDIIFetcher()._parse_fiidii(raw)
    assert len(out) == 1
    assert out[0]["dii_net"] == 5.0


def test_parse_unknown_category_falls_back_to_legacy_columns():
    raw = [{
        "category": "UNKNOWN",
        "date": "01-Jan-2026",
        "fiiBuy": 100, "fiiSell": 80, "fiiNet": 20,
    }]
    out = FIIDIIFetcher()._parse_fiidii(raw)
    assert len(out) == 1
    assert out[0]["fii_net"] == 20.0
    assert out[0]["dii_net"] == 0.0


def test_parse_dict_envelope():
    raw = {"data": [
        {"category": "FII/FPI", "date": "01-Jan-2026",
         "buyValue": "100", "sellValue": "50", "netValue": "50"},
    ]}
    out = FIIDIIFetcher()._parse_fiidii(raw)
    assert len(out) == 1
    assert out[0]["fii_net"] == 50.0


def _stub_series(fetcher: FIIDIIFetcher, fii_nets: list[float]) -> None:
    """Pin get_fii_dii_data to a deterministic series so trend logic is testable
    without hitting NSE. Each entry contributes only fii_net; other fields are
    zero-filled to satisfy get_summary's sum() calls."""
    series = [
        {'date': f'd{i}', 'fii_buy': 0.0, 'fii_sell': 0.0, 'fii_net': n,
         'dii_buy': 0.0, 'dii_sell': 0.0, 'dii_net': 0.0, 'combined_net': n}
        for i, n in enumerate(fii_nets)
    ]
    fetcher.get_fii_dii_data = lambda days=30: series  # type: ignore[assignment]


def test_summary_trend_accelerating():
    f = FIIDIIFetcher()
    # prev3 sum = 300, last3 sum = 900 → accelerating
    _stub_series(f, [50, 100, 150, 200, 300, 400])
    summary = f.get_summary(5)
    assert summary['fii_trend'] == 'Accelerating'


def test_summary_trend_decelerating():
    f = FIIDIIFetcher()
    # prev3 sum = 900, last3 sum = 300 → decelerating
    _stub_series(f, [200, 300, 400, 50, 100, 150])
    summary = f.get_summary(5)
    assert summary['fii_trend'] == 'Decelerating'


def test_summary_trend_uses_full_series_not_days_window():
    """Regression: previously trend was computed off recent[-days:] which
    capped at `days` items, making `len(recent) >= 6` structurally impossible
    for the default days=5 caller. The trend was permanently pinned to
    'Insufficient data'. Now trend reads from the full fetched series."""
    f = FIIDIIFetcher()
    _stub_series(f, [10, 20, 30, 40, 50, 60, 70, 80])
    summary = f.get_summary(5)
    assert summary['fii_trend'] in ('Accelerating', 'Decelerating')
    assert summary['fii_trend'] != 'Insufficient data'


def test_summary_trend_insufficient_short_series():
    f = FIIDIIFetcher()
    _stub_series(f, [100, 200, 300])  # only 3 days available
    summary = f.get_summary(5)
    assert summary['fii_trend'] == 'Insufficient data'
