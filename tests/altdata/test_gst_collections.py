"""T6 verification: GST YoY signal + staleness."""
from __future__ import annotations

import datetime as dt

from marketmind.core.altdata.gst_collections import (
    GST_COLLECTIONS_HISTORY,
    GstCollectionsSource,
    get_gst_source,
)


def test_gst_yoy_signal() -> None:
    table = []
    for i in range(12):
        table.append({"month": f"2025-{i + 1:02d}", "gross_inr_cr": 100000})
    table.append({"month": "2026-01", "gross_inr_cr": 120000})  # +20% YoY
    by_key = {s.key: s for s in GstCollectionsSource(table=table).fetch()}
    assert by_key["gst_yoy_pct"].value == 20.0


def test_gst_returns_empty_when_table_too_short() -> None:
    assert GstCollectionsSource(table=[{"month": "2025-04", "gross_inr_cr": 1}]).fetch() == []


def test_gst_3m_avg_uses_last_three_months() -> None:
    table = [{"month": f"2025-{i:02d}", "gross_inr_cr": 100000} for i in range(1, 13)] + [
        {"month": "2026-01", "gross_inr_cr": 130000},
    ]
    by_key = {s.key: s for s in GstCollectionsSource(table=table).fetch()}
    # Last 3: 100000, 100000, 130000 → avg ≈ 110000
    assert by_key["gst_3m_avg_inr_cr"].value == 110000.0


def test_gst_confidence_drops_when_stale() -> None:
    today = dt.datetime.now(dt.timezone.utc)
    table = []
    for offset in range(13, 0, -1):
        anchor = today - dt.timedelta(days=offset * 30 + 60)
        table.append({"month": f"{anchor.year}-{anchor.month:02d}",
                      "gross_inr_cr": 100000})
    by_key = {s.key: s for s in GstCollectionsSource(table=table).fetch()}
    assert by_key["gst_yoy_pct"].confidence < 0.5
    assert by_key["gst_staleness_days"].value > 35


def test_gst_staleness_signal_always_emitted() -> None:
    by_key = {s.key: s for s in GstCollectionsSource().fetch()}
    assert "gst_staleness_days" in by_key
    assert by_key["gst_staleness_days"].confidence == 1.0


def test_gst_default_table_chronological() -> None:
    months = [m["month"] for m in GST_COLLECTIONS_HISTORY]
    assert months == sorted(months)


def test_gst_singleton_returns_same_instance() -> None:
    assert get_gst_source() is get_gst_source()
