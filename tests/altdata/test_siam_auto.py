"""T5 verification: SIAM YoY math + staleness signal."""
from __future__ import annotations

import datetime as dt

from marketmind.core.altdata.siam_auto import (
    SIAM_MONTHLY_SALES,
    SiamAutoSource,
    get_siam_source,
)


def test_siam_yoy_growth_calc() -> None:
    """A 13-row table with a deterministic +10% YoY in PV must surface as 10.0%."""
    table = []
    for i in range(13):
        # 12 base months at 100k PV, then +10% in the 13th (= same calendar month + 1y)
        pv = 100000 if i < 12 else 110000
        table.append({
            "month": f"2025-{(i % 12) + 1:02d}", "pv": pv,
            "cv": 50000, "two_w": 1500000, "three_w": 50000,
        })
    signals = SiamAutoSource(table=table).fetch()
    by_key = {s.key: s for s in signals}
    assert by_key["passenger_vehicles_yoy"].value == 10.0
    # CV/2W/3W are flat → 0%
    assert by_key["commercial_vehicles_yoy"].value == 0.0


def test_siam_returns_empty_when_table_too_short() -> None:
    short = [{"month": "2025-04", "pv": 1, "cv": 1, "two_w": 1, "three_w": 1}]
    assert SiamAutoSource(table=short).fetch() == []


def test_siam_staleness_days_signal_present() -> None:
    """siam_staleness_days must always emit, even if the table is recent."""
    signals = SiamAutoSource().fetch()
    by_key = {s.key: s for s in signals}
    assert "siam_staleness_days" in by_key
    assert by_key["siam_staleness_days"].confidence == 1.0


def test_siam_confidence_drops_when_table_stale() -> None:
    """If the latest month is >45 days old, confidence on YoY must drop below 0.5."""
    # Table whose latest month is 90 days before today
    today = dt.datetime.now(dt.timezone.utc)
    months_back = []
    for offset in range(13, 0, -1):
        anchor = today - dt.timedelta(days=offset * 30 + 60)
        months_back.append({
            "month": f"{anchor.year}-{anchor.month:02d}",
            "pv": 100000, "cv": 50000, "two_w": 1500000, "three_w": 50000,
        })
    signals = SiamAutoSource(table=months_back).fetch()
    by_key = {s.key: s for s in signals}
    assert by_key["passenger_vehicles_yoy"].confidence < 0.5
    assert by_key["siam_staleness_days"].value > 45


def test_siam_3m_average_uses_last_three_months() -> None:
    table = [
        {"month": f"2025-{i:02d}", "pv": 100000 + i * 1000,
         "cv": 50000, "two_w": 1500000, "three_w": 50000} for i in range(1, 13)
    ] + [
        {"month": "2026-01", "pv": 200000, "cv": 50000, "two_w": 1500000, "three_w": 50000},
    ]
    signals = SiamAutoSource(table=table).fetch()
    by_key = {s.key: s for s in signals}
    # Last 3 PV values: 111000, 112000, 200000 → avg = 141000
    assert by_key["pv_3m_avg"].value == 141000.0


def test_siam_default_table_is_chronological() -> None:
    """Maintained table must remain in chronological order — YoY math depends on it."""
    months = [m["month"] for m in SIAM_MONTHLY_SALES]
    assert months == sorted(months)


def test_siam_singleton_returns_same_instance() -> None:
    a = get_siam_source()
    b = get_siam_source()
    assert a is b
