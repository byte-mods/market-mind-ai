"""T7 verification: IIP/CPI YoY + macro stance derivation."""
from __future__ import annotations

import datetime as dt

from marketmind.core.altdata.iip_cpi import (
    CPI_HISTORY,
    IIP_HISTORY,
    IipCpiSource,
    get_iip_cpi_source,
)


def test_iip_cpi_stance_inflationary_vs_disinflationary() -> None:
    """High CPI + high IIP → Inflationary. Low CPI + high IIP → Disinflationary."""
    iip_high = [{"month": f"2025-{i:02d}", "general_index": 100} for i in range(1, 13)] + [
        {"month": "2026-01", "general_index": 110},  # +10% YoY → high
    ]
    cpi_high = [{"month": f"2025-{i:02d}", "yoy_pct": 6.0} for i in range(1, 13)] + [
        {"month": "2026-01", "yoy_pct": 6.5},
    ]
    cpi_low = [{"month": f"2025-{i:02d}", "yoy_pct": 3.0} for i in range(1, 13)] + [
        {"month": "2026-01", "yoy_pct": 3.0},
    ]

    inflationary = {s.key: s for s in IipCpiSource(iip_high, cpi_high).fetch()}
    disinflationary = {s.key: s for s in IipCpiSource(iip_high, cpi_low).fetch()}

    assert inflationary["macro_stance"].value == "Inflationary"
    assert disinflationary["macro_stance"].value == "Disinflationary"


def test_iip_cpi_stance_stagflation() -> None:
    iip_low = [{"month": f"2025-{i:02d}", "general_index": 100} for i in range(1, 13)] + [
        {"month": "2026-01", "general_index": 100},  # 0% YoY → low
    ]
    cpi_high = [{"month": f"2025-{i:02d}", "yoy_pct": 6.0} for i in range(1, 13)] + [
        {"month": "2026-01", "yoy_pct": 7.5},
    ]
    by_key = {s.key: s for s in IipCpiSource(iip_low, cpi_high).fetch()}
    assert by_key["macro_stance"].value == "Stagflation"


def test_iip_cpi_stance_stable_when_mid_range() -> None:
    iip_mid = [{"month": f"2025-{i:02d}", "general_index": 100} for i in range(1, 13)] + [
        {"month": "2026-01", "general_index": 102.5},  # +2.5% YoY → mid
    ]
    cpi_mid = [{"month": f"2025-{i:02d}", "yoy_pct": 4.5} for i in range(1, 13)] + [
        {"month": "2026-01", "yoy_pct": 4.5},
    ]
    by_key = {s.key: s for s in IipCpiSource(iip_mid, cpi_mid).fetch()}
    assert by_key["macro_stance"].value == "Stable"


def test_iip_cpi_emits_when_only_cpi_table_present() -> None:
    """If IIP table is too short, CPI signals must still emit (no stance though)."""
    iip_short = [{"month": "2025-04", "general_index": 100}]
    cpi_full = [{"month": f"2025-{i:02d}", "yoy_pct": 4.5} for i in range(1, 13)] + [
        {"month": "2026-01", "yoy_pct": 4.0},
    ]
    by_key = {s.key: s for s in IipCpiSource(iip_short, cpi_full).fetch()}
    assert "cpi_yoy_pct" in by_key
    assert "iip_yoy_pct" not in by_key
    assert "macro_stance" not in by_key


def test_iip_cpi_confidence_drops_when_stale() -> None:
    """With anchor months 6mo old, both IIP and CPI confidence must drop."""
    today = dt.datetime.now(dt.timezone.utc)
    iip_old, cpi_old = [], []
    for offset in range(13, 0, -1):
        anchor = today - dt.timedelta(days=offset * 30 + 180)  # 6 months back
        iip_old.append({"month": f"{anchor.year}-{anchor.month:02d}", "general_index": 100})
        cpi_old.append({"month": f"{anchor.year}-{anchor.month:02d}", "yoy_pct": 4.5})
    by_key = {s.key: s for s in IipCpiSource(iip_old, cpi_old).fetch()}
    assert by_key["iip_yoy_pct"].confidence < 0.5
    assert by_key["cpi_yoy_pct"].confidence < 0.5


def test_iip_cpi_default_tables_chronological() -> None:
    assert [m["month"] for m in IIP_HISTORY] == sorted(m["month"] for m in IIP_HISTORY)
    assert [m["month"] for m in CPI_HISTORY] == sorted(m["month"] for m in CPI_HISTORY)


def test_iip_cpi_singleton() -> None:
    assert get_iip_cpi_source() is get_iip_cpi_source()
