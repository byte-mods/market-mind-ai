"""W5.3 T2 verification: insider-window pure compute over NSE announcements."""
from __future__ import annotations

import datetime as _dt

import pytest

from marketmind.compliance.insider_window import (
    WINDOW_REOPEN_DAYS_AFTER_RESULTS,
    InsiderWindowStatus,
    compute_insider_window,
)


def _ann(
    date_str: str,
    *,
    category: str = "results",
    desc: str = "Quarterly Financial Results",
) -> dict:
    """Convenience: build a minimal announcement row."""
    return {"date": date_str, "category": category, "desc": desc}


# ─── Empty-input contract ────────────────────────────────────────────────


def test_insider_window_open_when_no_announcements() -> None:
    status = compute_insider_window("RELIANCE", [], _dt.date(2026, 4, 28))
    assert status.is_open is True
    assert status.last_results_date is None
    assert status.closed_until is None
    assert "no announcements" in status.reason


# ─── Closed window: results within 48h ───────────────────────────────────


def test_insider_window_closed_within_48h_of_results() -> None:
    status = compute_insider_window(
        "RELIANCE",
        [_ann("2026-04-27")],
        _dt.date(2026, 4, 28),  # 1 day after
    )
    assert status.is_open is False
    assert status.last_results_date == _dt.date(2026, 4, 27)
    assert status.closed_until == _dt.date(
        2026, 4, 27
    ) + _dt.timedelta(days=WINDOW_REOPEN_DAYS_AFTER_RESULTS)
    assert "results declared 2026-04-27" in status.reason


# ─── Open window: > 48h after results, before next quarter-end ──────────


def test_insider_window_opens_after_48h() -> None:
    # Results declared Jan 25, today Feb 10 → past Jan 27 reopen, before Mar 31 close.
    status = compute_insider_window(
        "TCS",
        [_ann("2026-01-25")],
        _dt.date(2026, 2, 10),
    )
    assert status.is_open is True
    assert status.last_results_date == _dt.date(2026, 1, 25)
    assert status.closed_until is None


# ─── Closed: quarter ended, results not yet declared ─────────────────────


def test_insider_window_closed_when_awaiting_current_quarter_results() -> None:
    # Today April 28; last quarter ended Mar 31; only result on file is for Q3
    # (declared Jan 25), which is now older than q_end. → CLOSED awaiting.
    status = compute_insider_window(
        "INFY",
        [_ann("2026-01-25")],
        _dt.date(2026, 4, 28),
    )
    assert status.is_open is False
    assert status.last_results_date == _dt.date(2026, 1, 25)
    assert status.closed_until is None
    assert "awaiting results for quarter ending 2026-03-31" in status.reason


# ─── Most-recent results wins when multiple present ──────────────────────


def test_insider_window_picks_latest_results_when_multiple() -> None:
    status = compute_insider_window(
        "WIPRO",
        [
            _ann("2025-10-20"),
            _ann("2026-01-25"),
            _ann("2025-07-15"),
        ],
        _dt.date(2026, 1, 26),
    )
    assert status.last_results_date == _dt.date(2026, 1, 25)
    assert status.is_open is False  # within 2-day reopen window


# ─── Non-results announcements ignored ───────────────────────────────────


def test_insider_window_ignores_non_results_announcements() -> None:
    rows = [
        {"date": "2026-04-27", "category": "dividend", "desc": "Final Dividend"},
        {"date": "2026-04-26", "category": "insider", "desc": "Insider Trade"},
        {"date": "2026-04-25", "category": "governance", "desc": "Board Meeting"},
    ]
    status = compute_insider_window("HDFCBANK", rows, _dt.date(2026, 4, 28))
    # No results → CLOSED with "awaiting" reason (last_results is None < q_end).
    assert status.is_open is False
    assert status.last_results_date is None


# ─── Title-keyword fallback (when category missing) ──────────────────────


def test_insider_window_keyword_match_when_category_missing() -> None:
    rows = [
        {"date": "2026-01-25", "desc": "Audited Financial Results for Q3 FY26"},
    ]
    status = compute_insider_window("ICICIBANK", rows, _dt.date(2026, 2, 1))
    assert status.last_results_date == _dt.date(2026, 1, 25)
    assert status.is_open is True


# ─── Malformed rows skipped ──────────────────────────────────────────────


def test_insider_window_skips_malformed_rows() -> None:
    rows = [
        "garbage string",  # not a dict
        {"category": "results"},  # no date
        {"date": "not-a-date", "category": "results", "desc": "Quarterly Results"},
        _ann("2026-01-25"),  # one good row
    ]
    status = compute_insider_window("SBIN", rows, _dt.date(2026, 2, 1))
    assert status.last_results_date == _dt.date(2026, 1, 25)


# ─── Symbol normalisation ────────────────────────────────────────────────


def test_insider_window_normalises_symbol_uppercase() -> None:
    status = compute_insider_window(
        "  reliance  ",
        [_ann("2026-01-25")],
        _dt.date(2026, 2, 10),
    )
    assert status.symbol == "RELIANCE"


def test_insider_window_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_insider_window("", [], _dt.date(2026, 4, 28))


# ─── today: datetime accepted ────────────────────────────────────────────


def test_insider_window_accepts_datetime_today() -> None:
    status = compute_insider_window(
        "RELIANCE",
        [_ann("2026-04-27")],
        _dt.datetime(2026, 4, 28, 14, 30, 0),
    )
    assert status.is_open is False  # within 48h


def test_insider_window_rejects_non_date_today() -> None:
    with pytest.raises(ValueError, match="today must be"):
        compute_insider_window("RELIANCE", [], "2026-04-28")  # type: ignore[arg-type]


# ─── Quarter-end edge cases ──────────────────────────────────────────────


def test_insider_window_quarter_end_boundary_inclusive() -> None:
    # On Mar 31 itself: q_end = Mar 31, today = Mar 31, not yet "after end of quarter"
    # but the rule says "from the end of every quarter" — we treat q_end as the
    # last open day, so a stale-results scenario on Mar 31 still maps to that q_end.
    # Last results from Jan 25 (Q3) is < Mar 31 → CLOSED awaiting.
    status = compute_insider_window(
        "RELIANCE",
        [_ann("2026-01-25")],
        _dt.date(2026, 3, 31),
    )
    assert status.is_open is False
    assert "awaiting results for quarter ending 2026-03-31" in status.reason


def test_insider_window_results_exactly_on_reopen_boundary() -> None:
    # Results 2026-04-27, reopen = 2026-04-29. Today = 2026-04-29 → still CLOSED
    # (boundary inclusive), today = 2026-04-30 → OPEN.
    boundary = compute_insider_window(
        "RELIANCE", [_ann("2026-04-27")], _dt.date(2026, 4, 29)
    )
    assert boundary.is_open is False
    next_day = compute_insider_window(
        "RELIANCE", [_ann("2026-04-27")], _dt.date(2026, 4, 30)
    )
    assert next_day.is_open is True


# ─── Status frozen ───────────────────────────────────────────────────────


def test_insider_window_status_is_frozen() -> None:
    status = compute_insider_window("RELIANCE", [], _dt.date(2026, 4, 28))
    with pytest.raises((AttributeError, Exception)):
        status.is_open = True  # type: ignore[misc]
    assert isinstance(status, InsiderWindowStatus)


# ─── IST-vs-UTC boundary pins ────────────────────────────────────────────
# These tests pin the current UTC-date behavior of the insider-window
# compute. The Indian-market system runs on IST (UTC+5:30), so for the
# 5.5-hour window each evening UTC the IST calendar date is one day
# ahead of the UTC date. The snapshot's "Open invariants for next
# section" called this out as a deferred IST conversion. Locking the
# current behavior here means the eventual IST conversion is a
# deliberate, test-visible flip — not a silent regression.


def test_insider_window_quarter_end_day_inclusive() -> None:
    """Today == quarter-end day → that quarter-end is the most-recent
    boundary; with no results yet, window is CLOSED awaiting results."""
    # 2026-06-30 is a Q2 end. No results announcements yet.
    status = compute_insider_window("RELIANCE", [], _dt.date(2026, 6, 30))
    # Empty announcements → "no announcements available" path (open with warn).
    # Provide a non-results announcement to exercise the q_end branch:
    other = compute_insider_window(
        "RELIANCE",
        [{"date": "2026-06-15", "category": "other", "desc": "Press Release"}],
        _dt.date(2026, 6, 30),
    )
    assert other.is_open is False
    assert "awaiting results for quarter ending 2026-06-30" in other.reason
    _ = status  # empty-list path covered elsewhere


def test_insider_window_quarter_end_plus_one_day_falls_back_to_qe() -> None:
    """Today = quarter-end + 1 day → most-recent quarter-end is still
    the same calendar quarter end. Pins that crossing midnight UTC into
    the next calendar day does not advance to the *next* quarter end."""
    # July 1 is one day past the June 30 quarter end. Window remains
    # awaiting Q2 results.
    status = compute_insider_window(
        "RELIANCE",
        [{"date": "2026-06-15", "category": "other", "desc": "Press Release"}],
        _dt.date(2026, 7, 1),
    )
    assert status.is_open is False
    assert "2026-06-30" in status.reason  # NOT 2026-09-30


def test_insider_window_reopen_day_ist_vs_utc_pin() -> None:
    """Same wall-clock instant produces different decisions depending on
    whether ``today`` is derived from the UTC date or the IST date.

    Concrete instant: 2026-05-01T21:30:00Z
      UTC date  = 2026-05-01
      IST date  = 2026-05-02 (UTC+5:30 → 03:00 IST next day)

    With results declared 2026-04-29 and reopen = 2026-05-01:
      compute(today=UTC date 2026-05-01) → CLOSED (today <= reopen)
      compute(today=IST date 2026-05-02) → OPEN  (today >  reopen)

    Current callers (AppController.compliance_pretrade_check and
    compliance_get_insider_window) pass ``datetime.now(timezone.utc).date()``,
    which yields the UTC variant. This test pins both behaviors. When
    the deferred IST conversion lands, the second assertion will not
    change but the controller integration test should flip — making the
    semantic change auditable in a single diff."""
    instant_utc = _dt.datetime(2026, 5, 1, 21, 30, tzinfo=_dt.timezone.utc)
    today_utc = instant_utc.astimezone(_dt.timezone.utc).date()
    today_ist = instant_utc.astimezone(
        _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    ).date()
    assert today_utc == _dt.date(2026, 5, 1)
    assert today_ist == _dt.date(2026, 5, 2)

    announcements = [{"date": "2026-04-29", "category": "results"}]

    # Current production path (UTC) → CLOSED at the reopen-day boundary.
    via_utc = compute_insider_window("RELIANCE", announcements, today_utc)
    assert via_utc.is_open is False
    assert via_utc.closed_until == _dt.date(2026, 5, 1)

    # Hypothetical IST path → OPEN one calendar day earlier.
    via_ist = compute_insider_window("RELIANCE", announcements, today_ist)
    assert via_ist.is_open is True
    assert via_ist.last_results_date == _dt.date(2026, 4, 29)
