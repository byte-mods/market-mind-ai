"""Insider-trading window enforcement (W5.3 T2).

Pure compute over pre-fetched NSE corporate-announcements. The fetcher
(``marketmind.core.filings_ingest.FilingsIngester.fetch_announcements``)
already hits NSE through the warmed session; we just classify what came
back and decide whether the SEBI trading window is open for a given
designated-person symbol.

SEBI (Prohibition of Insider Trading) Regulations 2015 Reg 9(B), Code of
Conduct for designated persons:
    "The trading window shall be closed from the end of every quarter
     till 48 hours after the declaration of financial results."

Approximation implemented here:
    - Indian quarter-ends: 31-Mar, 30-Jun, 30-Sep, 31-Dec.
    - For each symbol, find the most recent results announcement.
    - Find the most recent quarter-end on or before ``today``: ``q_end``.
    - If no results announcement exists, OR the latest results predates
      ``q_end`` → window is CLOSED (results pending for current quarter).
    - Else the window for that quarter ran [q_end+1, results_date + 2 days].
      Today inside that range → CLOSED. Today after → OPEN.

Caller is responsible for the designated-symbol gate: this module always
computes a status; ``pretrade_check`` decides whether to enforce it.

Why pre-fetched: the NSE fetcher has its own session/TTL; keeping this
module free of I/O makes it deterministic, fast, and trivially testable.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Window opens 48h (2 days) after results declaration.
WINDOW_REOPEN_DAYS_AFTER_RESULTS = 2

# NSE corporate-announcements category strings that indicate quarterly
# financial results. ``filings_ingest._classify`` already labels these but
# we also pattern-match the raw `desc`/`subject` string defensively.
_RESULTS_KEYWORDS = (
    "financial result",
    "quarterly result",
    "audited result",
    "unaudited result",
)
_RESULTS_CATEGORY = "results"

# Calendar quarter ends (month, day). Independent of fiscal year.
_QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))


@dataclass(frozen=True)
class InsiderWindowStatus:
    """Per-symbol trading-window status as of a given date."""

    symbol: str
    is_open: bool
    closed_until: Optional[_dt.date]
    last_results_date: Optional[_dt.date]
    reason: str


def _coerce_to_date(value: Any) -> Optional[_dt.date]:
    """Return a ``date`` from ``date``/``datetime``/ISO string, else None."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str) and value:
        # NSE announcements use various formats; ``filings_ingest._normalise_date``
        # outputs ISO YYYY-MM-DD into the `date` metadata. Be tolerant either way.
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
            try:
                return _dt.datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return None


def _is_results_announcement(row: Dict[str, Any]) -> bool:
    cat = (row.get("category") or "").lower()
    if cat == _RESULTS_CATEGORY:
        return True
    blob = " ".join(
        str(row.get(k) or "") for k in ("desc", "subject", "title", "an_dt_subject")
    ).lower()
    return any(kw in blob for kw in _RESULTS_KEYWORDS)


def _latest_results_date(announcements: List[Dict[str, Any]]) -> Optional[_dt.date]:
    """Most recent results-announcement date in the input list, or None."""
    dates: List[_dt.date] = []
    for row in announcements:
        if not isinstance(row, dict):
            continue
        if not _is_results_announcement(row):
            continue
        # Try several fields — NSE shifts these around between feed versions.
        for field in ("date", "sort_date", "an_dt", "broadcastdate"):
            d = _coerce_to_date(row.get(field))
            if d is not None:
                dates.append(d)
                break
    if not dates:
        return None
    return max(dates)


def _most_recent_quarter_end(today: _dt.date) -> _dt.date:
    """The latest calendar quarter-end on or before ``today``."""
    candidates: List[_dt.date] = []
    for year in (today.year, today.year - 1):
        for month, day in _QUARTER_ENDS:
            try:
                d = _dt.date(year, month, day)
            except ValueError:
                continue
            if d <= today:
                candidates.append(d)
    return max(candidates)


def compute_insider_window(
    symbol: str,
    announcements: List[Dict[str, Any]],
    today: _dt.date,
) -> InsiderWindowStatus:
    """Decide whether ``symbol``'s SEBI trading window is open as of ``today``.

    ``announcements`` is a list of NSE corporate-announcement dicts (the
    raw shape returned by ``FilingsIngester.fetch_announcements``, or the
    enriched shape with ``category`` added). Empty list → assume open
    with a "no announcements available" reason (caller should surface as
    a warning, not silently allow).

    ``today`` must be a ``date`` (or ``datetime`` — its date component is
    used). Tz-naive is fine; quarter-ends are calendar days.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if isinstance(today, _dt.datetime):
        today = today.date()
    if not isinstance(today, _dt.date):
        raise ValueError("today must be a date or datetime")

    sym_norm = symbol.upper().strip()

    if not announcements:
        return InsiderWindowStatus(
            symbol=sym_norm,
            is_open=True,
            closed_until=None,
            last_results_date=None,
            reason="no announcements available; cannot enforce window",
        )

    last_results = _latest_results_date(announcements)
    q_end = _most_recent_quarter_end(today)

    if last_results is None or last_results < q_end:
        return InsiderWindowStatus(
            symbol=sym_norm,
            is_open=False,
            closed_until=None,  # unknown — depends on when results land
            last_results_date=last_results,
            reason=f"awaiting results for quarter ending {q_end.isoformat()}",
        )

    reopen_date = last_results + _dt.timedelta(days=WINDOW_REOPEN_DAYS_AFTER_RESULTS)
    if today <= reopen_date:
        return InsiderWindowStatus(
            symbol=sym_norm,
            is_open=False,
            closed_until=reopen_date,
            last_results_date=last_results,
            reason=(
                f"closed until {reopen_date.isoformat()} "
                f"(results declared {last_results.isoformat()})"
            ),
        )
    return InsiderWindowStatus(
        symbol=sym_norm,
        is_open=True,
        closed_until=None,
        last_results_date=last_results,
        reason=(
            f"open since {(reopen_date + _dt.timedelta(days=1)).isoformat()} "
            f"(results declared {last_results.isoformat()})"
        ),
    )
