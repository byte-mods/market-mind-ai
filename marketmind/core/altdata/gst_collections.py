"""
GST monthly collections alt-data source.

The Ministry of Finance / PIB releases gross GST collection figures on the
1st of each month for the previous month. We mirror this in a maintained
table (RBI-rate-style) since live PIB scraping is brittle (the press
release URL pattern changes per release).

Signals emitted:
    gst_yoy_pct                  # latest month gross collection vs same month last year
    gst_3m_avg_inr_cr            # trailing 3-month gross collection average
    gst_staleness_days           # days since the table's latest month

Confidence drops below 0.5 when staleness > 35 days (a release was likely
published that we haven't ingested).
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional, Sequence

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch

logger = logging.getLogger(__name__)


# Maintained gross-GST table — units: INR crore.
# Update from PIB press releases monthly. Schema-locked.
GST_COLLECTIONS_HISTORY: tuple[Dict[str, Any], ...] = (
    {"month": "2025-03", "gross_inr_cr": 196000},
    {"month": "2025-04", "gross_inr_cr": 210000},   # April peak (FY closing)
    {"month": "2025-05", "gross_inr_cr": 173000},
    {"month": "2025-06", "gross_inr_cr": 174000},
    {"month": "2025-07", "gross_inr_cr": 182000},
    {"month": "2025-08", "gross_inr_cr": 175000},
    {"month": "2025-09", "gross_inr_cr": 173000},
    {"month": "2025-10", "gross_inr_cr": 188000},   # festive
    {"month": "2025-11", "gross_inr_cr": 184000},
    {"month": "2025-12", "gross_inr_cr": 178000},
    {"month": "2026-01", "gross_inr_cr": 196000},
    {"month": "2026-02", "gross_inr_cr": 191000},
    {"month": "2026-03", "gross_inr_cr": 215000},
)


class GstCollectionsSource(AltDataSource):
    SLUG = "gst"

    def __init__(
        self, table: Sequence[Dict[str, Any]] = GST_COLLECTIONS_HISTORY
    ) -> None:
        self.table = list(table)

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        if len(self.table) < 13:
            logger.info("GST table needs ≥13 months for YoY; have %d", len(self.table))
            return []

        latest = self.table[-1]
        same_month_last_year = self.table[-13]
        as_of = _dt.datetime.now(_dt.timezone.utc)
        staleness_days = _staleness_days(latest["month"], as_of)
        confidence = 0.85 if staleness_days <= 35 else 0.4

        prev = float(same_month_last_year.get("gross_inr_cr") or 0)
        curr = float(latest.get("gross_inr_cr") or 0)
        yoy = round((curr / prev - 1) * 100, 2) if prev else 0.0

        last_3 = [float(m.get("gross_inr_cr") or 0) for m in self.table[-3:]]
        avg_3m = round(sum(last_3) / 3.0, 0) if last_3 else 0.0

        return [
            AltSignal(
                source=self.SLUG, key="gst_yoy_pct", value=yoy, unit="pct",
                as_of=as_of, confidence=confidence,
                raw={"latest_month": latest["month"], "curr_inr_cr": curr,
                     "prev_inr_cr": prev},
            ),
            AltSignal(
                source=self.SLUG, key="gst_3m_avg_inr_cr", value=avg_3m,
                unit="inr_cr", as_of=as_of, confidence=confidence,
                raw={"window": [m["month"] for m in self.table[-3:]]},
            ),
            AltSignal(
                source=self.SLUG, key="gst_staleness_days", value=staleness_days,
                unit="days", as_of=as_of, confidence=1.0,
                raw={"latest_month": latest["month"]},
            ),
        ]


def _staleness_days(month_str: str, now: _dt.datetime) -> int:
    """GST is published on day 1 of the next month, so anchor staleness on
    the *first day of the month after* the latest data row."""
    year, month = int(month_str[:4]), int(month_str[5:7])
    if month == 12:
        anchor = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
    else:
        anchor = _dt.datetime(year, month + 1, 1, tzinfo=_dt.timezone.utc)
    return max(0, (now - anchor).days)


_singleton: Optional[GstCollectionsSource] = None


def get_gst_source() -> GstCollectionsSource:
    global _singleton
    if _singleton is None:
        _singleton = GstCollectionsSource()
    return _singleton
