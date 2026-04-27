"""
IIP (Index of Industrial Production) + CPI (Consumer Price Index) source.

Both indices come from MOSPI (Ministry of Statistics) and the RBI bulletin.
Maintained-table pattern — same rationale as GST and SIAM.

Signals emitted:
    iip_yoy_pct                # latest IIP general index vs same month last year
    iip_3m_avg                 # trailing 3-month IIP general-index average
    cpi_yoy_pct                # latest published CPI inflation %
    cpi_3m_avg_pct             # trailing 3-month CPI average
    macro_stance               # categorical: Stable / Inflationary / Disinflationary / Stagflation
    iip_staleness_days
    cpi_staleness_days

The ``macro_stance`` derivation is intentionally simple — two-axis quadrant
on the latest IIP-YoY × CPI readings. It is a coarse signal, not a regime
classifier (W1.2 already covers regime). It exists so the dashboard can show
a one-word macro tilt next to the indices.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional, Sequence

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch

logger = logging.getLogger(__name__)


# Maintained tables. Update from RBI bulletin / MOSPI press releases.
# IIP: General Index value (base 2011-12 = 100).
# CPI: Combined-CPI YoY inflation %, headline.
IIP_HISTORY: tuple[Dict[str, Any], ...] = (
    {"month": "2025-02", "general_index": 145.6},
    {"month": "2025-03", "general_index": 158.2},   # March is seasonally high
    {"month": "2025-04", "general_index": 144.1},
    {"month": "2025-05", "general_index": 148.5},
    {"month": "2025-06", "general_index": 144.8},
    {"month": "2025-07", "general_index": 146.9},
    {"month": "2025-08", "general_index": 144.0},
    {"month": "2025-09", "general_index": 146.2},
    {"month": "2025-10", "general_index": 152.5},
    {"month": "2025-11", "general_index": 149.0},
    {"month": "2025-12", "general_index": 154.3},
    {"month": "2026-01", "general_index": 151.2},
    {"month": "2026-02", "general_index": 152.8},
)

CPI_HISTORY: tuple[Dict[str, Any], ...] = (
    {"month": "2025-02", "yoy_pct": 5.09},
    {"month": "2025-03", "yoy_pct": 4.85},
    {"month": "2025-04", "yoy_pct": 4.83},
    {"month": "2025-05", "yoy_pct": 4.75},
    {"month": "2025-06", "yoy_pct": 5.08},
    {"month": "2025-07", "yoy_pct": 3.54},   # base effect dip
    {"month": "2025-08", "yoy_pct": 3.65},
    {"month": "2025-09", "yoy_pct": 5.49},
    {"month": "2025-10", "yoy_pct": 6.21},   # food spike
    {"month": "2025-11", "yoy_pct": 5.48},
    {"month": "2025-12", "yoy_pct": 5.22},
    {"month": "2026-01", "yoy_pct": 4.31},
    {"month": "2026-02", "yoy_pct": 3.61},
    {"month": "2026-03", "yoy_pct": 3.34},
)


class IipCpiSource(AltDataSource):
    SLUG = "iip_cpi"

    # Stance thresholds. Tuned for Indian context (RBI 4±2% CPI band).
    HIGH_CPI: float = 5.0
    LOW_CPI:  float = 4.0
    HIGH_IIP_YOY: float = 4.0
    LOW_IIP_YOY:  float = 1.0

    def __init__(
        self,
        iip_table: Sequence[Dict[str, Any]] = IIP_HISTORY,
        cpi_table: Sequence[Dict[str, Any]] = CPI_HISTORY,
    ) -> None:
        self.iip_table = list(iip_table)
        self.cpi_table = list(cpi_table)

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        as_of = _dt.datetime.now(_dt.timezone.utc)
        signals: List[AltSignal] = []

        iip_yoy = self._iip_yoy()
        if iip_yoy is not None:
            iip_latest_month = self.iip_table[-1]["month"]
            iip_stale = _staleness_days(iip_latest_month, as_of, lag_days=45)
            iip_conf = 0.85 if iip_stale <= 60 else 0.4
            signals.append(AltSignal(
                source=self.SLUG, key="iip_yoy_pct", value=iip_yoy, unit="pct",
                as_of=as_of, confidence=iip_conf,
                raw={"latest_month": iip_latest_month},
            ))
            signals.append(AltSignal(
                source=self.SLUG, key="iip_3m_avg",
                value=round(_avg(self.iip_table[-3:], "general_index"), 2),
                unit="index", as_of=as_of, confidence=iip_conf, raw={},
            ))
            signals.append(AltSignal(
                source=self.SLUG, key="iip_staleness_days", value=iip_stale,
                unit="days", as_of=as_of, confidence=1.0,
                raw={"latest_month": iip_latest_month},
            ))

        cpi_latest = self.cpi_table[-1] if self.cpi_table else None
        cpi_yoy: Optional[float] = None
        if cpi_latest is not None:
            cpi_yoy = float(cpi_latest.get("yoy_pct") or 0)
            cpi_stale = _staleness_days(cpi_latest["month"], as_of, lag_days=15)
            cpi_conf = 0.9 if cpi_stale <= 30 else 0.4
            signals.append(AltSignal(
                source=self.SLUG, key="cpi_yoy_pct", value=cpi_yoy, unit="pct",
                as_of=as_of, confidence=cpi_conf,
                raw={"latest_month": cpi_latest["month"]},
            ))
            signals.append(AltSignal(
                source=self.SLUG, key="cpi_3m_avg_pct",
                value=round(_avg(self.cpi_table[-3:], "yoy_pct"), 2),
                unit="pct", as_of=as_of, confidence=cpi_conf, raw={},
            ))
            signals.append(AltSignal(
                source=self.SLUG, key="cpi_staleness_days", value=cpi_stale,
                unit="days", as_of=as_of, confidence=1.0,
                raw={"latest_month": cpi_latest["month"]},
            ))

        if iip_yoy is not None and cpi_yoy is not None:
            stance = self._derive_stance(iip_yoy, cpi_yoy)
            signals.append(AltSignal(
                source=self.SLUG, key="macro_stance", value=stance, unit="label",
                as_of=as_of, confidence=0.7,
                raw={"iip_yoy_pct": iip_yoy, "cpi_yoy_pct": cpi_yoy},
            ))
        return signals

    # ── Internal helpers ─────────────────────────────────────────────────
    def _iip_yoy(self) -> Optional[float]:
        if len(self.iip_table) < 13:
            return None
        latest = float(self.iip_table[-1].get("general_index") or 0)
        prev = float(self.iip_table[-13].get("general_index") or 0)
        if not prev:
            return None
        return round((latest / prev - 1) * 100, 2)

    def _derive_stance(self, iip_yoy: float, cpi_yoy: float) -> str:
        if cpi_yoy >= self.HIGH_CPI and iip_yoy < self.LOW_IIP_YOY:
            return "Stagflation"
        if cpi_yoy >= self.HIGH_CPI and iip_yoy >= self.HIGH_IIP_YOY:
            return "Inflationary"
        if cpi_yoy < self.LOW_CPI and iip_yoy >= self.HIGH_IIP_YOY:
            return "Disinflationary"
        return "Stable"


def _avg(rows: Sequence[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    vals = [float(r.get(key) or 0) for r in rows]
    return sum(vals) / len(vals)


def _staleness_days(month_str: str, now: _dt.datetime, lag_days: int) -> int:
    """Anchor staleness on (last day of month + lag_days), since IIP and
    CPI publish with a structural lag (CPI ~12 days, IIP ~45 days)."""
    year, month = int(month_str[:4]), int(month_str[5:7])
    if month == 12:
        next_month = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
    else:
        next_month = _dt.datetime(year, month + 1, 1, tzinfo=_dt.timezone.utc)
    anchor = next_month + _dt.timedelta(days=lag_days)
    return max(0, (now - anchor).days)


_singleton: Optional[IipCpiSource] = None


def get_iip_cpi_source() -> IipCpiSource:
    global _singleton
    if _singleton is None:
        _singleton = IipCpiSource()
    return _singleton
