"""
Google Trends ticker-interest source — OPTIONAL dependency.

Why optional: ``pytrends`` is brittle. Google rate-limits it aggressively,
the upstream maintainers have shipped breaking changes mid-version, and
the library hasn't been updated for newer Google response shapes. We
want it as a *bonus* signal, never a load-bearing one.

Behaviour:
- If ``pytrends`` is not installed → log once at INFO, return ``[]``.
- If ``pytrends`` is installed but the live call raises (429, parse
  error, etc.) → ``@safe_fetch`` catches it, returns ``[]``.
- Otherwise emit one signal per queried ticker:
      ``trend_<TICKER>_score`` value=mean 7-day interest, 0–100, confidence=0.6

Confidence is fixed at 0.6 because Google Trends interest values are
relative within a query batch — they don't have an absolute meaning.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Iterable, List, Optional

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch

logger = logging.getLogger(__name__)

# Default query batch — Google Trends caps batches at 5 keywords per request.
DEFAULT_BATCH: tuple[str, ...] = ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK")


class GoogleTrendsSource(AltDataSource):
    SLUG = "trends"
    GEO = "IN"
    TIMEFRAME = "now 7-d"

    def __init__(
        self,
        tickers: Iterable[str] = DEFAULT_BATCH,
        pytrends_factory: Optional[Any] = None,
    ) -> None:
        # Cap at 5 — Google's hard limit per query.
        self.tickers = tuple(t.upper() for t in tickers)[:5]
        # Tests inject a fake factory; production uses real pytrends.
        self._factory = pytrends_factory

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        client = self._build_client()
        if client is None:
            return []
        try:
            client.build_payload(
                kw_list=list(self.tickers),
                timeframe=self.TIMEFRAME,
                geo=self.GEO,
            )
            df = client.interest_over_time()
        except Exception as e:  # noqa: BLE001 — broad: pytrends raises a zoo
            logger.info("google trends call failed: %s", e)
            return []

        if df is None or len(df) == 0:
            return []

        as_of = _dt.datetime.now(_dt.timezone.utc)
        signals: List[AltSignal] = []
        for sym in self.tickers:
            if sym not in df.columns:
                continue
            try:
                mean_score = float(df[sym].mean())
            except Exception:
                continue
            if mean_score != mean_score:  # NaN guard
                continue
            signals.append(AltSignal(
                source=self.SLUG, key=f"trend_{sym}_score",
                value=round(mean_score, 1), unit="score_0_100",
                as_of=as_of, confidence=0.6,
                raw={"timeframe": self.TIMEFRAME, "geo": self.GEO},
            ))
        return signals

    def _build_client(self) -> Optional[Any]:
        if self._factory is not None:
            return self._factory()
        try:
            from pytrends.request import TrendReq  # type: ignore
        except ImportError:
            logger.info(
                "pytrends not installed — Google Trends source disabled. "
                "Install with `pip install pytrends` to enable."
            )
            return None
        try:
            return TrendReq(hl="en-IN", tz=330)  # IST = UTC+5:30 = 330 minutes
        except Exception as e:  # noqa: BLE001
            logger.info("pytrends init failed: %s", e)
            return None


_singleton: Optional[GoogleTrendsSource] = None


def get_trends_source() -> GoogleTrendsSource:
    global _singleton
    if _singleton is None:
        _singleton = GoogleTrendsSource()
    return _singleton
