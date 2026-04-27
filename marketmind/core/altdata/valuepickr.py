"""
ValuePickr forum activity alt-data source.

ValuePickr (forum.valuepickr.com) runs Discourse, which exposes
``/latest.json`` and ``/top/weekly.json`` as a public read-only API. We
sample weekly thread velocity to gauge what the long-form Indian retail
investor community is actively researching.

Signals emitted:
    weekly_thread_count        # active threads in /top/weekly
    weekly_total_replies       # summed reply_count across those threads
    weekly_total_views         # summed views across those threads
    top_thread_<TICKER>        # ticker mention count in titles

Why /top/weekly rather than /latest: ``/latest`` is dominated by replies
to long-running mega-threads. ``/top/weekly`` ranks by activity in the
window, which is closer to the signal we want.

Failure model: any HTTP/parse error returns ``[]`` (via ``@safe_fetch``).
Discourse occasionally 503s during deploys — that drops one cycle, no big deal.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

import requests

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch
from marketmind.core.altdata.reddit_sentiment import (
    TICKER_STOP_WORDS,
    TIER1_TICKERS,
)

logger = logging.getLogger(__name__)

TICKER_TOKEN_RX = re.compile(r"\b[A-Z][A-Z0-9&-]{2,11}\b")


class ValuePickrSource(AltDataSource):
    SLUG = "valuepickr"
    USER_AGENT = "marketmind/1.0 (alt-data ingestion; +https://localhost)"
    BASE = "https://forum.valuepickr.com"
    TOP_PATH = "/top/weekly.json"

    def __init__(
        self,
        tickers: Iterable[str] = TIER1_TICKERS,
        timeout_s: float = 8.0,
        max_threads: int = 50,
    ) -> None:
        self.tickers = frozenset(t.upper() for t in tickers)
        self.timeout_s = timeout_s
        self.max_threads = max_threads

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        topics = self._fetch_topics()
        if not topics:
            return []
        return self._summarise(topics[: self.max_threads])

    # ── HTTP ─────────────────────────────────────────────────────────────
    def _fetch_topics(self) -> List[Dict[str, Any]]:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",  # NEVER add 'br'
        })
        try:
            r = sess.get(f"{self.BASE}{self.TOP_PATH}", timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.info("valuepickr fetch error: %s", e)
            return []
        topic_list = (data.get("topic_list") or {}).get("topics") or []
        return [t for t in topic_list if isinstance(t, dict)]

    # ── Aggregation ──────────────────────────────────────────────────────
    def _summarise(self, topics: List[Dict[str, Any]]) -> List[AltSignal]:
        as_of = _dt.datetime.now(_dt.timezone.utc)
        n = len(topics)
        replies = sum(int(t.get("reply_count") or 0) for t in topics)
        views = sum(int(t.get("views") or 0) for t in topics)

        signals: List[AltSignal] = [
            AltSignal(
                source=self.SLUG, key="weekly_thread_count", value=n,
                unit="count", as_of=as_of, confidence=0.9, raw={},
            ),
            AltSignal(
                source=self.SLUG, key="weekly_total_replies", value=replies,
                unit="count", as_of=as_of, confidence=0.85, raw={},
            ),
            AltSignal(
                source=self.SLUG, key="weekly_total_views", value=views,
                unit="count", as_of=as_of, confidence=0.85, raw={},
            ),
        ]

        ticker_counts = self._count_ticker_mentions(topics)
        for sym, count in sorted(
            ticker_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:10]:
            signals.append(AltSignal(
                source=self.SLUG, key=f"top_thread_{sym}", value=count,
                unit="mentions", as_of=as_of, confidence=0.75, raw={},
            ))
        return signals

    def _count_ticker_mentions(self, topics: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in topics:
            title = str(t.get("title") or "").upper()
            for tok in TICKER_TOKEN_RX.findall(title):
                if tok in TICKER_STOP_WORDS:
                    continue
                if tok in self.tickers:
                    counts[tok] = counts.get(tok, 0) + 1
        return counts


_singleton: Optional[ValuePickrSource] = None


def get_valuepickr_source() -> ValuePickrSource:
    global _singleton
    if _singleton is None:
        _singleton = ValuePickrSource()
    return _singleton
