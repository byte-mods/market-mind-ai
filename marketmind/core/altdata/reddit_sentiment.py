"""
Reddit retail-sentiment alt-data source.

Polls the public ``top.json?t=week`` feed for r/IndianStockMarket and
r/IndiaInvestments — no auth, no PRAW. Emits four signal kinds:

    sentiment_tilt          # (bullish - bearish) / total_posts, ∈ [-1, 1]
    weekly_post_count       # raw count of top-100 posts that came in
    weekly_upvotes          # summed upvotes across those posts
    top_ticker_<SYMBOL>     # ticker mention count (curated tier-1 list)

Why curated tier-1 instead of full Nifty 500: a generic ``\\b[A-Z]{3,12}\\b``
regex matches noise (BUY, HOLD, DCF, NIFTY, GST) far more often than tickers.
A 50-symbol allow-list plus a stop-set keeps the signal usable. To extend,
pass ``tickers=`` to the constructor.

Failure model: any HTTP error or parse failure returns ``[]`` (via
``@safe_fetch``). 429 is treated specially — we still return ``[]`` but log
at INFO rather than WARNING since rate-limiting is expected on a public feed.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from typing import Dict, Iterable, List, Optional

import requests

from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch

logger = logging.getLogger(__name__)


# Curated tier-1 ticker list. Extend by passing tickers= to constructor.
TIER1_TICKERS: tuple[str, ...] = (
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "BAJFINANCE", "BAJAJFINSV", "HINDUNILVR", "ITC", "NESTLEIND",
    "ASIANPAINT", "TITAN", "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO",
    "EICHERMOT", "HEROMOTOCO", "BHARTIARTL", "WIPRO", "TECHM", "HCLTECH",
    "LT", "ULTRACEMCO", "GRASIM", "JSWSTEEL", "TATASTEEL", "HINDALCO",
    "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "NTPC", "POWERGRID",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "BRITANNIA",
    "BPCL", "IOC", "DMART", "ZOMATO", "NYKAA", "PAYTM", "IRCTC",
)

# Words that *look* like tickers but aren't — keeps signal clean even within
# the tier-1 allow-list (e.g. someone writing "ITC GST IOC NYKAA").
TICKER_STOP_WORDS = {
    "GST", "RBI", "SEBI", "NSE", "BSE", "NIFTY", "SENSEX", "USA", "USD", "INR",
    "EPS", "ROE", "ROI", "DCF", "BUY", "HOLD", "SELL", "PSU", "FII", "DII",
    "ETF", "IPO", "FNO", "PCR", "MTF", "SIP", "CAGR", "CEO", "CFO", "CMP",
}

BULLISH_RX = re.compile(
    r"\b(buy|long|bullish|accumulate|target|breakout|multibagger|undervalued|"
    r"rally|surge|moon|rocket|pump|bottom|reversal|uptrend|outperform)\b",
    re.IGNORECASE,
)
BEARISH_RX = re.compile(
    r"\b(sell|short|bearish|exit|breakdown|dump|bubble|overvalued|crash|"
    r"fall|drop|peak|top|downtrend|underperform|stop\s*loss|bagholder)\b",
    re.IGNORECASE,
)
TICKER_TOKEN_RX = re.compile(r"\b[A-Z][A-Z0-9&-]{2,11}\b")

DEFAULT_SUBREDDITS = ("IndianStockMarket", "IndiaInvestments")


class RedditSentimentSource(AltDataSource):
    SLUG = "reddit"
    USER_AGENT = "marketmind/1.0 (alt-data ingestion; +https://localhost)"
    BASE = "https://www.reddit.com/r/{sub}/top.json"

    def __init__(
        self,
        subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
        tickers: Iterable[str] = TIER1_TICKERS,
        timeout_s: float = 8.0,
    ) -> None:
        self.subreddits = tuple(subreddits)
        self.tickers = frozenset(t.upper() for t in tickers)
        self.timeout_s = timeout_s

    @safe_fetch
    def fetch(self) -> List[AltSignal]:
        posts = self._fetch_posts()
        if not posts:
            return []
        return self._summarise(posts)

    # ── HTTP ─────────────────────────────────────────────────────────────
    def _fetch_posts(self) -> List[Dict]:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",  # NEVER add 'br' — keystone NSE fix
        })
        all_posts: List[Dict] = []
        for sub in self.subreddits:
            url = self.BASE.format(sub=sub)
            try:
                r = sess.get(url, params={"t": "week", "limit": 100},
                             timeout=self.timeout_s)
                if r.status_code == 429:
                    logger.info("reddit %s rate-limited (429); skipping", sub)
                    continue
                r.raise_for_status()
                data = r.json()
                children = (data.get("data") or {}).get("children") or []
                all_posts.extend(c.get("data") or {} for c in children)
            except requests.RequestException as e:
                logger.info("reddit %s fetch error: %s", sub, e)
            # be polite to the public endpoint
            time.sleep(0.3)
        return all_posts

    # ── Aggregation ──────────────────────────────────────────────────────
    def _summarise(self, posts: List[Dict]) -> List[AltSignal]:
        as_of = _dt.datetime.now(_dt.timezone.utc)
        n = len(posts)
        upvotes = sum(int(p.get("ups") or 0) for p in posts)
        bullish = sum(1 for p in posts if BULLISH_RX.search(_title(p)))
        bearish = sum(1 for p in posts if BEARISH_RX.search(_title(p)))
        tilt = (bullish - bearish) / n if n else 0.0

        ticker_counts = self._count_ticker_mentions(posts)

        signals: List[AltSignal] = [
            AltSignal(
                source=self.SLUG, key="sentiment_tilt", value=round(tilt, 3),
                unit="ratio", as_of=as_of, confidence=0.8,
                raw={"bullish": bullish, "bearish": bearish, "n": n},
            ),
            AltSignal(
                source=self.SLUG, key="weekly_post_count", value=n,
                unit="count", as_of=as_of, confidence=0.9, raw={},
            ),
            AltSignal(
                source=self.SLUG, key="weekly_upvotes", value=upvotes,
                unit="count", as_of=as_of, confidence=0.9, raw={},
            ),
        ]
        # Top-10 most-mentioned tickers
        for sym, count in sorted(
            ticker_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:10]:
            signals.append(AltSignal(
                source=self.SLUG, key=f"top_ticker_{sym}", value=count,
                unit="mentions", as_of=as_of, confidence=0.7, raw={},
            ))
        return signals

    def _count_ticker_mentions(self, posts: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in posts:
            text = (_title(p) + " " + str(p.get("selftext") or "")).upper()
            for tok in TICKER_TOKEN_RX.findall(text):
                if tok in TICKER_STOP_WORDS:
                    continue
                if tok in self.tickers:
                    counts[tok] = counts.get(tok, 0) + 1
        return counts


def _title(post: Dict) -> str:
    return str(post.get("title") or "")


_singleton: Optional[RedditSentimentSource] = None


def get_reddit_source() -> RedditSentimentSource:
    global _singleton
    if _singleton is None:
        _singleton = RedditSentimentSource()
    return _singleton
