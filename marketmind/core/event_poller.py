"""
MarketMind AI - Event Poller (W2.2)

Async background task that polls the NSE corporate-announcements feed for a
watchlist of symbols on a fixed interval, classifies each new announcement
with the two-stage event classifier, persists the result to the Mongo
``events`` collection, and pushes severe events through the WebSocket
broadcaster so the UI drawer can flash.

Idempotency: events keyed by ``{symbol}:{seq_id}`` so re-runs don't duplicate.

Usage from server.py::

    from marketmind.core.event_poller import get_event_poller
    poller = get_event_poller()
    asyncio.create_task(poller.run_loop(broadcast=manager.broadcast))
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "TATAMOTORS", "HINDUNILVR", "BHARTIARTL", "WIPRO",
]


class EventPoller:
    POLL_INTERVAL_S: int = 60
    HISTORY_DAYS_FIRST_RUN: int = 3
    HISTORY_DAYS_STEADY: int = 1
    BROADCAST_THRESHOLD: int = 60   # severity ≥ → push WS event
    MAX_EVENTS_PER_API_CALL: int = 200

    def __init__(self):
        self.symbols: List[str] = list(DEFAULT_WATCHLIST)
        self._seen: Set[str] = set()
        self._first_run: bool = True
        self._last_poll_ts: float = 0.0
        self._last_poll_added: int = 0
        self._running: bool = False
        self._classifier = None
        self._ingester = None
        self._mongo_col = None
        self._mongo_init = False
        # Lazy load to avoid heavy imports at module import time
        self._load_watchlist_from_local_json()

    # ── Setup ──────────────────────────────────────────────────────────────
    def _load_watchlist_from_local_json(self) -> None:
        try:
            from pathlib import Path
            cfg = Path(__file__).resolve().parents[2] / "local.json"
            if cfg.exists():
                d = json.loads(cfg.read_text())
                wl = d.get("watchlist") or []
                if wl:
                    self.symbols = [s.upper().strip() for s in wl if s]
        except Exception as e:
            logger.debug(f"poller watchlist load failed: {e}")

    def _ensure_classifier(self):
        if self._classifier is None:
            from marketmind.core.event_classifier import get_event_classifier
            self._classifier = get_event_classifier()
        return self._classifier

    def _ensure_ingester(self):
        if self._ingester is None:
            from marketmind.core.filings_ingest import get_filings_ingester
            self._ingester = get_filings_ingester()
        return self._ingester

    def _ensure_mongo(self):
        if self._mongo_init:
            return self._mongo_col
        self._mongo_init = True
        try:
            import pymongo
            client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            client.server_info()
            db = client["marketmind"]
            col = db["events"]
            try:
                # 30-day TTL via detected_at_dt index
                col.create_index([("detected_at_dt", pymongo.ASCENDING)],
                                 expireAfterSeconds=30 * 24 * 3600)
                col.create_index([("symbol", pymongo.ASCENDING),
                                  ("detected_at", pymongo.DESCENDING)])
                col.create_index([("severity", pymongo.DESCENDING)])
            except Exception as e:
                logger.debug(f"events index init: {e}")
            self._mongo_col = col
            logger.info("EventPoller: Mongo `events` collection ready")
        except Exception as e:
            logger.warning(f"EventPoller: Mongo unavailable, using in-memory only: {e}")
            self._mongo_col = None
        return self._mongo_col

    # ── Watchlist management ───────────────────────────────────────────────
    def add(self, symbol: str) -> None:
        s = symbol.upper().strip()
        if s and s not in self.symbols:
            self.symbols.append(s)

    def remove(self, symbol: str) -> bool:
        s = symbol.upper().strip()
        if s in self.symbols:
            self.symbols.remove(s)
            return True
        return False

    # ── Polling ────────────────────────────────────────────────────────────
    def _poll_symbol_sync(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Blocking NSE call — must be run inside an executor."""
        try:
            ing = self._ensure_ingester()
            return ing.fetch_announcements(symbol, days=days)
        except Exception as e:
            logger.debug(f"poll {symbol} failed: {e}")
            return []

    async def poll_once(
        self,
        broadcast: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
        executor=None,
    ) -> Dict[str, Any]:
        """One pass over the watchlist. Returns a tally summary."""
        loop = asyncio.get_event_loop()
        days = self.HISTORY_DAYS_FIRST_RUN if self._first_run else self.HISTORY_DAYS_STEADY
        clf = self._ensure_classifier()
        col = self._ensure_mongo()

        async def _one(symbol: str) -> List[Any]:
            rows = await loop.run_in_executor(executor, self._poll_symbol_sync, symbol, days)
            new_events = []
            for row in rows or []:
                seq = str(row.get("seq_id") or row.get("an_dt") or row.get("sort_date") or "")
                key = f"{symbol}:{seq}"
                if not seq or key in self._seen:
                    continue
                self._seen.add(key)
                ev = await loop.run_in_executor(executor, clf.classify_row, symbol, row)
                new_events.append(ev)
            return new_events

        results = await asyncio.gather(*[_one(s) for s in self.symbols],
                                       return_exceptions=True)

        # Aggregate
        tally = {"polled": len(self.symbols), "added": 0, "broadcast": 0,
                 "by_category": {}, "by_severity": {"low": 0, "med": 0, "high": 0}}
        # On the first run we still want to push *severe* historical events into
        # the feed, but most of the noise (procedural board outcomes, etc.) we
        # consider "already seen" so the UI doesn't get spammed at boot.
        first_run_silent_threshold = self.BROADCAST_THRESHOLD if not self._first_run else 75

        from datetime import datetime, timezone
        for symbol_events in results:
            if isinstance(symbol_events, Exception):
                continue
            for ev in symbol_events:
                tally["added"] += 1
                tally["by_category"][ev.category] = tally["by_category"].get(ev.category, 0) + 1
                if ev.severity >= 75:
                    tally["by_severity"]["high"] += 1
                elif ev.severity >= 50:
                    tally["by_severity"]["med"] += 1
                else:
                    tally["by_severity"]["low"] += 1

                doc = ev.to_dict()
                doc["detected_at_dt"] = datetime.now(timezone.utc)
                if col is not None:
                    try:
                        col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
                    except Exception as e:
                        logger.debug(f"events upsert: {e}")

                # Broadcast severe ones (excluding noisy first-run history)
                if broadcast and ev.severity >= first_run_silent_threshold:
                    try:
                        await broadcast({
                            "type": "corp_event",
                            "data": {
                                "symbol": ev.symbol, "seq_id": ev.seq_id,
                                "date": ev.date, "title": ev.title,
                                "category": ev.category, "category_label": ev.category_label,
                                "severity": ev.severity, "direction": ev.direction,
                                "summary": ev.summary, "url": ev.url,
                            },
                        })
                        tally["broadcast"] += 1
                    except Exception as e:
                        logger.debug(f"events broadcast: {e}")

        self._first_run = False
        self._last_poll_ts = time.time()
        self._last_poll_added = tally["added"]
        return tally

    async def run_loop(
        self,
        broadcast: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
        executor=None,
    ):
        if self._running:
            logger.info("EventPoller already running")
            return
        self._running = True
        logger.info(f"EventPoller start — watching {len(self.symbols)} symbols every {self.POLL_INTERVAL_S}s")
        while self._running:
            try:
                tally = await self.poll_once(broadcast=broadcast, executor=executor)
                if tally["added"]:
                    logger.info(f"events: {tally}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"event poll loop error: {e}", exc_info=True)
            await asyncio.sleep(self.POLL_INTERVAL_S)
        self._running = False

    def stop(self) -> None:
        self._running = False

    # ── Read-side ──────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        col = self._ensure_mongo()
        total = 0
        if col is not None:
            try:
                total = col.estimated_document_count()
            except Exception:
                total = 0
        return {
            "running": self._running,
            "symbols": list(self.symbols),
            "poll_interval_s": self.POLL_INTERVAL_S,
            "last_poll_ts": self._last_poll_ts,
            "last_poll_added": self._last_poll_added,
            "broadcast_threshold": self.BROADCAST_THRESHOLD,
            "total_events": total,
            "in_memory_seen": len(self._seen),
        }

    def query(
        self,
        *,
        symbol: Optional[str] = None,
        min_severity: Optional[int] = None,
        category: Optional[str] = None,
        since_iso: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        col = self._ensure_mongo()
        if col is None:
            return []
        q: Dict[str, Any] = {}
        if symbol:
            q["symbol"] = symbol.upper()
        if min_severity is not None:
            q["severity"] = {"$gte": int(min_severity)}
        if category:
            q["category"] = category
        if since_iso:
            try:
                from datetime import datetime
                since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
                q["detected_at_dt"] = {"$gte": since_dt}
            except Exception:
                pass
        try:
            cur = col.find(q).sort([("detected_at", -1)]).limit(min(limit, self.MAX_EVENTS_PER_API_CALL))
            out = []
            for doc in cur:
                doc.pop("_id", None)
                doc.pop("detected_at_dt", None)
                out.append(doc)
            return out
        except Exception as e:
            logger.warning(f"events query: {e}")
            return []


_singleton: Optional[EventPoller] = None


def get_event_poller() -> EventPoller:
    global _singleton
    if _singleton is None:
        _singleton = EventPoller()
    return _singleton
