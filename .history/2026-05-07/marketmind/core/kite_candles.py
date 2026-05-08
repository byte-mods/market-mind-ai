"""
MarketMind AI - Kite Candle Data Manager
Fetches OHLCV candle data from Kite Connect historical API.
Caches in MongoDB (primary) or SQLite (fallback) with configurable TTL.

Cache strategy:
  - Daily/weekly bars: 3-day TTL (only fetch newly completed trading days)
  - Intraday bars:     4-hour TTL (refresh during market hours)
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DAILY_TTL_DAYS = 3
_INTRADAY_TTL_HOURS = 4

# SQLite path (inside marketmind package dir so it persists across runs)
_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'marketmind', 'candles.db'
)


class KiteCandles:
    """
    Manages OHLCV candle data sourced from Kite Connect historical API.

    Data hierarchy (per symbol + interval):
      1. MongoDB cache  → returned if fresh
      2. Kite API fetch → stored in cache on success
      3. Stale cache    → returned when Kite is unavailable
      4. NSE fallback   → daily bars only, via price_fetcher

    Interval aliases accepted:
      '1min' '5min' '10min' '15min' '30min' '60min' '1hour' 'day' 'week'
    """

    INTERVAL_MAP = {
        '1min':   'minute',
        '5min':   '5minute',
        '10min':  '10minute',
        '15min':  '15minute',
        '30min':  '30minute',
        '60min':  '60minute',
        '1hour':  '60minute',
        'day':    'day',
        'week':   'week',
        # Kite native names pass through unchanged
        'minute':    'minute',
        '5minute':   '5minute',
        '10minute':  '10minute',
        '15minute':  '15minute',
        '30minute':  '30minute',
        '60minute':  '60minute',
    }

    def __init__(self):
        self._kite = None
        self._mongo_col = None
        self._sqlite_path = None
        self._init_storage()

    # ── Storage init ────────────────────────────────────────────────────────

    def _init_storage(self):
        # Try MongoDB
        try:
            import pymongo
            client = pymongo.MongoClient(
                'mongodb://localhost:27017/', serverSelectionTimeoutMS=1500
            )
            client.server_info()
            db = client['marketmind']
            self._mongo_col = db['candles']
            self._mongo_col.create_index(
                [('symbol', 1), ('interval', 1)], unique=True
            )
            logger.info("KiteCandles: MongoDB storage active")
            return
        except Exception:
            pass

        # SQLite fallback
        try:
            os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
            conn = sqlite3.connect(_DB_PATH)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol     TEXT NOT NULL,
                    interval   TEXT NOT NULL,
                    data       TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, interval)
                )
            """)
            conn.commit()
            conn.close()
            self._sqlite_path = _DB_PATH
            logger.info("KiteCandles: SQLite fallback storage active")
        except Exception as e:
            logger.error(f"KiteCandles storage init failed: {e}")

    # ── Cache I/O ────────────────────────────────────────────────────────────

    def _cache_read(self, symbol: str, interval: str) -> Optional[Dict]:
        if self._mongo_col is not None:
            try:
                doc = self._mongo_col.find_one(
                    {'symbol': symbol, 'interval': interval}
                )
                if doc:
                    return {'data': doc['data'], 'fetched_at': doc['fetched_at']}
            except Exception as e:
                logger.debug(f"Mongo read error: {e}")

        if self._sqlite_path:
            try:
                conn = sqlite3.connect(self._sqlite_path)
                row = conn.execute(
                    "SELECT data, fetched_at FROM candles WHERE symbol=? AND interval=?",
                    (symbol, interval),
                ).fetchone()
                conn.close()
                if row:
                    return {'data': json.loads(row[0]), 'fetched_at': row[1]}
            except Exception as e:
                logger.debug(f"SQLite read error: {e}")
        return None

    def _cache_write(self, symbol: str, interval: str, data: List[Dict]):
        fetched_at = datetime.utcnow().isoformat()

        if self._mongo_col is not None:
            try:
                self._mongo_col.update_one(
                    {'symbol': symbol, 'interval': interval},
                    {'$set': {'data': data, 'fetched_at': fetched_at}},
                    upsert=True,
                )
                return
            except Exception as e:
                logger.debug(f"Mongo write error: {e}")

        if self._sqlite_path:
            try:
                conn = sqlite3.connect(self._sqlite_path)
                conn.execute(
                    "INSERT OR REPLACE INTO candles (symbol,interval,data,fetched_at) "
                    "VALUES (?,?,?,?)",
                    (symbol, interval, json.dumps(data), fetched_at),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"SQLite write error: {e}")

    def _is_stale(self, fetched_at: str, interval: str) -> bool:
        try:
            fetched = datetime.fromisoformat(fetched_at)
            age_sec = (datetime.utcnow() - fetched).total_seconds()
            is_intraday = interval not in ('day', 'week', 'month')
            ttl = (_INTRADAY_TTL_HOURS * 3600) if is_intraday else (_DAILY_TTL_DAYS * 86400)
            return age_sec > ttl
        except Exception:
            return True

    # ── Kite client ──────────────────────────────────────────────────────────

    def _get_kite(self):
        if self._kite is None:
            try:
                from .kite_client import KiteConfig, KiteClient
                cfg = KiteConfig()
                if cfg.is_authenticated:
                    kc = KiteClient(cfg)
                    if kc.is_connected:
                        self._kite = kc
            except Exception as e:
                logger.debug(f"Kite init error: {e}")
        return self._kite

    # ── Fetch from Kite ──────────────────────────────────────────────────────

    def _fetch_from_kite(self, symbol: str, kite_interval: str, days: int) -> List[Dict]:
        kite = self._get_kite()
        if not kite:
            return []
        try:
            candles = kite.get_historical_data_by_symbol(
                symbol=symbol, exchange='NSE', interval=kite_interval, days=days
            )
            result = []
            for c in candles:
                dt = c.get('date')
                if hasattr(dt, 'strftime'):
                    fmt = '%Y-%m-%d' if kite_interval == 'day' else '%Y-%m-%d %H:%M:%S'
                    dt = dt.strftime(fmt)
                else:
                    dt = str(dt)
                result.append({
                    'date':   dt,
                    'open':   float(c.get('open', 0)),
                    'high':   float(c.get('high', 0)),
                    'low':    float(c.get('low', 0)),
                    'close':  float(c.get('close', 0)),
                    'volume': int(c.get('volume', 0)),
                })
            return result
        except Exception as e:
            logger.error(f"Kite historical fetch error for {symbol}/{kite_interval}: {e}")
            return []

    # ── Public API ───────────────────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        interval: str = 'day',
        days: int = 365,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Return OHLCV candle list for symbol.

        interval: '1min' | '5min' | '10min' | '15min' | '30min' | '60min' | 'day'
        days:     how many calendar days of history to fetch
        Returns list of dicts: {date, open, high, low, close, volume}
        """
        symbol = symbol.upper()
        kite_interval = self.INTERVAL_MAP.get(interval, interval)

        # 1. Try fresh cache
        if not force_refresh:
            cached = self._cache_read(symbol, interval)
            if cached and not self._is_stale(cached['fetched_at'], interval):
                return cached['data']

        # 2. Fetch from Kite
        data = self._fetch_from_kite(symbol, kite_interval, days)
        if data:
            self._cache_write(symbol, interval, data)
            return data

        # 3. Return stale cache rather than empty
        if not force_refresh:
            cached = self._cache_read(symbol, interval)
            if cached:
                logger.debug(f"Returning stale cache for {symbol}/{interval}")
                return cached['data']

        # 4. NSE fallback (daily only)
        if interval in ('day', 'week'):
            try:
                from .price_fetcher import get_price_fetcher
                hist = get_price_fetcher().get_historical_data(symbol, days=days)
                if not hist.empty:
                    hist['date'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')
                    result = (
                        hist[['date', 'open', 'high', 'low', 'close', 'volume']]
                        .to_dict('records')
                    )
                    if result:
                        self._cache_write(symbol, interval, result)
                        return result
            except Exception as e:
                logger.debug(f"NSE fallback error for {symbol}: {e}")

        return []

    def get_candles_df(
        self,
        symbol: str,
        interval: str = 'day',
        days: int = 365,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Return candle data as pandas DataFrame for backtesting. None if insufficient."""
        data = self.get_candles(symbol, interval=interval, days=days,
                                force_refresh=force_refresh)
        if not data or len(data) < 30:
            return None
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna().reset_index(drop=True)


_instance: Optional[KiteCandles] = None


def get_kite_candles() -> KiteCandles:
    global _instance
    if _instance is None:
        _instance = KiteCandles()
    return _instance
