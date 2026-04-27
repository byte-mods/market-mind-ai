"""
MarketMind AI - Price Fetcher Module
Fetches Indian stock market data from NSE India API (live prices)
and Screener.in (fundamental data). Replaces yfinance entirely.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetches stock price data from NSE India API and Screener.in"""

    NSE_BASE = "https://www.nseindia.com/api"
    SCREENER_BASE = "https://www.screener.in"

    # NSE index symbol map (for allIndices API)
    INDICES = {
        'NIFTY500': 'NIFTY 500',
        'NIFTY50': 'NIFTY 50',
        'NIFTYBANK': 'NIFTY BANK',
        'NIFTYIT': 'NIFTY IT',
        'NIFTYAUTO': 'NIFTY AUTO',
        'NIFTYPHARMA': 'NIFTY PHARMA',
        'NIFTYFIN': 'NIFTY FINANCIAL SERVICES',
        'NIFTYFMCG': 'NIFTY FMCG',
        'NIFTYMETAL': 'NIFTY METAL',
        'NIFTYREALTY': 'NIFTY REALTY',
        'NIFTYENERGY': 'NIFTY ENERGY',
        'SENSEX': 'SENSEX',
        'INDIA VIX': 'INDIA VIX',
    }

    # Popular NSE stocks
    NSE_STOCKS = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
        'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'HINDUNILVR',
        'AXISBANK', 'KOTAKBANK', 'MARUTI', 'TATAMOTORS', 'M&M',
        'TATASTEEL', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'ONGC',
        'POWERGRID', 'NTPC', 'COALINDIA', 'BAJFINANCE', 'BAJAJFINSV',
        'HCLTECH', 'WIPRO', 'TECHM', 'ADANIPORTS', 'WIPRO',
    ]

    _NSE_HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/',
        'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    # NSE rate-limit guard: 1 req per ~700ms across the whole process
    _NSE_REQ_INTERVAL = 0.7

    def __init__(self):
        self.cache: Dict = {}
        self.cache_duration = 300   # 5 minutes for prices
        self.fund_cache_ttl = 3600  # 1 hour for fundamentals
        self.lock = threading.Lock()

        # Separate sessions for NSE (cookie-based) and Screener
        self._nse_session: Optional[requests.Session] = None
        self._nse_session_time = 0.0
        self._nse_session_lock = threading.Lock()
        self._nse_throttle_lock = threading.Lock()
        self._nse_last_req = 0.0
        self._screener_session = requests.Session()
        self._screener_session.headers.update({
            'User-Agent': self._NSE_HEADERS['User-Agent'],
            'Accept-Language': 'en-US,en;q=0.9',
        })

    # ------------------------------------------------------------------
    # NSE session management
    # ------------------------------------------------------------------

    def _build_nse_session(self) -> requests.Session:
        """Build a freshly-warmed NSE session. Two-step warm-up sets bm_sv cookies."""
        session = requests.Session()
        session.headers.update(self._NSE_HEADERS)
        try:
            # Step 1: homepage sets nsit / nseappid
            session.get('https://www.nseindia.com', timeout=10)
            # Step 2: equity quote page sets bm_sv and finishes the bot challenge
            session.get(
                'https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE',
                timeout=10,
            )
            logger.info("NSE session refreshed")
        except Exception as e:
            logger.warning(f"NSE warm-up failed (cookies may be stale): {e}")
        return session

    def _get_nse_session(self) -> requests.Session:
        """Return a valid NSE session (refresh if stale >15 min). Thread-safe."""
        now = time.time()
        with self._nse_session_lock:
            if self._nse_session is None or (now - self._nse_session_time) > 900:
                self._nse_session = self._build_nse_session()
                self._nse_session_time = now
            return self._nse_session

    def _throttle_nse(self):
        """Block until at least _NSE_REQ_INTERVAL has passed since the last NSE request."""
        with self._nse_throttle_lock:
            wait = self._NSE_REQ_INTERVAL - (time.time() - self._nse_last_req)
            if wait > 0:
                time.sleep(wait)
            self._nse_last_req = time.time()

    def _nse_get(self, path: str, params: dict = None, timeout: int = 12) -> Optional[dict]:
        """GET from NSE API with throttle + one retry on session expiry."""
        for attempt in range(2):
            try:
                self._throttle_nse()
                session = self._get_nse_session()
                url = f"{self.NSE_BASE}/{path.lstrip('/')}"
                resp = session.get(url, params=params, timeout=timeout)
                # Throttle / soft-block: 401/403/429 → reset session + back off
                if resp.status_code in (401, 403, 429):
                    logger.debug(f"NSE {resp.status_code} for {path} — resetting session")
                    with self._nse_session_lock:
                        self._nse_session = None
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                body = resp.text.strip() if resp.content else ''
                if not body or body[0] not in ('{', '['):
                    with self._nse_session_lock:
                        self._nse_session = None
                    time.sleep(0.5)
                    continue
                return resp.json()
            except Exception as e:
                logger.debug(f"NSE API attempt {attempt+1} for {path}: {e}")
                with self._nse_session_lock:
                    self._nse_session = None
                time.sleep(0.5)
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_symbol(self, symbol: str) -> str:
        """Return the canonical symbol (no suffix needed for NSE API)."""
        return symbol

    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get current stock price from NSE India API."""
        if symbol in self.INDICES:
            return self._get_index_price(symbol)

        cache_key = f"price_{symbol}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_duration:
                return cached['data']

        data = self._nse_get(f"quote-equity?symbol={symbol}")
        if not data:
            return self._screener_price_fallback(symbol)

        price_info = data.get('priceInfo', {})
        meta = data.get('metadata', {})
        intra = price_info.get('intraDayHighLow', {})
        week_hl = price_info.get('weekHighLow', {})

        current_price = price_info.get('lastPrice', 0)
        prev_close = price_info.get('previousClose', 0)
        change = current_price - prev_close
        change_pct = (change / prev_close) if prev_close else 0

        price_data: Dict = {
            'symbol': symbol,
            'name': meta.get('companyName', symbol),
            'current_price': current_price,
            'previous_close': prev_close,
            'open': price_info.get('open', 0),
            'day_high': intra.get('max', 0),
            'day_low': intra.get('min', 0),
            '52_week_high': week_hl.get('max', 0),
            '52_week_low': week_hl.get('min', 0),
            'volume': data.get('securityWiseDP', {}).get('quantityTraded', 0),
            'change': change,
            'change_pct': change_pct,
            'market_cap': None,
            'pe_ratio': None,
            'pb_ratio': None,
            'source': 'nse',
            'timestamp': time.time(),
        }

        # Enrich with fundamentals from Screener.in
        fundamentals = self._get_screener_fundamentals(symbol)
        if fundamentals:
            price_data.update(fundamentals)

        with self.lock:
            self.cache[cache_key] = {'data': price_data, 'timestamp': time.time()}
        return price_data

    def _get_index_price(self, index_name: str) -> Optional[Dict]:
        """Get index price from NSE allIndices endpoint."""
        cache_key = f"index_{index_name}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_duration:
                return cached['data']

        nse_name = self.INDICES.get(index_name, index_name)
        data = self._nse_get("allIndices")
        if not data:
            return None

        for idx in data.get('data', []):
            if idx.get('index') == nse_name or idx.get('indexSymbol') == nse_name:
                last = idx.get('last', 0)
                prev = idx.get('previousClose', last)
                change = last - prev
                result = {
                    'symbol': index_name,
                    'name': nse_name,
                    'current_price': last,
                    'change': change,
                    'change_pct': (change / prev) if prev else 0,
                    'day_high': idx.get('high', 0),
                    'day_low': idx.get('low', 0),
                    'previous_close': prev,
                    'source': 'nse',
                    'timestamp': time.time(),
                }
                with self.lock:
                    self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                return result
        return None

    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical OHLCV. Prefers Kite (reliable, authenticated) over NSE history API."""
        cache_key = f"hist_{symbol}_{days}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_duration:
                return cached['data']

        # ── Fast path: Kite historical (when authenticated) ──
        df = self._kite_historical(symbol, days)
        if df is None or df.empty:
            # NSE fallback
            if symbol in self.INDICES:
                df = self._get_index_historical(symbol, days)
            else:
                df = self._get_equity_historical(symbol, days)

        with self.lock:
            self.cache[cache_key] = {'data': df, 'timestamp': time.time()}
        return df

    def _kite_historical(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Pull historical bars from Kite Connect if a connected client is available."""
        try:
            # Lazy import: avoid hard coupling at module load
            from marketmind.core.kite_client import KiteClient  # noqa
        except Exception:
            return None
        try:
            # Find a singleton Kite via the AppController if it's been initialised.
            import sys as _sys
            ctrl = getattr(_sys.modules.get('marketmind.app_controller'), 'AppController', None)
            inst = None
            for mod in _sys.modules.values():
                if mod is None: continue
                cand = getattr(mod, 'controller', None)
                if cand is not None and hasattr(cand, 'kite'):
                    inst = cand
                    break
            kite = inst.kite if inst is not None else None
            if not kite or not kite.is_connected:
                return None
            # Map index symbols to Kite tradingsymbols
            kite_sym = {
                'NIFTY500': ('NIFTY 500', 'NSE'),
                'NIFTY50':  ('NIFTY 50',  'NSE'),
                'SENSEX':   ('SENSEX',    'BSE'),
                'NIFTYBANK':('NIFTY BANK','NSE'),
            }.get(symbol, (symbol, 'NSE'))
            tradingsymbol, exchange = kite_sym
            candles = kite.get_historical_data_by_symbol(tradingsymbol, exchange, 'day', days)
            if not candles:
                return None
            df = pd.DataFrame(candles)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            for col in ('open', 'high', 'low', 'close', 'volume'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        except Exception as e:
            logger.debug(f"Kite historical fallback for {symbol}: {e}")
            return None

    def _get_equity_historical(self, symbol: str, days: int) -> pd.DataFrame:
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        path = (
            f"historical/cm/equity?symbol={symbol}"
            f'&series=["EQ"]'
            f"&from={from_dt.strftime('%d-%m-%Y')}"
            f"&to={to_dt.strftime('%d-%m-%Y')}"
        )
        data = self._nse_get(path)
        if not data:
            return pd.DataFrame()

        records = data.get('data', [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        col_map = {
            'CH_TIMESTAMP': 'date',
            'CH_OPENING_PRICE': 'open',
            'CH_TRADE_HIGH_PRICE': 'high',
            'CH_TRADE_LOW_PRICE': 'low',
            'CH_CLOSING_PRICE': 'close',
            'CH_TOT_TRADED_QTY': 'volume',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep].copy()
        if 'volume' not in df.columns:
            df['volume'] = 0
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def _get_index_historical(self, index_name: str, days: int) -> pd.DataFrame:
        nse_name = self.INDICES.get(index_name, index_name)
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        import urllib.parse
        path = (
            f"historical/indicesHistory"
            f"?indexType={urllib.parse.quote(nse_name)}"
            f"&from={from_dt.strftime('%d-%m-%Y')}"
            f"&to={to_dt.strftime('%d-%m-%Y')}"
        )
        data = self._nse_get(path)
        if not data:
            return pd.DataFrame()

        records = (data.get('data', {}) or {}).get('indexCloseOnlineRecords', [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        col_map = {
            'EOD_TIMESTAMP': 'date',
            'EOD_OPEN_INDEX_VAL': 'open',
            'EOD_HIGH_INDEX_VAL': 'high',
            'EOD_LOW_INDEX_VAL': 'low',
            'EOD_CLOSE_INDEX_VAL': 'close',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        keep = [c for c in ['date', 'open', 'high', 'low', 'close'] if c in df.columns]
        df = df[keep].copy()
        df['volume'] = 0
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df.sort_values('date').reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Screener.in fundamentals
    # ------------------------------------------------------------------

    def _get_screener_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Scrape P/E, P/B, Market Cap, ROE, ROCE from Screener.in."""
        cache_key = f"screener_{symbol}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.fund_cache_ttl:
                return cached['data']

        try:
            # Resolve company URL via search
            search_url = f"{self.SCREENER_BASE}/api/company/search/?q={symbol}&v=3"
            resp = self._screener_session.get(search_url, timeout=10)
            resp.raise_for_status()
            results = resp.json()
            if not results:
                return None

            company_path = results[0].get('url', '').strip('/')
            if not company_path:
                return None

            company_url = f"{self.SCREENER_BASE}/{company_path}/"
            resp = self._screener_session.get(company_url, timeout=15)
            resp.raise_for_status()
            fundamentals = self._parse_screener_ratios(resp.text)

            with self.lock:
                self.cache[cache_key] = {'data': fundamentals, 'timestamp': time.time()}
            return fundamentals
        except Exception as e:
            logger.warning(f"Screener fundamentals error for {symbol}: {e}")
            return None

    def _parse_screener_ratios(self, html: str) -> Dict:
        """Parse top-ratios section from a Screener.in company page."""
        soup = BeautifulSoup(html, 'lxml')
        result: Dict = {}
        try:
            ratios_ul = soup.find('ul', {'id': 'top-ratios'})
            if not ratios_ul:
                return result
            for li in ratios_ul.find_all('li'):
                name_tag = li.find('span', class_='name')
                value_tag = (li.find('span', class_='value')
                             or li.find('span', class_='number'))
                if not (name_tag and value_tag):
                    continue
                name = name_tag.get_text(strip=True).lower()
                raw = (value_tag.get_text(strip=True)
                       .replace(',', '').replace('%', '')
                       .replace('₹', '').replace('Cr', '').strip())
                try:
                    value = float(raw)
                except ValueError:
                    continue

                if 'market cap' in name:
                    result['market_cap'] = value * 1e7  # Screener shows Cr
                elif name == 'stock p/e' or 'p/e' in name or name == 'pe':
                    result['pe_ratio'] = value
                elif 'p/b' in name or 'price to book' in name:
                    result['pb_ratio'] = value
                elif 'book value' in name:
                    result['book_value'] = value
                elif 'current price' in name:
                    result['screener_price'] = value
                elif 'roe' in name:
                    result['roe'] = value
                elif 'roce' in name:
                    result['roce'] = value
                elif 'div' in name and 'yield' in name:
                    result['dividend_yield'] = value
                elif 'debt' in name and 'equity' in name:
                    result['debt_equity'] = value
                elif 'eps' in name:
                    result['eps'] = value
                elif 'sales growth' in name or 'revenue growth' in name:
                    result['revenue_growth'] = value
                elif 'profit growth' in name:
                    result['profit_growth'] = value

            # P/B = price / book value (Screener exposes Book Value, not P/B directly)
            if 'pb_ratio' not in result and result.get('book_value') and result.get('screener_price'):
                bv = result['book_value']
                pr = result['screener_price']
                if bv > 0:
                    result['pb_ratio'] = round(pr / bv, 2)
        except Exception as e:
            logger.warning(f"Screener parse error: {e}")
        return result

    def _screener_price_fallback(self, symbol: str) -> Optional[Dict]:
        """Minimal fallback when NSE API is unavailable."""
        try:
            search_url = f"{self.SCREENER_BASE}/api/company/search/?q={symbol}&v=3"
            resp = self._screener_session.get(search_url, timeout=10)
            results = resp.json()
            if results:
                return {
                    'symbol': symbol,
                    'name': results[0].get('name', symbol),
                    'current_price': 0,
                    'change': 0,
                    'change_pct': 0,
                    'source': 'screener_fallback',
                    'timestamp': time.time(),
                }
        except Exception as e:
            logger.error(f"Screener price fallback error for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------
    # Multi-symbol helpers
    # ------------------------------------------------------------------

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple symbols."""
        results = {}
        for symbol in symbols:
            price = self.get_stock_price(symbol)
            if price:
                results[symbol] = price
        return results

    def get_index_components(self, index_name: str) -> List[str]:
        """Return representative stocks for a sector index."""
        sector_map = {
            'NIFTYIT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTTS', 'COFORGE'],
            'NIFTYBANK': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK'],
            'NIFTYAUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT'],
            'NIFTYPHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA', 'ZYDUSLIFE'],
            'NIFTYFIN': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'ICICIPRULI', 'MAXHEALTH'],
            'NIFTYFMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR', 'COLPAL', 'MARICO'],
            'NIFTYMETAL': ['TATASTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL'],
            'NIFTYREALTY': ['DLF', 'GODREJPROP', 'BRIGADE', 'PRESTIGE', 'OBEROIRLTY'],
            'NIFTYENERGY': ['ONGC', 'RELIANCE', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL'],
        }
        return sector_map.get(index_name, [])

    # ------------------------------------------------------------------
    # Technical indicators (unchanged logic, new data source)
    # ------------------------------------------------------------------

    def calculate_technical_indicators(self, symbol: str, days: int = 100) -> Dict:
        """Calculate technical indicators from NSE historical data."""
        hist = self.get_historical_data(symbol, days)
        if hist.empty or len(hist) < 20:
            return {}

        close = hist['close'].astype(float)
        high = hist['high'].astype(float)
        low = hist['low'].astype(float)
        volume = hist['volume'].astype(float)

        ma_20 = close.rolling(window=20).mean().iloc[-1]
        ma_50 = close.rolling(window=min(50, len(close))).mean().iloc[-1]
        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        bb_std = close.rolling(window=20).std()
        bb_upper = ma_20 + bb_std.iloc[-1] * 2
        bb_lower = ma_20 - bb_std.iloc[-1] * 2

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        # Volume
        avg_vol = volume.rolling(window=20).mean().iloc[-1]
        vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0

        # Momentum
        m5 = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else 0
        m10 = float(close.iloc[-1] / close.iloc[-11] - 1) if len(close) > 10 else 0
        m20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else 0

        return {
            'symbol': symbol,
            'current_price': float(close.iloc[-1]),
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'ma_200': float(ma_200) if ma_200 is not None else None,
            'rsi': float(rsi) if not np.isnan(rsi) else 50.0,
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(signal.iloc[-1]),
            'macd_histogram': float(macd.iloc[-1] - signal.iloc[-1]),
            'bb_upper': float(bb_upper),
            'bb_lower': float(bb_lower),
            'atr': float(atr),
            'volume_ratio': float(vol_ratio),
            'momentum_5': m5,
            'momentum_10': m10,
            'momentum_20': m20,
            'above_ma_20': bool(close.iloc[-1] > ma_20),
            'above_ma_50': bool(close.iloc[-1] > ma_50),
            'above_ma_200': bool(ma_200 is not None and close.iloc[-1] > ma_200),
        }

    # ------------------------------------------------------------------
    # Market status
    # ------------------------------------------------------------------

    def get_market_status(self) -> Dict:
        """Check if Indian market is open (IST, no yfinance needed)."""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_weekend = now.weekday() >= 5
        is_open = not is_weekend and market_open <= now <= market_close

        if is_weekend:
            next_open = now + timedelta(days=(7 - now.weekday()))
        elif now > market_close:
            next_open = now + timedelta(days=1)
            if next_open.weekday() >= 5:
                next_open += timedelta(days=7 - next_open.weekday())
        else:
            next_open = market_open

        return {
            'is_open': is_open,
            'is_weekend': is_weekend,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'market_open_time': market_open.strftime('%H:%M:%S'),
            'market_close_time': market_close.strftime('%H:%M:%S'),
            'next_open': next_open.strftime('%Y-%m-%d %H:%M:%S'),
        }


# Global instance
_price_fetcher: Optional[PriceFetcher] = None


def get_price_fetcher() -> PriceFetcher:
    """Get or create global price fetcher instance."""
    global _price_fetcher
    if _price_fetcher is None:
        _price_fetcher = PriceFetcher()
    return _price_fetcher
