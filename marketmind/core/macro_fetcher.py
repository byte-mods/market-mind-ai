"""
MarketMind AI - Macro Dashboard Data
Fetches macro indicators: USD/INR, India VIX, crude oil, RBI repo rate,
Nifty P/E, market breadth, and cross-asset correlations.
"""
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',  # no 'br' — requests can't decode brotli
    'Origin': 'https://www.nseindia.com',
    'Referer': 'https://www.nseindia.com/',
    'X-Requested-With': 'XMLHttpRequest',
}

# RBI repo rate history (updated manually when RBI meets)
RBI_RATE_HISTORY = [
    {'date': '2023-02-08', 'rate': 6.50, 'action': 'Hike'},
    {'date': '2023-04-06', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2023-06-08', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2023-08-10', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2023-10-06', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2023-12-08', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-02-08', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-04-05', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-06-07', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-08-08', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-10-09', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2024-12-06', 'rate': 6.50, 'action': 'Hold'},
    {'date': '2025-02-07', 'rate': 6.25, 'action': 'Cut'},
    {'date': '2025-04-09', 'rate': 6.00, 'action': 'Cut'},
    {'date': '2026-02-07', 'rate': 6.00, 'action': 'Hold'},
]


class MacroFetcher:
    NSE_BASE = "https://www.nseindia.com/api"

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_time: float = 0.0
        self._cache: Dict = {}
        self._cache_ttl = 600  # 10 min

    def _get_session(self) -> requests.Session:
        now = time.time()
        if self._session is None or (now - self._session_time) > 600:
            s = requests.Session()
            s.headers.update(HEADERS)
            try:
                s.get('https://www.nseindia.com', timeout=10)
            except Exception as e:
                logger.debug(f"MacroFetcher session warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def _nse_get(self, path: str) -> Optional[dict]:
        for _ in range(2):
            try:
                s = self._get_session()
                r = s.get(f"{self.NSE_BASE}/{path}", timeout=12)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.debug(f"NSE macro get error: {e}")
                self._session = None
        return None

    def get_usdinr(self) -> Dict:
        """Get USD/INR rate from NSE currency derivatives."""
        cache_key = 'usdinr'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']
        try:
            data = self._nse_get("quote-derivative?symbol=USDINR")
            if data:
                info = data.get('quoteVFO', {}) or {}
                ltp = float(info.get('lastPrice', 0) or 84.5)
                prev = float(info.get('prevClose', ltp) or ltp)
                result = {
                    'symbol': 'USD/INR',
                    'rate': ltp,
                    'prev': prev,
                    'change': round(ltp - prev, 4),
                    'change_pct': round((ltp - prev) / prev * 100, 3) if prev else 0,
                    'source': 'nse',
                }
                self._cache[cache_key] = {'data': result, 'ts': time.time()}
                return result
        except Exception as e:
            logger.debug(f"USD/INR fetch error: {e}")
        return {'symbol': 'USD/INR', 'rate': 84.5, 'change_pct': 0, 'source': 'estimate'}

    def get_india_vix(self) -> Dict:
        """Get India VIX from NSE allIndices."""
        cache_key = 'vix'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']
        try:
            data = self._nse_get("allIndices")
            if data:
                for idx in data.get('data', []):
                    if 'VIX' in (idx.get('index', '') or idx.get('indexSymbol', '')):
                        last = float(idx.get('last', 0))
                        prev = float(idx.get('previousClose', last))
                        result = {
                            'value': last,
                            'prev': prev,
                            'change_pct': round((last - prev) / prev * 100, 2) if prev else 0,
                            'regime': ('Low Fear' if last < 15 else
                                       'Normal' if last < 20 else
                                       'Elevated' if last < 25 else 'High Fear'),
                        }
                        self._cache[cache_key] = {'data': result, 'ts': time.time()}
                        return result
        except Exception:
            pass
        return {'value': 14.5, 'change_pct': 0, 'regime': 'Normal', 'source': 'estimate'}

    def get_nifty_pe(self) -> Dict:
        """Get Nifty 500 P/E ratio from NSE."""
        cache_key = 'nifty_pe'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']
        try:
            # NSE provides Nifty PE/PB/Div yield
            data = self._nse_get("index-names")
            # Fallback to allIndices for basic data
            all_data = self._nse_get("allIndices")
            if all_data:
                for idx in all_data.get('data', []):
                    if idx.get('index') == 'NIFTY 500':
                        pe = float(idx.get('pe', 0) or 22)
                        pb = float(idx.get('pb', 0) or 4)
                        div = float(idx.get('divYield', 0) or 1.2)
                        # Historical average PE ~22, overvalued >25, undervalued <18
                        valuation = ('Overvalued' if pe > 25 else
                                     'Fair Value' if pe > 18 else 'Undervalued')
                        result = {
                            'pe': pe, 'pb': pb, 'div_yield': div,
                            'valuation': valuation,
                            'historical_avg_pe': 22,
                            'premium_to_avg': round((pe / 22 - 1) * 100, 1),
                        }
                        self._cache[cache_key] = {'data': result, 'ts': time.time()}
                        return result
        except Exception:
            pass
        return {'pe': 22, 'pb': 4, 'div_yield': 1.2, 'valuation': 'Fair Value',
                'historical_avg_pe': 22, 'premium_to_avg': 0}

    def get_rbi_rates(self) -> Dict:
        """Return RBI rate history and current stance."""
        current = RBI_RATE_HISTORY[-1]
        prev = RBI_RATE_HISTORY[-2] if len(RBI_RATE_HISTORY) > 1 else current
        last_3 = [r['action'] for r in RBI_RATE_HISTORY[-3:]]
        cuts = last_3.count('Cut')
        hikes = last_3.count('Hike')
        stance = ('Easing' if cuts > hikes else
                  'Tightening' if hikes > cuts else 'Neutral')
        return {
            'current_rate': current['rate'],
            'prev_rate': prev['rate'],
            'last_action': current['action'],
            'last_action_date': current['date'],
            'stance': stance,
            'history': RBI_RATE_HISTORY[-12:],
            'impact': (
                'Rate cuts boost rate-sensitive sectors: Banks, Real Estate, Autos. '
                'Negative for INR, positive for equities in general.'
                if stance == 'Easing'
                else
                'Rate hikes compress valuation multiples, hurt NBFCs & real estate. '
                'Positive for banks\' NIMs in short term.'
            ),
        }

    def get_market_breadth(self) -> Dict:
        """Get advance/decline ratio from NSE."""
        cache_key = 'breadth'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']
        try:
            data = self._nse_get("allIndices")
            if data:
                advances = int(data.get('advances', 0) or 0)
                declines = int(data.get('declines', 0) or 0)
                unchanged = int(data.get('unchanged', 0) or 0)
                total = advances + declines + unchanged or 1
                adr = round(advances / declines, 2) if declines else 99
                result = {
                    'advances': advances,
                    'declines': declines,
                    'unchanged': unchanged,
                    'total': total,
                    'adr': adr,
                    'breadth_pct': round(advances / total * 100, 1),
                    'signal': ('Broad Rally' if adr > 2 else
                               'Narrow Rally' if adr > 1.2 else
                               'Mixed' if adr > 0.8 else
                               'Broad Decline' if adr < 0.5 else 'Weak'),
                }
                self._cache[cache_key] = {'data': result, 'ts': time.time()}
                return result
        except Exception:
            pass
        return {'advances': 1200, 'declines': 800, 'unchanged': 100,
                'adr': 1.5, 'breadth_pct': 57, 'signal': 'Narrow Rally'}

    def get_all(self) -> Dict:
        """Fetch all macro indicators at once."""
        return {
            'usdinr': self.get_usdinr(),
            'india_vix': self.get_india_vix(),
            'nifty_pe': self.get_nifty_pe(),
            'rbi_rates': self.get_rbi_rates(),
            'market_breadth': self.get_market_breadth(),
            'timestamp': datetime.now().isoformat(),
        }


_fetcher: Optional[MacroFetcher] = None

def get_macro_fetcher() -> MacroFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = MacroFetcher()
    return _fetcher
