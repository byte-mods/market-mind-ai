"""
MarketMind AI - NSE Options Chain Fetcher
Fetches real-time options data, PCR, max pain from NSE India API.
"""

import requests
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INDICES = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50'}


class OptionsFetcher:
    NSE_BASE = "https://www.nseindia.com/api"
    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        # 'br' omitted: requests can't decode brotli; NSE will use gzip instead
        'Accept-Encoding': 'gzip, deflate',
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/option-chain',
        'X-Requested-With': 'XMLHttpRequest',
    }

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_time: float = 0.0
        self._cache: Dict = {}
        self._cache_ttl = 180  # 3 minutes

    def _get_session(self) -> requests.Session:
        now = time.time()
        if self._session is None or (now - self._session_time) > 900:
            s = requests.Session()
            s.headers.update(self.HEADERS)
            try:
                s.get('https://www.nseindia.com', timeout=10)
                # Two-step warm-up: option-chain page sets bm_sv / _abck cookies
                s.get('https://www.nseindia.com/option-chain', timeout=10)
            except Exception as e:
                logger.debug(f"OptionsFetcher session warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def get_option_chain(self, symbol: str) -> Dict:
        """Fetch full options chain for equity or index symbol."""
        cache_key = f"opt_{symbol}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']

        try:
            session = self._get_session()
            if symbol.upper() in INDICES:
                url = f"{self.NSE_BASE}/option-chain-indices?symbol={symbol.upper()}"
            else:
                url = f"{self.NSE_BASE}/option-chain-equities?symbol={symbol.upper()}"

            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            result = self._parse_chain(resp.json(), symbol.upper())
            self._cache[cache_key] = {'data': result, 'ts': time.time()}
            return result
        except Exception as e:
            logger.warning(f"Options chain fetch error [{symbol}]: {e}")
            self._session = None
            return {'error': str(e), 'symbol': symbol, 'calls': [], 'puts': []}

    def _parse_chain(self, raw: dict, symbol: str) -> Dict:
        records = raw.get('records', {})
        data = records.get('data', [])
        expiry_dates = records.get('expiryDates', [])
        underlying = records.get('underlyingValue', 0)

        # NSE returns {} when markets are closed or data is unavailable
        if not raw or not records or not data:
            return {
                'symbol': symbol,
                'underlying': 0,
                'atm_strike': 0,
                'expiry_dates': [],
                'calls': [], 'puts': [],
                'total_call_oi': 0, 'total_put_oi': 0,
                'max_call_oi_strike': 0, 'max_put_oi_strike': 0,
                'pcr': 1.0, 'max_pain': 0.0,
                'sentiment': 'Unavailable',
                'unavailable': True,
                'reason': 'Options chain unavailable — markets closed or NSE not serving data',
                'timestamp': time.time(),
            }

        calls: List[Dict] = []
        puts: List[Dict] = []
        total_call_oi = 0
        total_put_oi = 0
        max_call_oi = 0
        max_put_oi = 0

        for item in data:
            strike = item.get('strikePrice', 0)
            ce = item.get('CE') or {}
            pe = item.get('PE') or {}

            if ce:
                c_oi = ce.get('openInterest', 0)
                total_call_oi += c_oi
                if c_oi > max_call_oi:
                    max_call_oi = c_oi
                calls.append({
                    'strike': strike,
                    'oi': c_oi,
                    'chg_oi': ce.get('changeinOpenInterest', 0),
                    'volume': ce.get('totalTradedVolume', 0),
                    'iv': ce.get('impliedVolatility', 0),
                    'ltp': ce.get('lastPrice', 0),
                    'bid': ce.get('bidprice', 0),
                    'ask': ce.get('askPrice', 0),
                })
            if pe:
                p_oi = pe.get('openInterest', 0)
                total_put_oi += p_oi
                if p_oi > max_put_oi:
                    max_put_oi = p_oi
                puts.append({
                    'strike': strike,
                    'oi': p_oi,
                    'chg_oi': pe.get('changeinOpenInterest', 0),
                    'volume': pe.get('totalTradedVolume', 0),
                    'iv': pe.get('impliedVolatility', 0),
                    'ltp': pe.get('lastPrice', 0),
                    'bid': pe.get('bidprice', 0),
                    'ask': pe.get('askPrice', 0),
                })

        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi else 1.0
        max_pain = self._calc_max_pain(data)

        # ATM strike
        atm = min(set(i.get('strikePrice', 0) for i in data),
                  key=lambda x: abs(x - underlying)) if data else 0

        return {
            'symbol': symbol,
            'underlying': underlying,
            'atm_strike': atm,
            'expiry_dates': expiry_dates[:8],
            'calls': calls,
            'puts': puts,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'max_call_oi_strike': max(calls, key=lambda x: x['oi'])['strike'] if calls else 0,
            'max_put_oi_strike': max(puts, key=lambda x: x['oi'])['strike'] if puts else 0,
            'pcr': pcr,
            'max_pain': max_pain,
            'sentiment': 'Bullish' if pcr > 1.2 else ('Bearish' if pcr < 0.7 else 'Neutral'),
            'timestamp': time.time(),
        }

    def _calc_max_pain(self, data: list) -> float:
        """Max pain = strike where option buyers lose the most money."""
        if not data:
            return 0.0
        strikes = sorted({item.get('strikePrice', 0) for item in data})
        if not strikes:
            return 0.0

        min_pain = float('inf')
        max_pain_strike = strikes[0]
        for target in strikes:
            pain = 0
            for item in data:
                s = item.get('strikePrice', 0)
                ce_oi = (item.get('CE') or {}).get('openInterest', 0)
                pe_oi = (item.get('PE') or {}).get('openInterest', 0)
                if target > s:
                    pain += (target - s) * ce_oi
                if target < s:
                    pain += (s - target) * pe_oi
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = target

        return float(max_pain_strike)


_fetcher: Optional[OptionsFetcher] = None


def get_options_fetcher() -> OptionsFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = OptionsFetcher()
    return _fetcher
