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
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        # 'br' omitted: requests can't decode brotli; NSE will use gzip instead
        'Accept-Encoding': 'gzip, deflate',
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/option-chain',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
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
                # Root page often 403s behind Akamai; option-chain page is the
                # one that actually sets nsit / _abck / bm_sz cookies.
                r = s.get('https://www.nseindia.com', timeout=10)
                if r.status_code == 403:
                    logger.debug("OptionsFetcher: nseindia.com root returned 403 (expected)")
                s.get('https://www.nseindia.com/option-chain', timeout=10)
            except Exception as e:
                logger.debug(f"OptionsFetcher session warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def get_fo_symbols(self) -> List[str]:
        """Fetch all F&O tradable symbols from NSE.

        Returns a sorted list of upper-case trading symbols (equities + indices).
        Cached for 1 hour — the F&O list changes only on monthly expiry cycles.
        """
        cache_key = "fo_symbols"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < 3600:
            return cached['data']

        try:
            session = self._get_session()
            url = f"{self.NSE_BASE}/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            symbols = sorted({d['symbol'].upper() for d in data if d.get('symbol')})
            # Inject well-known index symbols that NSE lists separately
            symbols = sorted(set(symbols) | INDICES)
            self._cache[cache_key] = {'data': symbols, 'ts': time.time()}
            return symbols
        except Exception as e:
            logger.warning(f"F&O symbols fetch error: {e}")
            # Fallback to well-known set so the UI isn't bricked
            fallback = sorted({
                'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50',
                'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN',
                'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK', 'AXISBANK', 'BAJFINANCE',
                'MARUTI', 'SUNPHARMA', 'TATAMOTORS', 'WIPRO', 'ADANIENT',
                'ADANIPORTS', 'ULTRACEMCO', 'NESTLEIND', 'TITAN', 'POWERGRID',
                'NTPC', 'INDUSINDBK', 'GRASIM', 'ONGC', 'HINDALCO', 'JSWSTEEL',
                'COALINDIA', 'BPCL', 'BRITANNIA', 'SHRIRAMFIN', 'CIPLA',
                'APOLLOHOSP', 'DRREDDY', 'EICHERMOT', 'DIVISLAB', 'HCLTECH',
                'M&M', 'TECHM', 'HEROMOTOCO', 'TATASTEEL', 'BAJAJ-AUTO',
                'HAL', 'PNB', 'IOB', 'ZOMATO', 'IRCTC', 'BEL', 'GAIL',
            })
            return fallback

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
            return {
                'error': str(e),
                'symbol': symbol,
                'calls': [],
                'puts': [],
                'unavailable': True,
                'reason': 'NSE option chain unavailable — markets may be closed (09:15–15:30 IST) or data feed is delayed',
            }

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
                'reason': 'Options chain unavailable — markets closed (09:15–15:30 IST) or NSE not serving data',
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

    def get_option_chain_from_kite(self, symbol: str, kite_client) -> Optional[Dict]:
        """Build option chain from Kite Connect when NSE is unavailable.

        Uses ``kite.instruments('NFO')`` to discover option contracts, then
        ``kite.quote()`` to fetch OI, volume, LTP for the nearest expiry.
        Returns None if Kite is not connected or no options are found.
        """
        if not kite_client or not getattr(kite_client, 'is_connected', False):
            return None

        try:
            # 1. Get all NFO instruments (cached inside KiteClient)
            instruments = kite_client.get_instruments('NFO')
            if not instruments:
                return None

            sym_upper = symbol.upper()
            # Kite lists indices differently — map common names
            kite_name = sym_upper
            if sym_upper == 'NIFTY':
                kite_name = 'NIFTY 50'
            elif sym_upper == 'BANKNIFTY':
                kite_name = 'NIFTY BANK'
            elif sym_upper == 'FINNIFTY':
                kite_name = 'NIFTY FINANCIAL SERVICES'
            elif sym_upper == 'MIDCPNIFTY':
                kite_name = 'NIFTY MIDCAP SELECT'
            elif sym_upper == 'NIFTYNXT50':
                kite_name = 'NIFTY NEXT 50'

            # 2. Filter options for this symbol
            opts = [
                inst for inst in instruments
                if inst.get('name') == kite_name
                and inst.get('instrument_type') in ('CE', 'PE')
                and inst.get('segment') == 'NFO-OPT'
            ]
            if not opts:
                # Try matching by tradingsymbol prefix for equities
                opts = [
                    inst for inst in instruments
                    if inst.get('tradingsymbol', '').startswith(sym_upper)
                    and inst.get('instrument_type') in ('CE', 'PE')
                    and inst.get('segment') == 'NFO-OPT'
                ]
            if not opts:
                return None

            # 3. Group by expiry, pick nearest
            from datetime import date
            today = date.today()
            expiries = sorted({
                inst['expiry'] for inst in opts
                if isinstance(inst.get('expiry'), date)
            })
            if not expiries:
                return None
            nearest_expiry = min(expiries, key=lambda d: abs((d - today).days) if (d - today).days >= 0 else float('inf'))
            # If all expiries are in the past, pick the closest one anyway
            if (nearest_expiry - today).days < 0:
                nearest_expiry = min(expiries, key=lambda d: abs((d - today).days))

            near_opts = [inst for inst in opts if inst.get('expiry') == nearest_expiry]

            # 4. Get underlying price
            exchange = 'NSE'
            if sym_upper in INDICES:
                exchange = 'NSE'
            ltp_resp = kite_client.get_ltp([f"{exchange}:{sym_upper}"])
            underlying = 0.0
            if ltp_resp:
                underlying = float(
                    (ltp_resp.get(f"{exchange}:{sym_upper}") or {}).get('last_price', 0)
                )

            # 5. Determine strikes to fetch — all strikes for nearest expiry
            # Sort strikes, pick a reasonable range around ATM
            strikes = sorted({float(inst['strike']) for inst in near_opts})
            if underlying and strikes:
                atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying))
                start = max(0, atm_idx - 10)
                end = min(len(strikes), atm_idx + 11)
                target_strikes = strikes[start:end]
            else:
                target_strikes = strikes

            # 6. Build instrument list for quote call
            quote_instruments = []
            strike_to_inst = {}
            for inst in near_opts:
                strike = float(inst['strike'])
                if strike in target_strikes:
                    key = f"NFO:{inst['tradingsymbol']}"
                    quote_instruments.append(key)
                    strike_to_inst[key] = {
                        'strike': strike,
                        'type': inst['instrument_type'],
                        'lot_size': inst.get('lot_size', 1),
                    }

            # 7. Fetch quotes in batches (Kite limits ~500 instruments per call)
            BATCH = 400
            quotes: Dict = {}
            for i in range(0, len(quote_instruments), BATCH):
                batch = quote_instruments[i:i+BATCH]
                batch_quotes = kite_client.get_quote(batch)
                if batch_quotes:
                    quotes.update(batch_quotes)

            # 8. Build chain format matching NSE output
            calls: List[Dict] = []
            puts: List[Dict] = []
            total_call_oi = 0
            total_put_oi = 0
            max_call_oi = 0
            max_put_oi = 0

            for key, meta in strike_to_inst.items():
                q = quotes.get(key, {})
                if not q:
                    continue
                item = {
                    'strike': meta['strike'],
                    'oi': int(q.get('oi', 0) or 0),
                    'chg_oi': 0,  # Kite quote doesn't provide change in OI
                    'volume': int(q.get('volume', 0) or 0),
                    'iv': 0.0,    # Kite doesn't provide IV in quote
                    'ltp': float(q.get('last_price', 0) or 0),
                    'bid': float(q.get('depth', {}).get('buy', [{}])[0].get('price', 0) or 0),
                    'ask': float(q.get('depth', {}).get('sell', [{}])[0].get('price', 0) or 0),
                }
                if meta['type'] == 'CE':
                    calls.append(item)
                    total_call_oi += item['oi']
                    if item['oi'] > max_call_oi:
                        max_call_oi = item['oi']
                else:
                    puts.append(item)
                    total_put_oi += item['oi']
                    if item['oi'] > max_put_oi:
                        max_put_oi = item['oi']

            calls.sort(key=lambda x: x['strike'])
            puts.sort(key=lambda x: x['strike'])

            pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi else 1.0
            atm = min((c['strike'] for c in calls), key=lambda x: abs(x - underlying)) if calls else 0

            return {
                'symbol': sym_upper,
                'underlying': underlying,
                'atm_strike': atm,
                'expiry_dates': [nearest_expiry.isoformat()],
                'calls': calls,
                'puts': puts,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'max_call_oi_strike': max(calls, key=lambda x: x['oi'])['strike'] if calls else 0,
                'max_put_oi_strike': max(puts, key=lambda x: x['oi'])['strike'] if puts else 0,
                'pcr': pcr,
                'max_pain': 0.0,  # Would need full chain for accurate max pain
                'sentiment': 'Bullish' if pcr > 1.2 else ('Bearish' if pcr < 0.7 else 'Neutral'),
                'timestamp': time.time(),
                'source': 'kite',
            }
        except Exception as e:
            logger.warning(f"Kite option chain build error [{symbol}]: {e}")
            return None

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
