"""
MarketMind AI — Multi-Asset Fetcher (W4.2)
MCX commodities via Kite, forex via NSE currency derivatives,
crypto via CoinGecko free tier, and cross-asset rolling correlations.
"""
import time
import logging
import threading
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── NSE session headers (same as macro_fetcher) ─────────────────────────────────
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Origin': 'https://www.nseindia.com',
    'Referer': 'https://www.nseindia.com/',
    'X-Requested-With': 'XMLHttpRequest',
}

# ── MCX commodity symbols (Kite trading symbol format) ─────────────────────────
# These are the generic/continuous symbols Kite uses for near-month futures.
# When Kite doesn't resolve them, we fall back to instrument-list lookup.
MCX_COMMODITIES = {
    'GOLD':     {'kite': 'MCX:GOLDM',     'name': 'Gold Mini Futures',      'unit': '₹/10g'},
    'SILVER':   {'kite': 'MCX:SILVERM',   'name': 'Silver Mini Futures',    'unit': '₹/kg'},
    'CRUDEOIL': {'kite': 'MCX:CRUDEOILM', 'name': 'Crude Oil Mini Futures', 'unit': '₹/barrel'},
    'ZINC':     {'kite': 'MCX:ZINCM',     'name': 'Zinc Mini Futures',      'unit': '₹/kg'},
}

# ── Forex symbols via NSE currency derivatives ─────────────────────────────────
FOREX_SYMBOLS = {
    'USDINR': {'nse': 'USDINR', 'name': 'USD/INR', 'base': 'USD', 'quote': 'INR'},
    'EURINR': {'nse': 'EURINR', 'name': 'EUR/INR', 'base': 'EUR', 'quote': 'INR'},
}

# ── Crypto via CoinGecko free tier ─────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
CRYPTO_ASSETS = {
    'BTC':  {'id': 'bitcoin',  'name': 'Bitcoin',  'pair': 'BTC/INR'},
    'ETH':  {'id': 'ethereum', 'name': 'Ethereum',  'pair': 'ETH/INR'},
}

# CoinGecko free tier: ~10-30 calls/min. 5-min cache to stay well under.
COINGECKO_CACHE_TTL = 300


class MultiAssetFetcher:
    """Fetches commodity, forex, crypto prices and cross-asset correlations."""

    def __init__(self):
        self._nse_session: Optional[requests.Session] = None
        self._nse_session_time: float = 0.0
        self._cache: Dict = {}
        self._cache_ttl = 600  # 10 min
        self._cg_cache: Dict = {}
        self._cg_last_call: float = 0.0
        self._cg_lock = threading.Lock()
        # MCX symbols resolved once from Kite instrument list; cached for session
        self._mcx_resolved: Optional[Dict[str, str]] = None
        self._mcx_resolved_time: float = 0.0

    # ── NSE session (shared with macro pattern) ────────────────────────────────
    def _get_nse_session(self) -> requests.Session:
        now = time.time()
        if self._nse_session is None or (now - self._nse_session_time) > 600:
            s = requests.Session()
            s.headers.update(HEADERS)
            try:
                s.get('https://www.nseindia.com', timeout=10)
            except Exception as e:
                logger.debug(f"MultiAsset NSE session warmup: {e}")
            self._nse_session = s
            self._nse_session_time = now
        return self._nse_session

    def _nse_get(self, path: str) -> Optional[dict]:
        for _ in range(2):
            try:
                s = self._get_nse_session()
                r = s.get(f"https://www.nseindia.com/api/{path}", timeout=12)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.debug(f"NSE multiasset get error: {e}")
                self._nse_session = None
        return None

    # ── MCX commodity resolution ───────────────────────────────────────────────
    def _resolve_mcx_symbols(self, kite_client) -> Dict[str, str]:
        """Map commodity name → Kite trading symbol using instrument list.
        Cached for the session (instruments rarely change intraday)."""
        now = time.time()
        if self._mcx_resolved is not None and (now - self._mcx_resolved_time) < 3600:
            return self._mcx_resolved

        resolved = {}
        try:
            instruments = kite_client.get_instruments('MCX')
            if not instruments:
                logger.debug("No MCX instruments returned from Kite")
                return resolved

            # Group instruments by name root, pick the most liquid near-month future
            by_name: Dict[str, list] = defaultdict(list)
            for inst in instruments:
                name = inst.get('name', '') or inst.get('tradingsymbol', '')
                ts = inst.get('tradingsymbol', '')
                # Only consider FUT contracts (not OPT)
                if 'FUT' in ts and ts.endswith('FUT'):
                    by_name[name].append(inst)

            # For each target commodity, find the best matching instrument
            target_roots = {'GOLD', 'GOLDM', 'SILVER', 'SILVERM', 'CRUDEOIL', 'CRUDEOILM', 'ZINC', 'ZINCM'}
            for name_root in target_roots:
                matches = []
                for inst_name, insts in by_name.items():
                    if inst_name.upper().startswith(name_root):
                        matches.extend(insts)
                if matches:
                    # Pick the one with the nearest expiry (largest lot size as proxy for liquidity)
                    best = max(matches, key=lambda x: x.get('lot_size', 0))
                    resolved[name_root] = f"MCX:{best['tradingsymbol']}"

            self._mcx_resolved = resolved
            self._mcx_resolved_time = now
            logger.debug(f"Resolved MCX symbols: {resolved}")
        except Exception as e:
            logger.debug(f"MCX symbol resolution error: {e}")
        return resolved

    # ── MCX commodities ────────────────────────────────────────────────────────
    def get_commodities(self, kite_client) -> Dict:
        """Get live MCX commodity prices via Kite."""
        cache_key = 'commodities'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']

        result = {}
        if not kite_client or not kite_client.is_connected:
            for slug, info in MCX_COMMODITIES.items():
                result[slug.lower()] = {
                    'name': info['name'], 'unit': info['unit'],
                    'price': None, 'change_pct': None,
                    'source': 'unavailable', 'error': 'kite_not_connected',
                }
            return result

        # Try generic symbols first, fall back to resolved symbols
        symbols_to_try = [info['kite'] for info in MCX_COMMODITIES.values()]
        try:
            ltp_data = kite_client.get_ltp(symbols_to_try)
        except Exception as e:
            logger.debug(f"Kite LTP error for commodities: {e}")
            ltp_data = {}

        # If generic symbols returned nothing, try instrument-list resolution
        if not ltp_data:
            resolved = self._resolve_mcx_symbols(kite_client)
            if resolved:
                resolved_symbols = list(resolved.values())
                try:
                    ltp_data = kite_client.get_ltp(resolved_symbols)
                except Exception as e:
                    logger.debug(f"Kite LTP error for resolved commodities: {e}")

        # Also try to get previous day close via OHLC for daily change
        ohlc_data = {}
        try:
            ohlc_data = kite_client.get_ohlc(symbols_to_try)
        except Exception:
            pass

        for slug, info in MCX_COMMODITIES.items():
            kite_sym = info['kite']
            ltp_info = ltp_data.get(kite_sym, {}) if ltp_data else {}
            price = float(ltp_info.get('last_price', 0)) if ltp_info else 0.0

            # If generic symbol failed, try resolved
            if price == 0.0:
                resolved = self._mcx_resolved or {}
                resolved_sym = None
                for root, rsym in resolved.items():
                    if root in (slug, slug.upper(), info['name'].split()[0].upper()):
                        resolved_sym = rsym
                        break
                if resolved_sym and resolved_sym in (ltp_data or {}):
                    ltp_info = ltp_data[resolved_sym]
                    price = float(ltp_info.get('last_price', 0))

            ohlc_info = ohlc_data.get(kite_sym, {}) if ohlc_data else {}
            prev_close = float(ohlc_info.get('ohlc', {}).get('close', 0) or
                               ltp_info.get('ohlc', {}).get('close', 0) if isinstance(ltp_info, dict) else 0)

            change_pct = None
            if price and prev_close and prev_close > 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)

            result[slug.lower()] = {
                'name': info['name'], 'unit': info['unit'],
                'price': price if price > 0 else None,
                'prev_close': prev_close if prev_close > 0 else None,
                'change_pct': change_pct,
                'source': 'kite' if price > 0 else 'unavailable',
            }

        self._cache[cache_key] = {'data': result, 'ts': time.time()}
        return result

    # ── Forex (USD/INR + EUR/INR) ──────────────────────────────────────────────
    def get_forex(self, kite_client=None) -> Dict:
        """Get USD/INR and EUR/INR — Kite CDS futures first, NSE derivatives fallback.

        Kite serves CDS instrument quotes even outside market hours (last close
        + prev close), whereas the NSE ``quote-derivative`` endpoint returns
        an empty ``quoteVFO`` until the CDS session is open. Trying Kite first
        gives a non-zero ``change_pct`` whenever the user is authenticated.
        """
        result = {}
        for sym, info in FOREX_SYMBOLS.items():
            row = None
            if kite_client and getattr(kite_client, 'is_connected', False):
                row = self._fetch_forex_kite(info, kite_client)
            if not row:
                row = self._fetch_forex_single(info)
            result[sym.lower()] = row
        return result

    def _fetch_forex_kite(self, info: dict, kite_client) -> Optional[Dict]:
        """Resolve nearest-month USDINR/EURINR future via Kite CDS instruments.

        Returns None on any failure so the caller can fall through to the NSE
        path. We pick the soonest non-past expiry, then read its OHLC for
        last-price + previous-day close.
        """
        sym = info['nse']
        cache_key = f"forex_kite_{sym}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']

        try:
            from datetime import date as _date
            insts = kite_client.get_instruments('CDS') or []
            today = _date.today()
            futs = [
                i for i in insts
                if i.get('name') == sym
                and i.get('instrument_type') == 'FUT'
                and isinstance(i.get('expiry'), _date)
                and (i['expiry'] - today).days >= 0
            ]
            if not futs:
                return None
            futs.sort(key=lambda i: i['expiry'])
            tradingsymbol = futs[0]['tradingsymbol']
            ohlc = kite_client.get_ohlc([f"CDS:{tradingsymbol}"]) or {}
            quote = ohlc.get(f"CDS:{tradingsymbol}") or {}
            ltp = float(quote.get('last_price') or 0)
            prev = float((quote.get('ohlc') or {}).get('close') or 0)
            # CDS hours are 09:00–17:00 IST. Outside that window Kite returns
            # last_price=0 but the OHLC envelope still carries the previous
            # session's close. Use that as the displayable rate so the macro
            # card shows the most recent quote instead of falling through to
            # the (broken) NSE estimate.
            if not ltp and prev:
                ltp = prev
            if not ltp:
                return None
            chg = round(ltp - prev, 4) if prev else 0.0
            row = {
                'symbol': info['name'], 'base': info['base'], 'quote': info['quote'],
                'rate': ltp, 'prev': prev,
                'change': chg,
                'change_pct': round(chg / prev * 100, 3) if prev else 0,
                'source': 'kite',
                'expiry': futs[0]['expiry'].isoformat(),
            }
            self._cache[cache_key] = {'data': row, 'ts': time.time()}
            return row
        except Exception as e:
            logger.debug(f"Kite forex {sym} fetch error: {e}")
            return None

    def _fetch_forex_single(self, info: dict) -> Dict:
        sym = info['nse']
        cache_key = f"forex_{sym}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']

        fallback_rates = {'USDINR': 84.5, 'EURINR': 91.0}
        fallback = fallback_rates.get(sym, 85.0)
        try:
            data = self._nse_get(f"quote-derivative?symbol={sym}")
            if data:
                # NSE shape: top-level `underlyingValue` (str) is the spot
                # during CDS hours; outside hours it returns "-". The legacy
                # `quoteVFO` key does not exist on this endpoint. Try
                # `underlyingValue` first, then the first front-month future
                # under `stocks[0].metadata.lastPrice`.
                ltp = 0.0
                prev = 0.0
                uv = data.get('underlyingValue')
                try:
                    if uv not in (None, '', '-'):
                        ltp = float(uv)
                except (TypeError, ValueError):
                    ltp = 0.0
                stocks = data.get('stocks') or []
                if not ltp and stocks:
                    md = (stocks[0] or {}).get('metadata') or {}
                    try:
                        ltp = float(md.get('lastPrice') or 0)
                    except (TypeError, ValueError):
                        ltp = 0.0
                    try:
                        prev = float(md.get('prevClose') or 0)
                    except (TypeError, ValueError):
                        prev = 0.0
                if ltp:
                    if not prev:
                        prev = ltp
                    chg = round(ltp - prev, 4)
                    result = {
                        'symbol': info['name'], 'base': info['base'], 'quote': info['quote'],
                        'rate': ltp, 'prev': prev,
                        'change': chg,
                        'change_pct': round(chg / prev * 100, 3) if prev else 0,
                        'source': 'nse',
                    }
                    self._cache[cache_key] = {'data': result, 'ts': time.time()}
                    return result
        except Exception as e:
            logger.debug(f"Forex {sym} fetch error: {e}")
        result = {
            'symbol': info['name'], 'base': info['base'], 'quote': info['quote'],
            'rate': fallback, 'change_pct': 0, 'source': 'estimate',
        }
        return result

    # ── Crypto via CoinGecko ───────────────────────────────────────────────────
    def _coingecko_get(self, path: str, cache_key: str) -> Optional[dict]:
        """Rate-limited CoinGecko GET with simple in-memory cache."""
        cached = self._cg_cache.get(cache_key)
        if cached and time.time() - cached['ts'] < COINGECKO_CACHE_TTL:
            return cached['data']

        with self._cg_lock:
            elapsed = time.time() - self._cg_last_call
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)

        try:
            url = f"{COINGECKO_BASE}/{path}"
            r = requests.get(url, timeout=15)
            self._cg_last_call = time.time()
            if r.status_code == 429:
                logger.warning("CoinGecko rate limited")
                return None
            r.raise_for_status()
            data = r.json()
            self._cg_cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
        except Exception as e:
            logger.debug(f"CoinGecko error: {e}")
            return None

    def get_crypto(self) -> Dict:
        """Get BTC/INR and ETH/INR live prices from CoinGecko."""
        cache_key = 'crypto_prices'
        cached = self._cg_cache.get(cache_key)
        if cached and time.time() - cached['ts'] < COINGECKO_CACHE_TTL:
            return cached['data']

        result = {}
        try:
            ids = ','.join(info['id'] for info in CRYPTO_ASSETS.values())
            data = self._coingecko_get(
                f"simple/price?ids={ids}&vs_currencies=inr&include_24hr_change=true",
                'crypto_price_raw',
            )
            for slug, info in CRYPTO_ASSETS.items():
                coin_data = (data or {}).get(info['id'], {})
                price = coin_data.get('inr')
                change_pct = coin_data.get('inr_24h_change')
                result[slug.lower()] = {
                    'name': info['name'], 'pair': info['pair'],
                    'price_inr': price,
                    'change_24h_pct': round(change_pct, 2) if change_pct is not None else None,
                    'source': 'coingecko' if price else 'unavailable',
                }
        except Exception as e:
            logger.debug(f"Crypto price error: {e}")
            for slug, info in CRYPTO_ASSETS.items():
                result[slug.lower()] = {
                    'name': info['name'], 'pair': info['pair'],
                    'price_inr': None, 'change_24h_pct': None, 'source': 'error',
                }

        self._cg_cache[cache_key] = {'data': result, 'ts': time.time()}
        return result

    def get_crypto_history(self, coin_id: str, days: int = 90) -> List[Dict]:
        """Get daily OHLC history for a crypto asset from CoinGecko."""
        cache_key = f"cg_hist_{coin_id}_{days}"
        data = self._coingecko_get(
            f"coins/{coin_id}/market_chart?vs_currency=inr&days={days}",
            cache_key,
        )
        if not data or 'prices' not in data:
            return []
        # CoinGecko returns [timestamp_ms, price] pairs
        out = []
        for ts_ms, price in data['prices']:
            out.append({
                'date': datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d'),
                'price': price,
            })
        return out

    # ── Cross-asset correlations ───────────────────────────────────────────────
    def get_cross_asset_correlations(
        self, kite_client, price_fetcher, windows: List[int] = None
    ) -> Dict:
        """Compute rolling Pearson correlations across asset classes.
        windows: list of day-counts, e.g. [30, 90]. Default [30, 90].
        Returns {window_days: {correlation_matrix: {...}, asset_prices: {...}}}.
        """
        if windows is None:
            windows = [30, 90]

        # Collect 90-day daily close series for each asset
        max_window = max(windows)
        series = {}
        labels = {}

        # Nifty 500 (equity proxy)
        try:
            df = price_fetcher.get_historical_data('NIFTY500', days=max_window + 5)
            if df is not None and len(df) >= 30:
                series['nifty500'] = df['close'].astype(float)
                labels['nifty500'] = 'Nifty 500'
        except Exception as e:
            logger.debug(f"Nifty history for correlation: {e}")

        # MCX Gold
        if kite_client and kite_client.is_connected:
            try:
                gold_sym = MCX_COMMODITIES['GOLD']['kite']
                df = kite_client.get_historical_data(
                    gold_sym, interval='day', days=max_window + 5)
                if hasattr(df, 'get'):
                    pass
                if df is not None and len(df) >= 30:
                    series['gold'] = df['close'].astype(float)
                    labels['gold'] = 'Gold (MCX)'
            except Exception as e:
                logger.debug(f"Gold history for correlation: {e}")

        # Crypto BTC
        try:
            btc_hist = self.get_crypto_history('bitcoin', days=max_window + 5)
            if btc_hist and len(btc_hist) >= 30:
                import pandas as pd
                btc_df = pd.DataFrame(btc_hist)
                btc_df['date'] = pd.to_datetime(btc_df['date'])
                btc_df = btc_df.set_index('date')
                series['btc'] = btc_df['price'].astype(float)
                labels['btc'] = 'Bitcoin (INR)'
        except Exception as e:
            logger.debug(f"BTC history for correlation: {e}")

        # USD/INR — use the current rate direction as a proxy;
        # NSE doesn't provide easy historical forex via the public API.
        # We note this gap; forex is excluded from the correlation matrix for now.

        # Compute daily returns for each series
        returns = {}
        for key, s in series.items():
            ret = s.pct_change().dropna()
            if len(ret) >= 30:
                returns[key] = ret

        # Compute correlation matrices per window
        result = {}
        for w in windows:
            window_returns = {k: v.iloc[-w:] for k, v in returns.items() if len(v) >= w}
            if len(window_returns) < 2:
                result[str(w)] = {'correlation_matrix': {}, 'asset_labels': labels,
                                  'note': 'insufficient data for correlation'}
                continue

            import pandas as pd
            combined = pd.DataFrame(window_returns)
            corr = combined.corr()

            matrix = {}
            for col in corr.columns:
                matrix[str(col)] = {
                    str(idx): (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                               else round(float(v), 4))
                    for idx, v in corr[col].items()
                }
            result[str(w)] = {
                'correlation_matrix': matrix,
                'asset_labels': labels,
                'data_points': {k: len(v) for k, v in window_returns.items()},
            }

        return result

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def get_all(self, kite_client=None, price_fetcher=None) -> Dict:
        """Fetch all multi-asset data at once."""
        return {
            'commodities': self.get_commodities(kite_client) if kite_client else {},
            'forex': self.get_forex(kite_client),
            'crypto': self.get_crypto(),
            'timestamp': datetime.now().isoformat(),
        }


# ── Singleton ───────────────────────────────────────────────────────────────────
_fetcher: Optional[MultiAssetFetcher] = None


def get_multiasset_fetcher() -> MultiAssetFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = MultiAssetFetcher()
    return _fetcher
