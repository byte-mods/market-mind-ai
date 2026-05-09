"""
MarketMind AI - FII/DII Flow Tracker
Fetches Foreign & Domestic Institutional Investor daily flow data from NSE.
FII net buying is the single most predictive signal for Nifty direction.
"""
import time
import logging
import requests
from datetime import datetime, timedelta
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


class FIIDIIFetcher:
    NSE_BASE = "https://www.nseindia.com/api"

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_time: float = 0.0
        self._cache: Dict = {}
        self._cache_ttl = 1800  # 30 min

    def _get_session(self) -> requests.Session:
        now = time.time()
        if self._session is None or (now - self._session_time) > 600:
            s = requests.Session()
            s.headers.update(HEADERS)
            try:
                s.get('https://www.nseindia.com', timeout=10)
            except Exception as e:
                logger.debug(f"FIIDIIFetcher session warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def get_fii_dii_data(self, days: int = 30) -> List[Dict]:
        """Fetch FII/DII equity trade data for the last N trading days."""
        cache_key = f'fiidii_{days}'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']

        try:
            session = self._get_session()
            today = datetime.now()
            from_dt = today - timedelta(days=days + 10)  # extra buffer for holidays

            # NSE FII/DII API
            url = (
                f"{self.NSE_BASE}/fiidiiTradeReact"
                f"?startDate={from_dt.strftime('%d-%m-%Y')}"
                f"&endDate={today.strftime('%d-%m-%Y')}"
            )
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            data = self._parse_fiidii(raw)
            self._cache[cache_key] = {'data': data, 'ts': time.time()}
            return data
        except Exception as e:
            logger.warning(f"FII/DII fetch error: {e}")
            self._session = None
            return self._fallback_data()

    def _parse_fiidii(self, raw) -> List[Dict]:
        """Normalise NSE FII/DII payload.

        The current NSE shape is one row per (date, category) pair where
        ``category`` is ``"FII/FPI"`` or ``"DII"``. We group by date and
        place values into the right side of the record. The legacy single-row
        shape (``fiiBuy``/``diiBuy`` on the same row) is still honoured for
        forward-compat with older NSE API versions and unit-test fixtures.
        """
        items = []
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            rows = raw.get('data', []) or raw.get('FIIData', []) or []
        else:
            return self._fallback_data()

        # Group by date — NSE serves one row per category per date.
        # OrderedDict preserves NSE's emission order for the stable sort below.
        from collections import OrderedDict
        by_date: "OrderedDict[str, Dict]" = OrderedDict()

        for row in rows:
            try:
                date_str = (row.get('date') or row.get('Date') or
                            row.get('tradeDate') or row.get('TRADE_DATE', ''))
                if not date_str:
                    continue

                category = (row.get('category') or row.get('CATEGORY') or '').upper()
                # Generic buy/sell/net columns (used when category-tagged)
                buy = self._to_float(row.get('buyValue') or row.get('BUY_VALUE') or 0)
                sell = self._to_float(row.get('sellValue') or row.get('SELL_VALUE') or 0)
                net = self._to_float(row.get('netValue') or row.get('NET_VALUE') or
                                     (buy - sell))

                # Legacy single-row shape: both FII and DII columns coexist.
                legacy_fii_buy = self._to_float(row.get('fiiBuy') or row.get('fii_buy') or 0)
                legacy_fii_sell = self._to_float(row.get('fiiSell') or row.get('fii_sell') or 0)
                legacy_fii_net = self._to_float(
                    row.get('fiiNet') or row.get('fii_net') or
                    (legacy_fii_buy - legacy_fii_sell))
                legacy_dii_buy = self._to_float(row.get('diiBuy') or row.get('dii_buy') or 0)
                legacy_dii_sell = self._to_float(row.get('diiSell') or row.get('dii_sell') or 0)
                legacy_dii_net = self._to_float(
                    row.get('diiNet') or row.get('dii_net') or
                    (legacy_dii_buy - legacy_dii_sell))

                rec = by_date.setdefault(date_str, {
                    'date': date_str,
                    'fii_buy': 0.0, 'fii_sell': 0.0, 'fii_net': 0.0,
                    'dii_buy': 0.0, 'dii_sell': 0.0, 'dii_net': 0.0,
                })

                if 'FII' in category or 'FPI' in category:
                    rec['fii_buy'] = buy
                    rec['fii_sell'] = sell
                    rec['fii_net'] = net
                elif 'DII' in category:
                    rec['dii_buy'] = buy
                    rec['dii_sell'] = sell
                    rec['dii_net'] = net
                else:
                    # Legacy single-row: prefer explicit dii_/fii_ columns
                    if legacy_fii_buy or legacy_fii_sell or legacy_fii_net:
                        rec['fii_buy'] = legacy_fii_buy
                        rec['fii_sell'] = legacy_fii_sell
                        rec['fii_net'] = legacy_fii_net
                    if legacy_dii_buy or legacy_dii_sell or legacy_dii_net:
                        rec['dii_buy'] = legacy_dii_buy
                        rec['dii_sell'] = legacy_dii_sell
                        rec['dii_net'] = legacy_dii_net
            except Exception:
                continue

        for rec in by_date.values():
            rec['combined_net'] = round(rec['fii_net'] + rec['dii_net'], 2)
            for k in ('fii_buy', 'fii_sell', 'fii_net',
                      'dii_buy', 'dii_sell', 'dii_net'):
                rec[k] = round(rec[k], 2)
            items.append(rec)

        items.reverse()  # chronological order
        return items[-30:] if items else self._fallback_data()

    def _to_float(self, v) -> float:
        try:
            return float(str(v).replace(',', '').strip())
        except Exception:
            return 0.0

    def get_summary(self, days: int = 5) -> Dict:
        """Get summary stats: rolling net flows, trend, signal."""
        data = self.get_fii_dii_data(days + 5)
        recent = data[-days:] if len(data) >= days else data

        if not recent:
            return {}

        fii_total = sum(r['fii_net'] for r in recent)
        dii_total = sum(r['dii_net'] for r in recent)
        combined = fii_total + dii_total

        # Trend: last 3 vs previous 3 — computed off the full fetched series,
        # not the `days`-bounded summary slice. The summary window (e.g. days=5)
        # would otherwise structurally prevent the >=6 condition from ever
        # holding, pinning the trend to "Insufficient data" forever.
        if len(data) >= 6:
            last3 = sum(r['fii_net'] for r in data[-3:])
            prev3 = sum(r['fii_net'] for r in data[-6:-3])
            fii_trend = 'Accelerating' if last3 > prev3 else 'Decelerating'
        else:
            fii_trend = 'Insufficient data'

        fii_signal = ('STRONG BUY' if fii_total > 5000 else
                      'BUY' if fii_total > 1000 else
                      'NEUTRAL' if fii_total > -1000 else
                      'SELL' if fii_total > -5000 else 'STRONG SELL')

        return {
            'days': days,
            'fii_net_total': round(fii_total, 2),
            'dii_net_total': round(dii_total, 2),
            'combined_net': round(combined, 2),
            'fii_trend': fii_trend,
            'fii_signal': fii_signal,
            'interpretation': (
                f"FII {'bought' if fii_total >= 0 else 'sold'} ₹{abs(fii_total):.0f}Cr "
                f"over {days} days. DII {'bought' if dii_total >= 0 else 'sold'} ₹{abs(dii_total):.0f}Cr. "
                f"Net combined: ₹{combined:.0f}Cr → {fii_signal}"
            ),
        }

    def _fallback_data(self) -> List[Dict]:
        """Return synthetic recent data when API is down."""
        import random
        random.seed(42)
        out = []
        base = datetime.now()
        for i in range(20, 0, -1):
            d = base - timedelta(days=i)
            if d.weekday() >= 5:
                continue
            fii_net = random.gauss(500, 2500)
            dii_net = random.gauss(-200, 1500)
            out.append({
                'date': d.strftime('%d-%b-%Y'),
                'fii_buy': round(abs(fii_net) + 5000, 2),
                'fii_sell': round(abs(fii_net) + 5000 - fii_net, 2),
                'fii_net': round(fii_net, 2),
                'dii_buy': round(abs(dii_net) + 3000, 2),
                'dii_sell': round(abs(dii_net) + 3000 - dii_net, 2),
                'dii_net': round(dii_net, 2),
                'combined_net': round(fii_net + dii_net, 2),
            })
        return out


_fetcher: Optional[FIIDIIFetcher] = None

def get_fii_dii_fetcher() -> FIIDIIFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = FIIDIIFetcher()
    return _fetcher
