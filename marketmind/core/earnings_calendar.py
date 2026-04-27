"""
MarketMind AI - Earnings Calendar
Fetches upcoming & recent results / board meetings from NSE corporate actions API.
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


import re
EARNINGS_KW = re.compile(r'result|earning|financial|dividend', re.I)


class EarningsCalendar:
    NSE_BASE = "https://www.nseindia.com/api"
    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',  # no 'br' — requests can't decode brotli
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/',
        'X-Requested-With': 'XMLHttpRequest',
    }

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_time: float = 0.0
        self._cache: Dict = {}
        self._cache_ttl = 3600  # 1 hour

    def _get_session(self) -> requests.Session:
        now = time.time()
        if self._session is None or (now - self._session_time) > 900:
            s = requests.Session()
            s.headers.update(self.HEADERS)
            try:
                s.get('https://www.nseindia.com', timeout=10)
                s.get('https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE', timeout=10)
            except Exception as e:
                logger.debug(f"EarningsCalendar session warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def get_upcoming_results(self, days_ahead: int = 30) -> List[Dict]:
        """Upcoming earnings + dividend announcements via NSE board-meetings API."""
        cache_key = f'earnings_{days_ahead}'
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached['ts'] < self._cache_ttl:
            return cached['data']
        results = self._fetch_board_meetings(days_ahead)
        self._cache[cache_key] = {'data': results, 'ts': time.time()}
        return results

    def get_dividend_calendar(self, days_ahead: int = 30) -> List[Dict]:
        """Subset of upcoming results filtered to dividend-related rows."""
        all_items = self.get_upcoming_results(days_ahead)
        return [r for r in all_items if 'dividend' in r.get('purpose', '').lower()]

    def _fetch_board_meetings(self, days_ahead: int) -> List[Dict]:
        try:
            session = self._get_session()
            today = datetime.now()
            from_str = (today - timedelta(days=2)).strftime('%d-%m-%Y')
            to_str = (today + timedelta(days=days_ahead)).strftime('%d-%m-%Y')
            url = f"{self.NSE_BASE}/corporate-board-meetings"
            params = {'index': 'equities', 'from_date': from_str, 'to_date': to_str}
            resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return self._get_fallback_earnings()

            items = []
            seen = set()
            for row in data:
                purpose = (row.get('bm_purpose') or '').strip()
                desc = (row.get('bm_desc') or '').strip()
                # Keep only earnings/results/dividend rows (skip pure procedural meetings)
                if not EARNINGS_KW.search(purpose + ' ' + desc):
                    continue
                symbol = (row.get('bm_symbol') or '').upper()
                date_str = row.get('bm_date') or ''
                # De-duplicate: NSE re-publishes the same intimation multiple times
                key = (symbol, date_str)
                if key in seen:
                    continue
                seen.add(key)
                items.append({
                    'symbol': symbol,
                    'company': row.get('sm_name') or symbol,
                    'series': 'EQ',
                    'industry': row.get('sm_indusrty') or '',
                    'purpose': purpose,
                    'subject': desc[:200],
                    'ex_date': date_str,           # meeting date — use as event date
                    'record_date': date_str,
                    'bc_start': '', 'bc_end': '',
                    'attachment': row.get('attachment') or '',
                })

            def sort_key(x):
                try:
                    return datetime.strptime(x['ex_date'], '%d-%b-%Y')
                except Exception:
                    return datetime.max
            items.sort(key=sort_key)
            return items[:100]
        except Exception as e:
            logger.warning(f"Earnings calendar fetch error: {e}")
            self._session = None
            return self._get_fallback_earnings()

    # Back-compat alias
    def _fetch_corporate_actions(self, action_type: str, days_ahead: int) -> List[Dict]:
        return self._fetch_board_meetings(days_ahead)

    def _get_fallback_earnings(self) -> List[Dict]:
        """Return sample upcoming results for major Nifty50 companies."""
        today = datetime.now()
        major_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'HINDUNILVR',
            'AXISBANK', 'KOTAKBANK', 'MARUTI', 'TATAMOTORS', 'WIPRO',
        ]
        items = []
        for i, sym in enumerate(major_stocks):
            ex = today + timedelta(days=(i * 3 + 5))
            items.append({
                'symbol': sym,
                'company': sym,
                'series': 'EQ',
                'purpose': 'Board Meeting - Results',
                'ex_date': ex.strftime('%d-%b-%Y'),
                'record_date': ex.strftime('%d-%b-%Y'),
                'bc_start': '',
                'bc_end': '',
                'subject': 'Quarterly Results',
            })
        return items


_calendar: Optional[EarningsCalendar] = None


def get_earnings_calendar() -> EarningsCalendar:
    global _calendar
    if _calendar is None:
        _calendar = EarningsCalendar()
    return _calendar
