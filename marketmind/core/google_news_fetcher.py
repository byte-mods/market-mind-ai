"""
MarketMind AI - Google News Fetcher
Fetches financial news for all Indian market sectors via Google News RSS.
Replaces yfinance/DuckDuckGo news scraping.
"""

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time
import threading
import urllib.parse

logger = logging.getLogger(__name__)


class GoogleNewsFetcher:
    """Fetches financial news from Google News RSS for all sectors."""

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

    # Per-sector RSS query strings
    SECTOR_QUERIES: Dict[str, str] = {
        'IT': 'Indian IT software technology TCS Infosys Wipro HCL stocks NSE',
        'Banking': 'Indian banking HDFC ICICI SBI Kotak RBI credit policy NSE stocks',
        'Auto': 'Indian automobile auto EV Maruti Tata Motors Mahindra Bajaj NSE stocks',
        'Pharma': 'Indian pharma pharmaceutical Sun Pharma Dr Reddy USFDA Cipla NSE stocks',
        'FMCG': 'Indian FMCG consumer goods HUL ITC Nestle Dabur Marico NSE stocks',
        'Metal': 'Indian metal steel aluminum coal Tata Steel Hindalco Vedanta NSE stocks',
        'Energy': 'Indian energy oil gas ONGC Reliance IOC BPCL power NSE stocks',
        'Realty': 'Indian real estate realty DLF Godrej property housing NSE stocks',
        'Finance': 'Indian NBFC finance insurance Bajaj Finance SBI Life LIC NSE stocks',
        'Market': 'Indian stock market Nifty Sensex NSE BSE economy today',
        'Global': 'global markets US Fed interest rates China economy impact India rupee',
        'Geopolitical': 'geopolitical war sanctions crude oil OPEC India market impact',
    }

    # Stock-specific query template
    STOCK_QUERY_TPL = '{symbol} NSE India stock results earnings quarterly'

    def __init__(self):
        self.cache: Dict = {}
        self.cache_duration = 600   # 10 minutes for news
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'application/rss+xml,application/xml,text/xml,*/*',
            'Accept-Language': 'en-IN,en;q=0.9',
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_market_news(self, max_per_sector: int = 6) -> List[Dict]:
        """Fetch news for every sector; deduplicate by URL."""
        all_news: List[Dict] = []
        seen_urls: set = set()

        for sector, query in self.SECTOR_QUERIES.items():
            try:
                items = self._fetch_rss(query, max_per_sector)
                for item in items:
                    url = item.get('url', '')
                    if url and url not in seen_urls:
                        item['sectors'] = [sector]
                        all_news.append(item)
                        seen_urls.add(url)
                time.sleep(0.4)   # be respectful to Google
            except Exception as e:
                logger.error(f"Google News error for sector '{sector}': {e}")

        return all_news

    def fetch_sector_news(self, sector: str, max_items: int = 10) -> List[Dict]:
        """Fetch news for a single sector (with caching)."""
        cache_key = f"sector_{sector}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_duration:
                return cached['data']

        query = self.SECTOR_QUERIES.get(sector, f'Indian {sector} stock market NSE')
        items = self._fetch_rss(query, max_items)
        for item in items:
            item.setdefault('sectors', [sector])

        with self.lock:
            self.cache[cache_key] = {'data': items, 'timestamp': time.time()}
        return items

    def fetch_stock_news(self, symbol: str, max_items: int = 5) -> List[Dict]:
        """Fetch news for a specific stock symbol (with caching)."""
        cache_key = f"stock_{symbol}"
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.time() - cached['timestamp'] < self.cache_duration:
                return cached['data']

        query = self.STOCK_QUERY_TPL.format(symbol=symbol)
        items = self._fetch_rss(query, max_items)

        with self.lock:
            self.cache[cache_key] = {'data': items, 'timestamp': time.time()}
        return items

    def get_market_news(self, days: int = 1) -> List[Dict]:
        """Alias used by legacy NewsFetcher callers."""
        return self.fetch_all_market_news(max_per_sector=5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_rss(self, query: str, max_items: int) -> List[Dict]:
        """Download and parse a Google News RSS feed."""
        try:
            params = {
                'q': query,
                'hl': 'en-IN',
                'gl': 'IN',
                'ceid': 'IN:en',
            }
            url = f"{self.GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"
            resp = self.session.get(url, timeout=12)
            resp.raise_for_status()
            return self._parse_rss_xml(resp.content, max_items)
        except Exception as e:
            logger.error(f"RSS fetch error for '{query[:60]}': {e}")
            return []

    def _parse_rss_xml(self, content: bytes, max_items: int) -> List[Dict]:
        """Parse RSS XML content into structured news dicts."""
        items: List[Dict] = []
        try:
            root = ET.fromstring(content)
            channel = root.find('channel')
            if channel is None:
                return items

            for entry in channel.findall('item')[:max_items]:
                title = (entry.findtext('title') or '').strip()
                link = (entry.findtext('link') or '').strip()
                pub_date = (entry.findtext('pubDate') or '').strip()
                description = (entry.findtext('description') or '').strip()

                source_el = entry.find('source')
                source = source_el.text.strip() if source_el is not None else 'Google News'

                # Strip HTML from description
                if description:
                    description = BeautifulSoup(description, 'html.parser').get_text(' ', strip=True)

                # Parse publish date
                try:
                    pub_dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                    published_at = pub_dt.isoformat()
                except Exception:
                    published_at = datetime.now().isoformat()

                if title and link:
                    items.append({
                        'title': title,
                        'summary': description[:500] if description else title,
                        'content': description,
                        'url': link,
                        'source': source,
                        'published_at': published_at,
                        'fetched_at': datetime.now().isoformat(),
                        'sectors': [],
                    })
        except ET.ParseError as e:
            logger.error(f"RSS XML parse error: {e}")
        return items


# Global instance
_google_news_fetcher: Optional[GoogleNewsFetcher] = None


def get_google_news_fetcher() -> GoogleNewsFetcher:
    """Get or create global GoogleNewsFetcher instance."""
    global _google_news_fetcher
    if _google_news_fetcher is None:
        _google_news_fetcher = GoogleNewsFetcher()
    return _google_news_fetcher
