"""
MarketMind AI - News Fetcher Module
Fetches news from various sources using web search and scraping
"""

import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class NewsFetcher:
    """Fetches financial news from various sources"""

    # News sources
    SOURCES = [
        'Economic Times',
        'MoneyControl',
        'Business Standard',
        'Reuters India',
        'Bloomberg India',
        'CNBC-TV18',
        'Zee Business',
        'NDTV Profit'
    ]

    # Search queries for Indian markets
    SEARCH_QUERIES = [
        'NSE stock market India',
        'BSE Sensex today',
        'Indian stock recommendations',
        'RBI monetary policy India',
        'FII DII activity India',
        'Nifty 500 analysis',
        'sectoral news India',
        'global market news India impact',
        'geopolitical news market impact',
        'war news market India',
    ]

    # Sector keywords for classification
    SECTOR_KEYWORDS = {
        'IT': ['IT company', 'software', 'TCS', 'Infosys', 'Wipro', 'HCL Tech', 'Tech Mahindra', 'IT sector', 'digital', 'AI'],
        'Banking': ['bank', 'banking', 'HDFC', 'ICICI', 'SBI', 'Axis Bank', 'Kotak', 'NBFC', 'credit', 'loan', 'deposit'],
        'Auto': ['auto', 'car', 'SUV', 'electric vehicle', 'EV', 'Maruti', 'Tata Motors', 'Mahindra', 'Bajaj Auto', 'automobile'],
        'Pharma': ['pharma', 'pharmaceutical', 'sun pharma', 'Dr Reddy', 'Cipla', 'Lupin', 'drug', 'medicine', 'API'],
        'FMCG': ['FMCG', 'consumer goods', 'HUL', 'ITC', 'Nestle', 'Dabur', 'Britannia', 'Marico', 'personal care'],
        'Metal': ['metal', 'steel', 'Tata Steel', 'Hindalco', 'Vedanta', 'coal', 'mining', 'aluminum', 'copper'],
        'Energy': ['oil', 'gas', 'ONGC', 'Reliance', 'IOC', 'BPCL', 'petrol', 'diesel', 'energy', 'renewable', 'solar'],
        'Realty': ['real estate', 'property', 'DLF', 'Godrej', 'budget housing', 'REIT', 'commercial property'],
        'Finance': ['finance', 'insurance', 'Bajaj Finance', 'mutual fund', 'SIP', 'LIC', 'HDFC Life', 'stock market'],
        'Global': ['Fed rate', 'US markets', 'Wall Street', 'China economy', 'Europe markets', 'OPEC', 'crude oil prices', 'global'],
        'Geopolitical': ['war', 'Russia Ukraine', 'Middle East', 'US China', 'sanctions', 'geopolitical', 'conflict', 'nuclear'],
    }

    def __init__(self, db=None):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    def fetch_news_api(self, query: str = None, days: int = 7, max_results: int = 50) -> List[Dict]:
        """Fetch news via web scraping (NewsAPI not used)"""
        news_items = []

        if query is None:
            query = "Indian stock market news today"

        news_items = self._fetch_web_search(query, max_results)

        # Also fetch some curated sector-specific news
        try:
            sector_news = self._fetch_sector_news(max_results // 4)
            news_items.extend(sector_news)
        except Exception as e:
            print(f"Sector news fetch failed: {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_news = []
        for item in news_items:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_news.append(item)

        return unique_news[:max_results]

    def _fetch_newsapi(self, query: str, max_results: int) -> List[Dict]:
        """Fetch from NewsAPI"""
        # Note: In production, use an API key
        api_key = None  # Would be loaded from config
        if not api_key:
            raise Exception("No NewsAPI key")

        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': max_results
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        news_items = []
        for article in data.get('articles', []):
            news_items.append({
                'title': article.get('title', ''),
                'content': article.get('description', ''),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'fetched_at': datetime.now().isoformat(),
                'sectors': self._classify_sectors(article.get('title', '') + ' ' + article.get('description', ''))
            })

        return news_items

    def _fetch_web_search(self, query: str, max_results: int) -> List[Dict]:
        """Fetch news by web scraping search results"""
        news_items = []

        # Use DuckDuckGo news for scraping
        search_url = f"https://duckduckgo.com/?q={query}+India+stock+market&ia=news"

        try:
            response = self.session.get(search_url, timeout=8)
            soup = BeautifulSoup(response.text, 'lxml')

            # Parse news results
            articles = soup.find_all('div', class_='result')
            for article in articles[:max_results]:
                try:
                    title_elem = article.find('a', class_='result__a')
                    snippet_elem = article.find('a', class_='result__snippet')

                    if title_elem:
                        news_items.append({
                            'title': title_elem.get_text(strip=True),
                            'content': snippet_elem.get_text(strip=True) if snippet_elem else '',
                            'source': '',
                            'url': title_elem.get('href', ''),
                            'published_at': datetime.now().isoformat(),
                            'fetched_at': datetime.now().isoformat(),
                            'sectors': self._classify_sectors(title_elem.get_text(strip=True))
                        })
                except Exception:
                    continue

        except Exception as e:
            print(f"Web search failed: {e}")

        return news_items

    def _fetch_sector_news(self, max_per_sector: int = 5) -> List[Dict]:
        """Fetch news specific to each sector"""
        sector_queries = [
            ('IT', 'Indian IT sector stock news TCS Infosys'),
            ('Banking', 'Indian banking sector news HDFC ICICI SBI'),
            ('Auto', 'Indian automobile sector news Maruti Tata Motors'),
            ('Pharma', 'Indian pharma sector news Sun Pharma Dr Reddy'),
            ('Energy', 'Indian energy oil gas news Reliance ONGC'),
            ('Global', 'global markets news impact India'),
        ]

        all_news = []
        for sector, query in sector_queries:
            try:
                news = self._fetch_web_search(query, max_per_sector)
                for item in news:
                    item['sectors'] = [sector]
                all_news.extend(news)
                time.sleep(0.5)  # Rate limiting
            except Exception:
                continue

        return all_news

    def _classify_sectors(self, text: str) -> List[str]:
        """Classify which sectors a news item belongs to"""
        text_lower = text.lower()
        matched_sectors = []

        for sector, keywords in self.SECTOR_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if sector not in matched_sectors:
                        matched_sectors.append(sector)
                    break

        # Default to Global if no specific sector matched
        if not matched_sectors:
            matched_sectors.append('Global')

        return matched_sectors

    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')

            # Try common article containers
            for selector in ['article', '.article-content', '.story-body', '.post-content']:
                content = soup.select_one(selector)
                if content:
                    return content.get_text(strip=True)

            # Fallback to main content
            main = soup.find('main')
            if main:
                return main.get_text(strip=True)

            return soup.get_text(strip=True)[:2000]

        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def search_news_terminal(self, query: str, max_results: int = 20) -> List[Dict]:
        """Use SerpAPI for news search (simulated without API key)"""
        # This would use actual SerpAPI in production
        # For now, use web scraping approach

        search_url = f"https://serpapi.com/search.json"

        params = {
            'q': query,
            'tbm': 'nws',  # News tab
            'num': max_results
        }

        try:
            # Simulated - in production use actual API
            # response = requests.get(search_url, params=params, timeout=15)
            # data = response.json()

            # For now, fallback to web scraping
            return self._fetch_web_search(query, max_results)

        except Exception as e:
            print(f"SerpAPI search failed: {e}")
            return self._fetch_web_search(query, max_results)

    def get_market_news(self, days: int = 1) -> List[Dict]:
        """Get today's market news"""
        queries = [
            'NSE BSE stock market today India',
            'Nifty 500 share market news',
            'FII DII buying selling India',
            'RBI policy rate India today',
            'Indian stock market analysis today'
        ]

        all_news = []
        for query in queries:
            try:
                news = self.fetch_news_api(query, days=days, max_results=10)
                all_news.extend(news)
                time.sleep(0.3)
            except Exception:
                continue

        # Deduplicate
        seen_urls = set()
        unique_news = []
        for item in all_news:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_news.append(item)

        return unique_news

    def get_geopolitical_news(self, days: int = 7) -> List[Dict]:
        """Get geopolitical news that might affect markets"""
        queries = [
            'Russia Ukraine war impact markets',
            'US China trade news markets',
            'Middle East oil prices impact',
            'global recession fears markets',
            'Fed interest rate decision markets'
        ]

        all_news = []
        for query in queries:
            try:
                news = self.fetch_news_api(query, days=days, max_results=5)
                all_news.extend(news)
                time.sleep(0.3)
            except Exception:
                continue

        return all_news


# Global instance
_news_fetcher = None


def get_news_fetcher(db=None) -> NewsFetcher:
    """Get or create global news fetcher instance"""
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsFetcher(db)
    return _news_fetcher
