"""
MarketMind AI - App Controller
Main controller for managing data and business logic.
Integrates Kite Connect for live data, NSE/Screener.in for prices/fundamentals,
Google News RSS for sector news. All data cached in MongoDB with 10-min TTL.
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

from .core.database import Database
from .core.google_news_fetcher import GoogleNewsFetcher
from .core.sentiment_analyzer import SentimentAnalyzer
from .core.sector_classifier import SectorClassifier
from .core.price_fetcher import PriceFetcher
from .core.kite_client import KiteConfig, KiteClient, AutoTrader
from .core.claude_news_fetcher import run_claude_news_pipeline, get_recent_news_from_mongo
from .ml.rl_agent import RLDecisionEngine, DQNTradingAgent
from .ml.trading_env import TradingEnvironment
from .analysis.portfolio_simulator import PortfolioSimulator
from .analysis.tax_rebalancer import (
    CurrentHolding,
    recommend_tax_optimal_rebalance as _recommend_tax_optimal_rebalance,
)
from .analysis.tax_lots import TaxLot
from .compliance.audit_log import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    SOURCE_ORDER_ATTEMPT,
    AuditLogEntry,
    AuditLogStore,
)
from .compliance.insider_window import compute_insider_window
from .compliance.pretrade_check import PretradeChecker
from dataclasses import asdict
from datetime import date, datetime, timezone

try:
    import pymongo
    _mongo_available = True
except ImportError:
    _mongo_available = False

logger = logging.getLogger(__name__)


class AppController:
    """
    Main application controller.
    Manages data flow between UI and backend services.
    Uses Kite Connect as primary data source (when authenticated),
    falls back to NSE/Screener.in for price data.
    """

    # MongoDB TTL for cached data (seconds)
    _MONGO_TTL = 600  # 10 minutes

    def __init__(self):
        # Core data services
        self.db = Database()
        self.news_fetcher = GoogleNewsFetcher()   # Google News RSS
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sector_classifier = SectorClassifier()
        self.price_fetcher = PriceFetcher()       # NSE + Screener.in

        # MongoDB connection (for persistent cache with TTL)
        self._mongo_client = None
        self._mongo_db = None
        self._init_mongo()

        # SEBI compliance layer (W5.3) — must be constructed AFTER _init_mongo
        # so the audit-log + designated-symbols collections are reachable.
        # Designated-symbols cache lives here; invalidated on
        # `compliance_set_designated_symbols`.
        self._compliance_designated_cache: Optional[set] = None
        self._compliance_audit_store = AuditLogStore(
            mongo_col=self._mongo_col("compliance_audit_log")
        )
        self.compliance = PretradeChecker(
            audit_log=self._compliance_audit_store,
            designated_symbols_provider=self._compliance_designated_symbols,
        )

        # Kite Connect
        self.kite_config = KiteConfig()
        self.kite = KiteClient(self.kite_config)
        self.auto_trader = AutoTrader(self.kite, self.kite_config)

        # RL & Simulation
        self.rl_engine = RLDecisionEngine()
        self.portfolio_simulator = PortfolioSimulator()

        # State
        self.is_initialized = False
        self.is_loading = False
        self.latest_news: List[Dict] = []
        self.latest_signals: List[Dict] = []
        self.sector_data: Dict = {}
        self.watchlist_data: Dict = {}
        self.live_ticks: Dict = {}       # token -> latest tick
        self._tick_lock = threading.Lock()

        # UI update callback (called from bg thread → must use Qt signal/QTimer)
        self._on_data_updated_callbacks: List = []

        # Background update thread
        self.update_thread: Optional[threading.Thread] = None
        self.update_running = False

        # Set up Kite WebSocket callbacks
        self._setup_kite_callbacks()

    # ----------------------------------------------------------
    # MONGODB CACHE LAYER
    # ----------------------------------------------------------

    def _init_mongo(self):
        """Connect to MongoDB and ensure TTL index exists."""
        if not _mongo_available:
            return
        try:
            self._mongo_client = pymongo.MongoClient(
                'mongodb://localhost:27017/', serverSelectionTimeoutMS=2000
            )
            self._mongo_client.server_info()   # quick connectivity check
            self._mongo_db = self._mongo_client['marketmind']
            # TTL index: documents expire automatically after _MONGO_TTL seconds
            col = self._mongo_db['fetch_cache']
            col.create_index('fetched_at', expireAfterSeconds=self._MONGO_TTL)
            logger.info("MongoDB cache initialized (TTL=10 min)")
        except Exception as e:
            logger.warning(f"MongoDB unavailable, using in-memory only: {e}")
            self._mongo_client = None
            self._mongo_db = None

    def _mongo_col(self, name: str):
        """Return a MongoDB collection or None if not connected."""
        if self._mongo_db is not None:
            return self._mongo_db[name]
        return None

    def _should_fetch(self, cache_key: str) -> bool:
        """Return True if data is stale (>10 min) or not yet in MongoDB."""
        col = self._mongo_col('fetch_cache')
        if col is None:
            return True   # no Mongo → always fetch
        try:
            doc = col.find_one({'_id': cache_key})
            if doc is None:
                return True
            age = (datetime.now(timezone.utc) - doc['fetched_at'].replace(tzinfo=timezone.utc)).total_seconds()
            return age > self._MONGO_TTL
        except Exception:
            return True

    def _mark_fetched(self, cache_key: str):
        """Record a successful fetch timestamp in MongoDB."""
        col = self._mongo_col('fetch_cache')
        if col is None:
            return
        try:
            col.replace_one(
                {'_id': cache_key},
                {'_id': cache_key, 'fetched_at': datetime.utcnow()},
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"MongoDB mark_fetched error: {e}")

    def _write_news_to_mongo(self, news_items: List[Dict]):
        """Upsert news items into MongoDB (deduplicate by URL)."""
        col = self._mongo_col('news')
        if col is None or not news_items:
            return
        try:
            for item in news_items:
                url = item.get('url') or item.get('link')
                if not url:
                    continue
                col.replace_one({'url': url}, {**item, 'saved_at': datetime.utcnow()}, upsert=True)
            logger.debug(f"Wrote {len(news_items)} news items to MongoDB")
        except Exception as e:
            logger.warning(f"MongoDB news write error: {e}")

    def _write_prices_to_mongo(self, prices: Dict[str, Dict]):
        """Upsert watchlist price data into MongoDB."""
        col = self._mongo_col('prices')
        if col is None or not prices:
            return
        try:
            now = datetime.utcnow()
            for symbol, data in prices.items():
                col.replace_one(
                    {'symbol': symbol},
                    {**data, 'symbol': symbol, 'saved_at': now},
                    upsert=True,
                )
            logger.debug(f"Wrote {len(prices)} price records to MongoDB")
        except Exception as e:
            logger.warning(f"MongoDB price write error: {e}")

    def _write_rl_signals_to_mongo(self, signals: List[Dict]):
        """Write RL signals to MongoDB."""
        col = self._mongo_col('rl_signals')
        if col is None or not signals:
            return
        try:
            for sig in signals:
                col.replace_one(
                    {'symbol': sig.get('symbol'), 'timestamp': sig.get('timestamp')},
                    {**sig, 'saved_at': datetime.utcnow()},
                    upsert=True,
                )
        except Exception as e:
            logger.warning(f"MongoDB RL signal write error: {e}")

    # ----------------------------------------------------------
    # KITE SETUP
    # ----------------------------------------------------------

    def _setup_kite_callbacks(self):
        """Register Kite WebSocket callbacks"""
        self.kite.on_tick(self._on_kite_tick)
        self.kite.on_order_update(self._on_kite_order_update)
        self.kite.on_connect(self._on_kite_connect)
        self.kite.on_close(self._on_kite_close)
        self.kite.on_error(self._on_kite_error)

        # Auto-trader callbacks
        self.auto_trader.on_order_placed(self._on_auto_order_placed)
        self.auto_trader.on_order_executed(self._on_auto_order_executed)

    def _on_kite_tick(self, ticks: list):
        """Handle live ticks from KiteTicker"""
        with self._tick_lock:
            for tick in ticks:
                token = tick.get('instrument_token')
                if token:
                    self.live_ticks[token] = tick

    def _on_kite_order_update(self, data: dict):
        logger.info(f"Order update: {data.get('order_id')} → {data.get('status')}")

    def _on_kite_connect(self):
        logger.info("Kite WebSocket connected")

    def _on_kite_close(self):
        logger.info("Kite WebSocket closed")

    def _on_kite_error(self, code, reason):
        logger.error(f"Kite WebSocket error: {code} - {reason}")

    def _on_auto_order_placed(self, order_id: str, signal: dict):
        logger.info(f"Auto-buy order placed: {order_id} for {signal.get('symbol')}")

    def _on_auto_order_executed(self, order_id: str, order: dict, status: str):
        logger.info(f"Auto-buy order {order_id} finished with status: {status}")

    # ----------------------------------------------------------
    # KITE AUTHENTICATION API
    # ----------------------------------------------------------

    def get_kite_login_url(self) -> str:
        return self.kite.get_login_url()

    def kite_generate_session(self, request_token: str) -> bool:
        ok = self.kite.generate_session(request_token)
        if ok:
            self._setup_kite_callbacks()
            self.kite.start_ticker()
        return ok

    def kite_invalidate_session(self):
        self.kite.invalidate_session()

    @property
    def kite_is_authenticated(self) -> bool:
        return self.kite.is_connected

    @property
    def kite_is_configured(self) -> bool:
        return self.kite_config.is_configured

    def reload_kite_config(self):
        """Reload Kite config from local.json and re-init if needed"""
        self.kite_config.reload()
        if self.kite_config.is_authenticated:
            self.kite._initialize_kite()

    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------

    def initialize(self):
        """Initialize application — fetch data and train models"""
        if self.is_initialized:
            return
        self.is_loading = True
        try:
            # Start Kite ticker if authenticated
            if self.kite_config.is_authenticated:
                self.kite.start_ticker()
                logger.info("Kite ticker started")

            self.fetch_news()
            self.fetch_prices()
            self.initialize_rl_model()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
        finally:
            self.is_loading = False

    def start_background_updates(self):
        """Start background data updates"""
        self.update_running = True
        self.update_thread = threading.Thread(
            target=self._background_update_loop,
            name="MarketMindUpdater",
            daemon=True
        )
        self.update_thread.start()

    def stop_background_updates(self):
        """Stop background updates"""
        self.update_running = False
        self.kite.stop_ticker()
        self.auto_trader.stop_buyer_loop()
        if self.update_thread:
            self.update_thread.join(timeout=2)

    def _background_update_loop(self):
        """Periodic background data refresh (every 5 minutes)"""
        while self.update_running:
            try:
                self.fetch_news()
                self.fetch_prices()
                self.update_rl_signals()
            except Exception as e:
                logger.error(f"Background update error: {e}")
            time.sleep(300)

    # ----------------------------------------------------------
    # DATA FETCHING
    # ----------------------------------------------------------

    def fetch_all_data(self):
        """Fetch all market data (called by UI refresh button)"""
        self.fetch_news()
        self.fetch_prices()
        self.update_rl_signals()

    def fetch_news(self):
        """Fetch market news from Google News RSS for all sectors.
        Uses MongoDB as 10-min TTL cache: skip if data is fresh."""
        if not self._should_fetch('news'):
            logger.info("News cache is fresh (<10 min) — skipping fetch")
            return self.latest_news

        try:
            # Primary: Google News RSS (all sectors)
            news_items = self.news_fetcher.fetch_all_market_news(max_per_sector=6)

            # Fallback: Claude web-search pipeline
            if not news_items:
                api_key = getattr(self.kite_config, 'anthropic_api_key', None)
                news_items = run_claude_news_pipeline(api_key) or []

            # Annotate with sentiment + sectors and persist
            for item in news_items:
                text = (item.get('title', '') + ' '
                        + item.get('summary', '') + ' '
                        + item.get('content', ''))
                sentiment = self.sentiment_analyzer.analyze(text)
                item.setdefault('sentiment_score', sentiment['score'])
                item.setdefault('sentiment_label', sentiment['label'])
                if not item.get('sectors'):
                    classified = self.sector_classifier.classify_news(text)
                    item['sectors'] = [s[0] for s in classified[:2]]
                try:
                    self.db.store_news(item)
                except Exception:
                    pass

            # Persist to MongoDB and mark TTL
            self._write_news_to_mongo(news_items)
            self._mark_fetched('news')

            self.latest_news = news_items
            logger.info(f"Fetched {len(news_items)} news items from Google News")
            return news_items

        except Exception as e:
            logger.error(f"News fetch error: {e}")
            # Last resort: return recent items from MongoDB
            cached = get_recent_news_from_mongo(30)
            if cached:
                self.latest_news = cached
                return cached
            return self.latest_news

    def fetch_prices(self):
        """Fetch prices — Kite quotes if authenticated, else NSE/Screener.in.
        Writes results to MongoDB; skips if cache is fresh (<10 min)."""
        if not self._should_fetch('prices'):
            logger.info("Price cache is fresh (<10 min) — skipping fetch")
            return self.watchlist_data

        watchlist = self.kite_config.watchlist or [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'SBIN', 'TATAMOTORS', 'HINDUNILVR', 'BHARTIARTL'
        ]
        if self.kite.is_connected:
            self._fetch_prices_kite(watchlist)
        else:
            self._fetch_prices_nse(watchlist)

        self._write_prices_to_mongo(self.watchlist_data)
        self._mark_fetched('prices')
        return self.watchlist_data

    def _fetch_prices_kite(self, symbols: List[str]):
        """Fetch prices via Kite API"""
        try:
            kite_symbols = [f"NSE:{s}" for s in symbols]
            quotes = self.kite.get_quote(kite_symbols)
            for symbol in symbols:
                key = f"NSE:{symbol}"
                if key in quotes:
                    q = quotes[key]
                    ohlc = q.get('ohlc', {})
                    self.watchlist_data[symbol] = {
                        'symbol': symbol,
                        'name': q.get('instrument_token', symbol),
                        'current_price': q.get('last_price', 0),
                        'previous_close': ohlc.get('close', 0),
                        'open': ohlc.get('open', 0),
                        'day_high': ohlc.get('high', 0),
                        'day_low': ohlc.get('low', 0),
                        'volume': q.get('volume', 0),
                        'change': q.get('net_change', 0),
                        'change_pct': q.get('net_change', 0) / ohlc.get('close', 1) * 100
                            if ohlc.get('close') else 0,
                        'buy_qty': q.get('buy_quantity', 0),
                        'sell_qty': q.get('sell_quantity', 0),
                        'source': 'kite',
                    }
        except Exception as e:
            logger.error(f"Kite price fetch error: {e}")
            self._fetch_prices_nse(symbols)

    def _fetch_prices_nse(self, symbols: List[str]):
        """Fetch prices via NSE India API + Screener.in fundamentals."""
        for symbol in symbols:
            try:
                price_data = self.price_fetcher.get_stock_price(symbol)
                if price_data:
                    self.watchlist_data[symbol] = price_data
            except Exception as e:
                logger.error(f"NSE price fetch error for {symbol}: {e}")

    # ----------------------------------------------------------
    # MARKET STATUS
    # ----------------------------------------------------------

    def get_market_status(self) -> Dict:
        """Get current market status"""
        return self.price_fetcher.get_market_status()

    def get_index_data(self, index_name: str) -> Optional[Dict]:
        """Get index data — Kite first, NSE API fallback"""
        if self.kite.is_connected:
            try:
                symbol_map = {
                    'NIFTY500': 'NSE:NIFTY 500',
                    'NIFTY50': 'NSE:NIFTY 50',
                    'SENSEX': 'BSE:SENSEX',
                    'NIFTYBANK': 'NSE:NIFTY BANK',
                    'INDIA VIX': 'NSE:INDIA VIX',
                }
                kite_sym = symbol_map.get(index_name)
                if kite_sym:
                    q = self.kite.get_quote([kite_sym])
                    if kite_sym in q:
                        data = q[kite_sym]
                        ohlc = data.get('ohlc', {})
                        price = data.get('last_price', 0)
                        prev = ohlc.get('close', price)
                        change_pct = ((price - prev) / prev * 100) if prev else 0
                        return {
                            'price': price,
                            'change_pct': change_pct,
                            'change': price - prev,
                            'source': 'kite',
                        }
            except Exception as e:
                logger.error(f"Kite index data error: {e}")

        # Fallback to NSE API
        data = self.price_fetcher.get_stock_price(index_name)
        if data:
            data['source'] = 'nse'
        return data

    # ----------------------------------------------------------
    # STOCK DATA
    # ----------------------------------------------------------

    def get_latest_news(self) -> List[Dict]:
        return self.latest_news

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get full stock data with indicators and RL signal"""
        try:
            # Price
            if self.kite.is_connected:
                q = self.kite.get_quote([f"NSE:{symbol}"])
                kite_key = f"NSE:{symbol}"
                if kite_key in q:
                    qd = q[kite_key]
                    ohlc = qd.get('ohlc', {})
                    price = qd.get('last_price', 0)
                    prev = ohlc.get('close', price)
                    price_data = {
                        'symbol': symbol,
                        'name': symbol,
                        'current_price': price,
                        'change': price - prev,
                        'change_pct': ((price - prev) / prev) if prev else 0,
                        'day_high': ohlc.get('high', 0),
                        'day_low': ohlc.get('low', 0),
                        'volume': qd.get('volume', 0),
                        'source': 'kite',
                    }
                else:
                    price_data = self.price_fetcher.get_stock_price(symbol)
            else:
                price_data = self.price_fetcher.get_stock_price(symbol)

            if not price_data:
                return None

            # Fundamentals from Screener.in (only fields not already provided by Kite)
            try:
                fund = self.price_fetcher._get_screener_fundamentals(symbol) or {}
                for k in ('pe_ratio', 'pb_ratio', 'roe', 'roce', 'market_cap',
                          'book_value', 'dividend_yield', 'debt_equity', 'eps',
                          'revenue_growth', 'profit_growth'):
                    if price_data.get(k) is None and fund.get(k) is not None:
                        price_data[k] = fund[k]
            except Exception as e:
                logger.debug(f"Fundamentals merge failed for {symbol}: {e}")

            # Historical for chart
            if self.kite.is_connected:
                candles = self.kite.get_historical_data_by_symbol(symbol, 'NSE', 'day', 90)
                price_history = [c['close'] for c in candles] if candles else []
            else:
                hist_data = self.price_fetcher.get_historical_data(symbol, days=90)
                price_history = hist_data['close'].tolist() if not hist_data.empty else []

            # Technical indicators from NSE historical data
            indicators = self.price_fetcher.calculate_technical_indicators(symbol)

            # RL signal (None-safe)
            rl_signal = self.get_rl_signal_for_stock(symbol) or {}

            # Stock-specific news from Google News RSS
            related_news = self.news_fetcher.fetch_stock_news(symbol, max_items=5)

            return {
                'symbol': symbol,
                'name': price_data.get('name', symbol),
                'current_price': price_data.get('current_price', 0),
                'change': price_data.get('change', 0),
                'change_pct': price_data.get('change_pct', 0),
                'pe_ratio': price_data.get('pe_ratio'),
                'pb_ratio': price_data.get('pb_ratio'),
                'roe': price_data.get('roe'),
                'roce': price_data.get('roce'),
                'market_cap': price_data.get('market_cap'),
                'volume': price_data.get('volume'),
                '52_week_high': price_data.get('52_week_high'),
                '52_week_low': price_data.get('52_week_low'),
                'day_high': price_data.get('day_high'),
                'day_low': price_data.get('day_low'),
                'price_history': price_history,
                'technical_indicators': indicators,
                'rl_signal': rl_signal,
                'related_news': related_news,
                'source': price_data.get('source', 'nse'),
            }
        except Exception as e:
            logger.error(f"Stock data error for {symbol}: {e}")
            return None

    # ----------------------------------------------------------
    # SECTOR DATA
    # ----------------------------------------------------------

    def get_sector_data(self) -> Dict[str, Dict]:
        """Get data for all sectors"""
        sectors = ['IT', 'Banking', 'Auto', 'Pharma', 'FMCG', 'Metal', 'Energy', 'Realty', 'Finance']
        for sector in sectors:
            sector_news = self.sector_classifier.get_sector_news(self.latest_news, sector)
            sentiment = self.sentiment_analyzer.aggregate_sentiment(sector_news)
            change_pct = sentiment['score'] * 2
            self.sector_data[sector] = {
                'name': self.sector_classifier.get_sector_name(sector),
                'sentiment': sentiment['score'],
                'sentiment_label': sentiment['label'],
                'change_pct': change_pct,
                'news_count': sentiment['count'],
            }
        return self.sector_data

    def get_sector_details(self, sector: str) -> Dict:
        """Get detailed data for a sector"""
        stocks = self.sector_classifier.get_sector_stocks(sector)
        stock_prices = []
        for symbol in stocks[:5]:
            price = self.price_fetcher.get_stock_price(symbol)
            if price:
                stock_prices.append(price.get('current_price', 0))
        sector_news = self.sector_classifier.get_sector_news(self.latest_news, sector)
        sentiment = self.sentiment_analyzer.aggregate_sentiment(sector_news)
        return {
            'name': self.sector_classifier.get_sector_name(sector),
            'stocks': stocks,
            'change_pct': sentiment['score'] * 2,
            'sentiment': sentiment['score'],
            'news': sector_news[:5],
            'avg_price': sum(stock_prices) / len(stock_prices) if stock_prices else 0,
            'signal': self._get_signal_from_sentiment(sentiment['score']),
        }

    def get_all_correlations(self) -> Dict:
        """Get sector correlations"""
        sector_prices = {}
        for sector in self.sector_classifier.sectors:
            stocks = self.sector_classifier.get_sector_stocks(sector)
            if stocks:
                hist = self.price_fetcher.get_historical_data(stocks[0], days=90)
                if not hist.empty:
                    sector_prices[sector] = hist
        return self.sector_classifier.calculate_correlation_matrix(sector_prices)

    # ----------------------------------------------------------
    # PORTFOLIO (Kite + simulation fallback)
    # ----------------------------------------------------------

    def get_holdings(self) -> List[Dict]:
        """Get holdings from Kite or return empty list"""
        if self.kite.is_connected:
            return self.kite.get_holdings()
        return []

    def get_positions(self) -> Dict:
        """Get positions from Kite"""
        if self.kite.is_connected:
            return self.kite.get_positions()
        return {'day': [], 'net': []}

    def get_margins(self) -> Dict:
        """Get account margins from Kite"""
        if self.kite.is_connected:
            return self.kite.get_margins()
        return {}

    def get_available_cash(self) -> float:
        return self.kite.get_available_cash()

    def get_portfolio_summary(self) -> Dict:
        """Compute portfolio summary from holdings.

        Always returns ``authenticated`` (kite connection state) and ``error``
        (None on success, user-facing string on failure) so the frontend can
        distinguish "no holdings" from "fetch failed".
        """
        empty = {
            'total_value': 0,
            'invested': 0,
            'pnl': 0,
            'pnl_pct': 0,
            'day_change': 0,
            'holdings': [],
        }
        if not self.kite.is_connected:
            return {
                **empty,
                'authenticated': False,
                'error': 'Kite not connected. Click Login to authenticate.',
            }

        holdings, err = self.kite.get_holdings_with_diagnostic()
        if err is not None:
            return {**empty, 'authenticated': True, 'error': err}
        if not holdings:
            return {**empty, 'authenticated': True, 'error': None}

        total_value = sum(h.get('last_price', 0) * h.get('quantity', 0) for h in holdings)
        invested = sum(h.get('average_price', 0) * h.get('quantity', 0) for h in holdings)
        pnl = total_value - invested
        pnl_pct = (pnl / invested * 100) if invested else 0
        day_change = sum(
            h.get('day_change', 0) * h.get('quantity', 0)
            for h in holdings
        )

        return {
            'total_value': total_value,
            'invested': invested,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'day_change': day_change,
            'holdings': holdings,
            'authenticated': True,
            'error': None,
        }

    def recommend_tax_optimal_rebalance(
        self,
        target_weights: Dict[str, float],
        ltcg_used_inr: float = 0.0,
        new_symbol_prices: Optional[Dict[str, float]] = None,
        harvest_losses: bool = False,
        harvest_min_loss_inr: float = 1_000.0,
        harvest_min_loss_pct: float = 5.0,
        as_of: Optional[date] = None,
        lots_override: Optional[Dict[str, List[Dict]]] = None,
    ) -> Dict:
        """Build a tax-aware rebalance recommendation (W4.1).

        Wraps :func:`marketmind.analysis.tax_rebalancer.recommend_tax_optimal_rebalance`
        with Kite holdings fetch and the same ``authenticated`` / ``error``
        envelope used by :meth:`get_portfolio_summary`. Returns a flat dict
        ready for JSON serialisation.

        Lot-level acquisition dates are NOT available from Kite's ``holdings()``
        endpoint — every position is materialised as a single UNKNOWN-date lot,
        which the tax engine buckets as STCG worst-case (15%). The rebalancer
        surfaces this in the response ``warnings`` list.

        ``lots_override`` (optional, keyed by symbol) lets callers with their
        own lot ledger (broker-statement CSV, etc.) bypass the UNKNOWN
        fallback. Each lot dict needs ``quantity`` (float), ``cost_basis``
        (float), and optional ``acquisition_date`` (ISO YYYY-MM-DD or None).
        Per-symbol lot quantities must reconcile with the live Kite holding
        quantity within ±0.0001 — the rebalancer falls back to UNKNOWN
        single-lot with a warning otherwise.
        """
        empty: Dict = {
            'trades': [],
            'realized_gains': [],
            'tax_summary': {},
            'naive_tax_summary': {},
            'savings_inr': 0.0,
            'savings_pct': 0.0,
            'tracking_error_pct': 0.0,
            'harvest_candidates': [],
            'warnings': [],
        }

        if not self.kite.is_connected:
            return {
                **empty,
                'authenticated': False,
                'error': 'Kite not connected. Click Login to authenticate.',
            }

        holdings_raw, err = self.kite.get_holdings_with_diagnostic()
        if err is not None:
            return {**empty, 'authenticated': True, 'error': err}

        # Surface malformed Kite rows (missing tradingsymbol) as warnings instead
        # of silently dropping them — caller needs to know Kite returned junk.
        # Capped at 5 individual lines + 1 summary so a degenerate Kite response
        # cannot blow up the warnings payload.
        prelim_warnings: List[str] = []
        malformed_count = 0
        for idx, h in enumerate(holdings_raw):
            if not h.get('tradingsymbol'):
                malformed_count += 1
                if malformed_count <= 5:
                    prelim_warnings.append(
                        f"skipped Kite holdings row {idx}: missing tradingsymbol"
                    )
        if malformed_count > 5:
            prelim_warnings.append(
                f"... and {malformed_count - 5} more malformed Kite rows skipped"
            )

        # Materialise caller-supplied lot ledger (if any) into TaxLot lists
        # keyed by symbol. Bad date strings surface as a warning + fallback
        # to UNKNOWN for that symbol — never as a 5xx.
        lots_by_symbol: Dict[str, Optional[List[TaxLot]]] = {}
        if lots_override:
            for sym, lot_dicts in lots_override.items():
                try:
                    lots_by_symbol[sym] = [
                        TaxLot(
                            symbol=sym,
                            quantity=float(d.get('quantity', 0) or 0),
                            cost_basis=float(d.get('cost_basis', 0) or 0),
                            acquisition_date=(
                                datetime.strptime(d['acquisition_date'], '%Y-%m-%d').date()
                                if d.get('acquisition_date') else None
                            ),
                        )
                        for d in lot_dicts
                    ]
                except (ValueError, TypeError, KeyError) as e:
                    prelim_warnings.append(
                        f"{sym}: lots_override parse error "
                        f"({type(e).__name__}: {e}); falling back to UNKNOWN"
                    )
                    lots_by_symbol[sym] = None

        try:
            current = [
                CurrentHolding(
                    symbol=h.get('tradingsymbol', ''),
                    quantity=float(h.get('quantity', 0) or 0),
                    last_price=float(h.get('last_price', 0) or 0),
                    avg_cost=float(h.get('average_price', 0) or 0),
                    lots=lots_by_symbol.get(h.get('tradingsymbol', '')),
                )
                for h in holdings_raw
                if h.get('tradingsymbol')
            ]
        except (ValueError, TypeError) as e:
            return {**empty, 'authenticated': True,
                    'error': f"holdings parse error: {type(e).__name__}: {e}"}

        try:
            rec = _recommend_tax_optimal_rebalance(
                holdings=current,
                target_weights=target_weights,
                as_of=as_of or date.today(),
                ltcg_used_inr=ltcg_used_inr,
                new_symbol_prices=new_symbol_prices,
                harvest_losses=harvest_losses,
                harvest_min_loss_inr=harvest_min_loss_inr,
                harvest_min_loss_pct=harvest_min_loss_pct,
            )
        except ValueError as e:
            return {**empty, 'authenticated': True, 'error': str(e)}

        return {
            'trades': [asdict(t) for t in rec.trades],
            'realized_gains': [asdict(g) for g in rec.realized_gains],
            'tax_summary': dict(rec.tax_summary),
            'naive_tax_summary': dict(rec.naive_tax_summary),
            'savings_inr': rec.savings_inr,
            'savings_pct': rec.savings_pct,
            'tracking_error_pct': rec.tracking_error_pct,
            'harvest_candidates': [asdict(c) for c in rec.harvest_candidates],
            'warnings': prelim_warnings + list(rec.warnings),
            'authenticated': True,
            'error': None,
        }

    # ----------------------------------------------------------
    # SEBI COMPLIANCE (W5.3)
    # ----------------------------------------------------------

    def _compliance_designated_symbols(self) -> set:
        """Return the user-maintained designated-persons symbol set.

        Cached in-memory; invalidated by ``compliance_set_designated_symbols``.
        Returns an empty set when Mongo is unavailable — pretrade_check then
        skips the insider-window check for every symbol (safe-by-default).

        Returns a fresh copy each call so callers cannot mutate the cache
        through the returned reference.
        """
        if self._compliance_designated_cache is not None:
            return set(self._compliance_designated_cache)
        col = self._mongo_col("compliance_designated")
        if col is None:
            self._compliance_designated_cache = set()
            return set()
        try:
            doc = col.find_one({"_id": "symbols"})
        except Exception as e:
            logger.warning(f"compliance_designated load failed: {e}")
            self._compliance_designated_cache = set()
            return set()
        if not doc:
            self._compliance_designated_cache = set()
        else:
            self._compliance_designated_cache = {
                s.upper().strip() for s in (doc.get("symbols") or [])
                if isinstance(s, str) and s.strip()
            }
        return set(self._compliance_designated_cache)

    def _compliance_record_order_attempt(
        self, *, symbol, transaction_type, quantity, price, decision, reasons,
    ) -> None:
        """Best-effort audit append from place_order. Never raises."""
        try:
            entry = AuditLogEntry(
                ts=datetime.now(timezone.utc),
                symbol=str(symbol or "?").upper(),
                transaction_type=str(transaction_type or "?").upper(),
                quantity=float(quantity or 0),
                price=float(price or 0),
                decision=decision,
                reasons=list(reasons),
                source=SOURCE_ORDER_ATTEMPT,
            )
            self._compliance_audit_store.append(entry)
        except Exception as e:
            logger.warning(f"order_attempt audit failed: {e}")

    def compliance_pretrade_check(
        self,
        symbol: str,
        transaction_type: str,
        quantity: float,
        price: float,
    ) -> Dict:
        """Run the SEBI pre-trade gate. Always returns the standard envelope."""
        sym_norm = (symbol or "").upper().strip()
        if not sym_norm:
            return {
                'authenticated': self.kite_is_authenticated,
                'error': "symbol must be non-empty",
                'decision': 'BLOCK',
                'reasons': ["symbol must be non-empty"],
                'audit_id': None,
                'insider_window_open': None,
                'insider_window_reason': None,
                'ts': None,
            }
        # Only fetch announcements for designated symbols — saves the NSE round-trip.
        designated = self._compliance_designated_symbols()
        announcements: List[Dict] = []
        if sym_norm in designated:
            try:
                from marketmind.core.filings_ingest import get_filings_ingester
                announcements = get_filings_ingester().fetch_announcements(sym_norm)
            except Exception as e:
                logger.warning(f"announcement fetch failed for {sym_norm}: {e}")
        holdings: List[Dict] = []
        if self.kite_is_authenticated:
            try:
                holdings = self.kite.get_holdings() or []
            except Exception as e:
                logger.warning(f"holdings fetch failed: {e}")
        decision = self.compliance.check(
            symbol=sym_norm,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            announcements=announcements,
            holdings=holdings,
        )
        return {
            'authenticated': self.kite_is_authenticated,
            'error': None,
            **decision.to_dict(),
        }

    def compliance_get_audit_log(
        self,
        symbol: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> Dict:
        """Return audit-log entries newest-first with envelope."""
        parsed_since: Optional[datetime] = None
        if since:
            try:
                parsed_since = datetime.fromisoformat(since)
                if parsed_since.tzinfo is None:
                    parsed_since = parsed_since.replace(tzinfo=timezone.utc)
            except ValueError:
                return {
                    'authenticated': self.kite_is_authenticated,
                    'error': f"since must be ISO datetime, got {since!r}",
                    'entries': [],
                }
        try:
            rows = self._compliance_audit_store.query(
                symbol=symbol, since=parsed_since, limit=limit
            )
        except Exception as e:
            return {
                'authenticated': self.kite_is_authenticated,
                'error': f"audit log query failed: {type(e).__name__}: {e}",
                'entries': [],
            }
        entries = []
        for r in rows:
            e = dict(r)
            # Strip Mongo-internal `_id` — it is a write-collision-safe key
            # (symbol:ts:rand6) and not part of the regulatory audit contract.
            # Callers reference rows by `audit_id` from the pretrade response.
            e.pop('_id', None)
            ts = e.get('ts')
            if isinstance(ts, datetime):
                e['ts'] = ts.isoformat()
            entries.append(e)
        return {
            'authenticated': self.kite_is_authenticated,
            'error': None,
            'entries': entries,
        }

    def compliance_get_insider_window(self, symbol: str) -> Dict:
        """Compute insider-window status for a symbol (regardless of designation)."""
        sym_norm = (symbol or "").upper().strip()
        if not sym_norm:
            return {
                'authenticated': self.kite_is_authenticated,
                'error': "symbol must be non-empty",
                'symbol': '',
                'is_open': None,
                'closed_until': None,
                'last_results_date': None,
                'reason': '',
            }
        announcements: List[Dict] = []
        fetch_err: Optional[str] = None
        try:
            from marketmind.core.filings_ingest import get_filings_ingester
            announcements = get_filings_ingester().fetch_announcements(sym_norm)
        except Exception as e:
            fetch_err = f"announcement fetch failed: {type(e).__name__}: {e}"
        today = datetime.now(timezone.utc).date()
        status = compute_insider_window(sym_norm, announcements, today)
        return {
            'authenticated': self.kite_is_authenticated,
            'error': fetch_err,
            'symbol': status.symbol,
            'is_open': status.is_open,
            'closed_until': status.closed_until.isoformat() if status.closed_until else None,
            'last_results_date': (
                status.last_results_date.isoformat() if status.last_results_date else None
            ),
            'reason': status.reason,
        }

    def compliance_set_designated_symbols(self, symbols: List[str]) -> Dict:
        """Replace the designated-symbols list. Persists to Mongo + invalidates cache."""
        cleaned = [
            s.upper().strip() for s in (symbols or [])
            if isinstance(s, str) and s.strip()
        ]
        # Dedupe preserving order.
        seen: set = set()
        deduped: List[str] = []
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        col = self._mongo_col("compliance_designated")
        if col is None:
            self._compliance_designated_cache = set(deduped)
            return {
                'authenticated': self.kite_is_authenticated,
                'error': "mongo not available; designated list not persisted",
                'symbols': deduped,
            }
        try:
            col.replace_one(
                {"_id": "symbols"},
                {"_id": "symbols", "symbols": deduped},
                upsert=True,
            )
        except Exception as e:
            return {
                'authenticated': self.kite_is_authenticated,
                'error': f"failed to persist: {type(e).__name__}: {e}",
                'symbols': deduped,
            }
        self._compliance_designated_cache = set(deduped)
        return {
            'authenticated': self.kite_is_authenticated,
            'error': None,
            'symbols': deduped,
        }

    # ----------------------------------------------------------
    # ORDERS
    # ----------------------------------------------------------

    def get_orders(self) -> List[Dict]:
        """Get all orders for the day"""
        if self.kite.is_connected:
            return self.kite.get_orders()
        return []

    def get_trades(self) -> List[Dict]:
        """Get all executed trades"""
        if self.kite.is_connected:
            return self.kite.get_trades()
        return []

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        product: str = 'CNC',
        price: float = 0,
        trigger_price: float = 0,
        stoploss: float = 0,
        squareoff: float = 0,
        trailing_stoploss: float = 0,
        variety: str = 'regular',
        tag: str = 'marketmind',
    ) -> Optional[str]:
        """Place an order via Kite. Writes one compliance-audit row per call
        (source=order_attempt) — covers SEBI Algo PDA post-trade audit."""
        audit_kwargs = dict(
            symbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
        )
        if not self.kite.is_connected:
            logger.warning("Kite not authenticated — cannot place order")
            self._compliance_record_order_attempt(
                decision=DECISION_BLOCK,
                reasons=["kite not authenticated"],
                **audit_kwargs,
            )
            return None
        # Audit on EVERY attempt — including the rare case where KiteClient
        # leaks an exception out of its internal try/except. Catching here
        # (rather than relying on KiteClient's wrapper) keeps the
        # regulatory-audit guarantee intact under future KiteClient changes.
        try:
            order_id = self.kite.place_order(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                price=price,
                trigger_price=trigger_price,
                stoploss=stoploss,
                squareoff=squareoff,
                trailing_stoploss=trailing_stoploss,
                variety=variety,
                tag=tag,
            )
        except Exception as e:
            logger.error(f"Order placement raised: {e}")
            self._compliance_record_order_attempt(
                decision=DECISION_BLOCK,
                reasons=[f"kite raised: {type(e).__name__}: {e}"],
                **audit_kwargs,
            )
            return None
        self._compliance_record_order_attempt(
            decision=DECISION_ALLOW if order_id else DECISION_BLOCK,
            reasons=[f"order placed: {order_id}"] if order_id else ["kite rejected order"],
            **audit_kwargs,
        )
        return order_id

    def cancel_order(self, order_id: str, variety: str = 'regular') -> Optional[str]:
        return self.kite.cancel_order(order_id, variety) if self.kite.is_connected else None

    def modify_order(self, order_id: str, **kwargs) -> Optional[str]:
        return self.kite.modify_order(order_id, **kwargs) if self.kite.is_connected else None

    # ----------------------------------------------------------
    # AUTO TRADER
    # ----------------------------------------------------------

    def start_buyer_loop(self):
        """Start automated buyer loop"""
        self.auto_trader.start_buyer_loop()

    def stop_buyer_loop(self):
        """Stop automated buyer loop"""
        self.auto_trader.stop_buyer_loop()

    @property
    def buyer_loop_active(self) -> bool:
        return self.auto_trader.buyer_loop_active

    def get_auto_trader_orders(self) -> Dict:
        return self.auto_trader.active_orders

    # ----------------------------------------------------------
    # RL SIGNALS
    # ----------------------------------------------------------

    def initialize_rl_model(self):
        """Initialize RL model with historical data"""
        try:
            hist_data = self.price_fetcher.get_historical_data('NIFTY500', days=365)
            if not hist_data.empty:
                env = TradingEnvironment(hist_data)
                self.rl_engine.dqn_agent.train(env, episodes=20)
                self.rl_engine.initialized = True
                self.update_rl_signals()
        except Exception as e:
            logger.error(f"RL init error: {e}")

    def update_rl_signals(self):
        """Update RL trading signals and optionally feed buyer loop"""
        try:
            signals = []
            for symbol in ['NIFTY500', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK']:
                indicators = self.price_fetcher.calculate_technical_indicators(symbol)
                sentiment = self._get_sentiment_for_stock(symbol)
                state = self._build_state_vector(indicators, sentiment)
                decision = self.rl_engine.make_decision(state)

                if decision['confidence'] > 0.4:
                    signal = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'position_size': decision['position_size'],
                        'entry_price': indicators.get('current_price', 0),
                        'exit_price': indicators.get('current_price', 0) * 1.05,
                        'stop_loss': indicators.get('current_price', 0) * 0.97,
                        'rationale': '; '.join(decision['reasoning'][:3]),
                    }
                    signals.append(signal)
                    self.db.store_rl_signal(signal)

                    # Feed BUY signals to auto-trader buyer loop
                    if (decision['action'] == 'BUY'
                            and self.auto_trader.buyer_loop_active
                            and indicators.get('current_price', 0) > 0):
                        qty = max(1, int(
                            self.kite_config.max_order_value / indicators['current_price']
                        ))
                        self.auto_trader.add_buy_signal(
                            symbol=symbol,
                            exchange='NSE',
                            price=indicators['current_price'],
                            quantity=qty,
                            confidence=decision['confidence'],
                            reason=signal['rationale'],
                        )

            self.latest_signals = signals
            self._write_rl_signals_to_mongo(signals)
        except Exception as e:
            logger.error(f"RL signals update error: {e}")

    def get_rl_signals(self) -> List[Dict]:
        if not self.latest_signals:
            self.update_rl_signals()
        return self.latest_signals

    def get_rl_signal_for_stock(self, symbol: str) -> Optional[Dict]:
        for s in self.latest_signals:
            if s.get('symbol') == symbol:
                return s
        return None

    # ----------------------------------------------------------
    # SIMULATION
    # ----------------------------------------------------------

    def run_simulation(self) -> Dict:
        """Run Monte Carlo portfolio simulation"""
        try:
            results = self.portfolio_simulator.run_simulations(
                expected_return=0.12,
                volatility=0.18,
                num_simulations=1000
            )
            stats = self.portfolio_simulator.calculate_summary_stats(results)
            stats['returns'] = list(results.get('returns', []))
            return stats
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {}

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    def _build_state_vector(self, indicators: Dict, sentiment: float):
        import numpy as np
        state = np.zeros(50)
        state[0] = indicators.get('momentum_5', 0)
        state[1] = indicators.get('momentum_10', 0)
        state[2] = indicators.get('momentum_20', 0)
        state[10] = (indicators.get('rsi', 50) - 50) / 100
        state[11] = indicators.get('macd', 0) / 100
        state[12] = indicators.get('volume_ratio', 1) - 1
        state[20] = sentiment
        return state

    def _get_sentiment_for_stock(self, symbol: str) -> float:
        total, count = 0.0, 0
        for news in self.latest_news:
            text = news.get('title', '') + ' ' + news.get('content', '')
            if symbol.lower() in text.lower():
                total += news.get('sentiment_score', 0)
                count += 1
        return total / count if count > 0 else 0

    def _get_signal_from_sentiment(self, sentiment: float) -> str:
        if sentiment > 0.3:
            return "STRONG BUY"
        elif sentiment > 0.1:
            return "BUY"
        elif sentiment > -0.1:
            return "NEUTRAL"
        elif sentiment > -0.3:
            return "SELL"
        else:
            return "STRONG SELL"
