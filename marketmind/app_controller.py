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
        """Compute portfolio summary from holdings"""
        holdings = self.get_holdings()
        if not holdings:
            return {
                'total_value': 0,
                'invested': 0,
                'pnl': 0,
                'pnl_pct': 0,
                'day_change': 0,
                'holdings': [],
            }

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
        """Place an order via Kite"""
        if not self.kite.is_connected:
            logger.warning("Kite not authenticated — cannot place order")
            return None
        return self.kite.place_order(
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
