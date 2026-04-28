"""
MarketMind AI - Kite Connect Client
Full Kite Connect API integration for live price feeds, historical candles,
market quotes, instruments, portfolio, orders with trailing stop-loss,
WebSocket callbacks, and automated buyer-loop trading.
"""

import json
import os
import threading
import time
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# Path to local.json config (project root)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'local.json'
)


# ============================================================
# CONFIGURATION
# ============================================================

class KiteConfig:
    """Manages Kite Connect configuration from local.json"""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return {}

    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def reload(self):
        self.config = self._load_config()

    @property
    def api_key(self) -> str:
        return os.environ.get('KITE_API_KEY') or self.config.get('kite', {}).get('api_key', '')

    @property
    def api_secret(self) -> str:
        return os.environ.get('KITE_API_SECRET') or self.config.get('kite', {}).get('api_secret', '')

    @property
    def access_token(self) -> str:
        return os.environ.get('KITE_ACCESS_TOKEN') or self.config.get('kite', {}).get('access_token', '')

    @property
    def request_token(self) -> str:
        return os.environ.get('KITE_REQUEST_TOKEN') or self.config.get('kite', {}).get('request_token', '')

    def set_access_token(self, token: str):
        if 'kite' not in self.config:
            self.config['kite'] = {}
        self.config['kite']['access_token'] = token
        self.save_config()

    def set_request_token(self, token: str):
        if 'kite' not in self.config:
            self.config['kite'] = {}
        self.config['kite']['request_token'] = token
        self.save_config()

    def set_api_credentials(self, api_key: str, api_secret: str):
        if 'kite' not in self.config:
            self.config['kite'] = {}
        self.config['kite']['api_key'] = api_key
        self.config['kite']['api_secret'] = api_secret
        self.save_config()

    @property
    def anthropic_api_key(self) -> str:
        # Env var wins over file (12-factor)
        return (
            os.environ.get('ANTHROPIC_API_KEY', '')
            or self.config.get('anthropic', {}).get('api_key', '')
        )

    @property
    def is_configured(self) -> bool:
        key = self.api_key
        return bool(key and key != 'your_api_key_here')

    @property
    def is_authenticated(self) -> bool:
        return bool(self.is_configured and self.access_token)

    # App settings
    @property
    def auto_trading_enabled(self) -> bool:
        return self.config.get('app', {}).get('auto_trading_enabled', False)

    @property
    def max_order_value(self) -> float:
        return self.config.get('app', {}).get('max_order_value', 50000)

    @property
    def trailing_sl_percent(self) -> float:
        return self.config.get('app', {}).get('trailing_sl_percent', 1.5)

    @property
    def target_percent(self) -> float:
        return self.config.get('app', {}).get('target_percent', 3.0)

    @property
    def buyer_loop_interval(self) -> int:
        return self.config.get('app', {}).get('buyer_loop_interval_seconds', 60)

    @property
    def min_confidence(self) -> float:
        return self.config.get('app', {}).get('min_confidence_threshold', 0.6)

    @property
    def default_product(self) -> str:
        return self.config.get('app', {}).get('default_product', 'CNC')

    @property
    def default_exchange(self) -> str:
        return self.config.get('app', {}).get('default_exchange', 'NSE')

    @property
    def watchlist(self) -> List[str]:
        return self.config.get('watchlist', [])


# ============================================================
# KITE CLIENT
# ============================================================

class KiteClient:
    """
    Full Kite Connect API client wrapper.

    Handles:
    - Authentication & session management
    - WebSocket live price feeds (KiteTicker)
    - Market quotes, OHLC, LTP
    - Historical candle data (any interval)
    - Instruments list & search
    - Portfolio: holdings, positions, margins
    - Orders: place, modify, cancel (market, limit, SL, bracket + trailing SL)
    - Trades & order history
    - Order update callbacks via WebSocket
    """

    def __init__(self, config: KiteConfig):
        self.config = config
        self.kite = None
        self.ticker = None
        self.is_connected = False
        self.is_ticker_running = False

        # Callback registries
        self._tick_callbacks: List[Callable] = []
        self._order_update_callbacks: List[Callable] = []
        self._connect_callbacks: List[Callable] = []
        self._close_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []

        # Instrument cache
        self._instruments_cache: Dict[str, List] = {}

        # Subscribed tokens set
        self._subscribed_tokens: set = set()

        # Initialize if credentials available
        if self.config.is_authenticated:
            self._initialize_kite()

    # ----------------------------------------------------------
    # INITIALIZATION & AUTH
    # ----------------------------------------------------------

    def _initialize_kite(self):
        """Initialize KiteConnect REST client"""
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.config.api_key)
            self.kite.set_access_token(self.config.access_token)
            self.is_connected = True
            logger.info("KiteConnect initialized")
        except ImportError:
            logger.error("kiteconnect not installed. Run: pip install kiteconnect")
            self.is_connected = False
        except Exception as e:
            logger.error(f"KiteConnect init error: {e}")
            self.is_connected = False

    def get_login_url(self) -> str:
        """Return Kite login URL for browser-based authentication"""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.config.api_key)
            return kite.login_url()
        except Exception as e:
            logger.error(f"Login URL error: {e}")
            return ""

    def generate_session(self, request_token: str) -> bool:
        """Exchange request_token for access_token. Call after user logs in."""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.config.api_key)
            data = kite.generate_session(request_token, api_secret=self.config.api_secret)
            self.config.set_access_token(data["access_token"])
            self._initialize_kite()
            logger.info("Session generated successfully")
            return True
        except Exception as e:
            logger.error(f"Session generation error: {e}")
            return False

    def invalidate_session(self):
        """Log out and clear access token"""
        try:
            if self.kite:
                self.kite.invalidate_access_token()
        except Exception:
            pass
        self.config.set_access_token('')
        self.kite = None
        self.is_connected = False

    # ----------------------------------------------------------
    # CALLBACK REGISTRATION
    # ----------------------------------------------------------

    def on_tick(self, callback: Callable):
        """Register live tick callback: callback(ticks: list)"""
        self._tick_callbacks.append(callback)

    def on_order_update(self, callback: Callable):
        """Register order update callback: callback(data: dict)"""
        self._order_update_callbacks.append(callback)

    def on_connect(self, callback: Callable):
        """Register WebSocket connect callback: callback()"""
        self._connect_callbacks.append(callback)

    def on_close(self, callback: Callable):
        """Register WebSocket close callback: callback()"""
        self._close_callbacks.append(callback)

    def on_error(self, callback: Callable):
        """Register WebSocket error callback: callback(code, reason)"""
        self._error_callbacks.append(callback)

    def _fire(self, callbacks: list, *args):
        for cb in callbacks:
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # ----------------------------------------------------------
    # WEBSOCKET TICKER
    # ----------------------------------------------------------

    def start_ticker(self, tokens: List[int] = None):
        """
        Start KiteTicker WebSocket for live price feeds.
        tokens: list of instrument tokens to subscribe immediately.
        """
        if not self.config.is_authenticated:
            logger.warning("Not authenticated - cannot start ticker")
            return

        try:
            from kiteconnect import KiteTicker

            self.ticker = KiteTicker(self.config.api_key, self.config.access_token)

            def _on_ticks(ws, ticks):
                self._fire(self._tick_callbacks, ticks)

            def _on_connect(ws, response):
                self.is_ticker_running = True
                logger.info("KiteTicker connected")
                # Subscribe to tokens
                all_tokens = list(self._subscribed_tokens)
                if tokens:
                    all_tokens = list(set(all_tokens + tokens))
                if all_tokens:
                    ws.subscribe(all_tokens)
                    ws.set_mode(ws.MODE_FULL, all_tokens)
                self._fire(self._connect_callbacks)

            def _on_close(ws, code, reason):
                self.is_ticker_running = False
                logger.info(f"KiteTicker closed: {code} - {reason}")
                self._fire(self._close_callbacks)

            def _on_error(ws, code, reason):
                logger.error(f"KiteTicker error: {code} - {reason}")
                self._fire(self._error_callbacks, code, reason)

            def _on_order_update(ws, data):
                self._fire(self._order_update_callbacks, data)

            def _on_reconnect(ws, attempts_count):
                logger.info(f"KiteTicker reconnecting: attempt {attempts_count}")

            def _on_noreconnect(ws):
                logger.error("KiteTicker max reconnection attempts reached")

            self.ticker.on_ticks = _on_ticks
            self.ticker.on_connect = _on_connect
            self.ticker.on_close = _on_close
            self.ticker.on_error = _on_error
            self.ticker.on_order_update = _on_order_update
            self.ticker.on_reconnect = _on_reconnect
            self.ticker.on_noreconnect = _on_noreconnect

            t = threading.Thread(target=self.ticker.connect, kwargs={'threaded': True}, daemon=True)
            t.start()

        except ImportError:
            logger.error("kiteconnect not installed")
        except Exception as e:
            logger.error(f"Ticker start error: {e}")

    def stop_ticker(self):
        """Stop the WebSocket ticker"""
        if self.ticker:
            try:
                self.ticker.close()
            except Exception as e:
                logger.error(f"Ticker stop error: {e}")
            self.is_ticker_running = False

    def subscribe(self, tokens: List[int]):
        """Subscribe to instrument tokens for live data"""
        self._subscribed_tokens.update(tokens)
        if self.ticker and self.is_ticker_running:
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)

    def unsubscribe(self, tokens: List[int]):
        """Unsubscribe from tokens"""
        for t in tokens:
            self._subscribed_tokens.discard(t)
        if self.ticker and self.is_ticker_running:
            self.ticker.unsubscribe(tokens)

    # ----------------------------------------------------------
    # MARKET DATA - QUOTES
    # ----------------------------------------------------------

    def get_quote(self, symbols: List[str]) -> Dict:
        """
        Get full market quote for symbols.
        symbols format: ['NSE:RELIANCE', 'BSE:SENSEX', 'NSE:NIFTY 50']
        Returns dict keyed by symbol with full quote data.
        """
        if not self.kite:
            return {}
        try:
            return self.kite.quote(symbols)
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return {}

    def get_ltp(self, symbols: List[str]) -> Dict:
        """Get last traded price for symbols."""
        if not self.kite:
            return {}
        try:
            return self.kite.ltp(symbols)
        except Exception as e:
            logger.error(f"LTP error: {e}")
            return {}

    def get_ohlc(self, symbols: List[str]) -> Dict:
        """Get OHLC data for symbols."""
        if not self.kite:
            return {}
        try:
            return self.kite.ohlc(symbols)
        except Exception as e:
            logger.error(f"OHLC error: {e}")
            return {}

    # ----------------------------------------------------------
    # MARKET DATA - HISTORICAL CANDLES
    # ----------------------------------------------------------

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: date,
        to_date: date,
        interval: str = 'day',
        continuous: bool = False,
        oi: bool = False
    ) -> List[Dict]:
        """
        Get historical OHLCV candle data.

        interval options:
            minute, 3minute, 5minute, 10minute, 15minute, 30minute,
            60minute, day, week, month

        Returns list of dicts with keys: date, open, high, low, close, volume, oi
        """
        if not self.kite:
            return []
        try:
            return self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous,
                oi=oi
            )
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return []

    def get_historical_data_by_symbol(
        self,
        symbol: str,
        exchange: str = 'NSE',
        interval: str = 'day',
        days: int = 365
    ) -> List[Dict]:
        """
        Convenience: get historical data by trading symbol.
        Resolves symbol -> instrument_token automatically.
        """
        token = self.get_instrument_token(symbol, exchange)
        if not token:
            return []
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        return self.get_historical_data(token, from_date, to_date, interval)

    # ----------------------------------------------------------
    # INSTRUMENTS
    # ----------------------------------------------------------

    def get_instruments(self, exchange: str = None) -> List[Dict]:
        """
        Get full instruments list.
        exchange: 'NSE', 'BSE', 'NFO', 'MCX', etc. (None = all)
        Cached in memory for the session.
        """
        if not self.kite:
            return []
        cache_key = exchange or 'ALL'
        if cache_key in self._instruments_cache:
            return self._instruments_cache[cache_key]
        try:
            insts = self.kite.instruments(exchange) if exchange else self.kite.instruments()
            self._instruments_cache[cache_key] = insts
            return insts
        except Exception as e:
            logger.error(f"Instruments error: {e}")
            return []

    def get_instrument_token(self, symbol: str, exchange: str = 'NSE') -> Optional[int]:
        """Resolve trading symbol to instrument token."""
        for inst in self.get_instruments(exchange):
            if inst.get('tradingsymbol') == symbol and inst.get('exchange') == exchange:
                return inst.get('instrument_token')
        return None

    def search_instruments(self, query: str, exchange: str = 'NSE') -> List[Dict]:
        """Search instruments by symbol or name."""
        query = query.upper()
        return [
            i for i in self.get_instruments(exchange)
            if query in i.get('tradingsymbol', '').upper()
            or query in i.get('name', '').upper()
        ][:30]

    # ----------------------------------------------------------
    # PORTFOLIO
    # ----------------------------------------------------------

    def get_holdings(self) -> List[Dict]:
        """
        Get long-term portfolio holdings (equity delivery positions).
        Returns list with: tradingsymbol, exchange, quantity, average_price,
        last_price, pnl, day_change, day_change_percentage, product, etc.

        Silent-failure variant — returns [] on any error. Prefer
        ``get_holdings_with_diagnostic()`` when the caller needs to surface
        a token-expiry / connectivity failure to the UI.
        """
        holdings, _ = self.get_holdings_with_diagnostic()
        return holdings

    def get_holdings_with_diagnostic(self) -> Tuple[List[Dict], Optional[str]]:
        """
        Like ``get_holdings()`` but returns ``(holdings, error_message)``.

        ``error_message`` is None on success or a short, user-facing string
        describing the failure (token expired, network, client uninitialised).
        TokenException — the daily access-token expiry — is recognised and
        translated into an actionable "please re-login" message.
        """
        if not self.kite:
            return [], (
                "Kite client not initialised — check api_key / access_token "
                "in local.json or restart the server."
            )
        try:
            return self.kite.holdings(), None
        except Exception as e:
            cls_name = type(e).__name__
            msg = str(e) or repr(e)
            logger.error(f"Holdings error: {cls_name}: {msg}")
            if "Token" in cls_name or "token" in msg.lower():
                return [], (
                    "Kite access token expired or invalid — please re-login "
                    "(Kite tokens auto-revoke at the daily session boundary)."
                )
            return [], f"{cls_name}: {msg}"

    def get_positions(self) -> Dict:
        """
        Get current intraday/F&O positions.
        Returns {'day': [...], 'net': [...]}
        Each entry: tradingsymbol, exchange, product, quantity, average_price,
        last_price, pnl, unrealised, realised, etc.
        """
        if not self.kite:
            return {'day': [], 'net': []}
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return {'day': [], 'net': []}

    def get_profile(self) -> Dict:
        """Get logged-in user profile."""
        if not self.kite:
            return {}
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Profile error: {e}")
            return {}

    def get_margins(self) -> Dict:
        """
        Get account margins.
        Returns {'equity': {'available': {...}, 'utilised': {...}}, 'commodity': {...}}
        """
        if not self.kite:
            return {}
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Margins error: {e}")
            return {}

    def get_available_cash(self) -> float:
        """Get available cash balance."""
        margins = self.get_margins()
        try:
            return margins.get('equity', {}).get('available', {}).get('cash', 0.0)
        except Exception:
            return 0.0

    # ----------------------------------------------------------
    # ORDERS
    # ----------------------------------------------------------

    def get_orders(self) -> List[Dict]:
        """
        Get all orders for the day.
        Each order: order_id, tradingsymbol, exchange, transaction_type,
        order_type, product, quantity, price, trigger_price, status,
        filled_quantity, pending_quantity, placed_at, variety, tag, etc.
        """
        if not self.kite:
            return []
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Orders error: {e}")
            return []

    def get_order_history(self, order_id: str) -> List[Dict]:
        """Get full status history of an order."""
        if not self.kite:
            return []
        try:
            return self.kite.order_history(order_id)
        except Exception as e:
            logger.error(f"Order history error: {e}")
            return []

    def get_order_trades(self, order_id: str) -> List[Dict]:
        """Get trades executed for a specific order."""
        if not self.kite:
            return []
        try:
            return self.kite.order_trades(order_id)
        except Exception as e:
            logger.error(f"Order trades error: {e}")
            return []

    def get_trades(self) -> List[Dict]:
        """Get all trades executed today."""
        if not self.kite:
            return []
        try:
            return self.kite.trades()
        except Exception as e:
            logger.error(f"Trades error: {e}")
            return []

    def get_order_status(self, order_id: str) -> str:
        """Get latest status string for an order."""
        history = self.get_order_history(order_id)
        if history:
            return history[-1].get('status', 'UNKNOWN')
        return 'UNKNOWN'

    # ----------------------------------------------------------
    # ORDER PLACEMENT
    # ----------------------------------------------------------

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,   # 'BUY' or 'SELL'
        quantity: int,
        order_type: str,          # 'MARKET', 'LIMIT', 'SL', 'SL-M'
        product: str = 'CNC',     # 'CNC', 'MIS', 'NRML'
        price: float = 0,
        trigger_price: float = 0,
        stoploss: float = 0,      # for bracket orders: absolute points
        trailing_stoploss: float = 0,  # for bracket orders: trailing in points
        squareoff: float = 0,     # for bracket orders: target points
        validity: str = 'DAY',
        variety: str = 'regular', # 'regular', 'amo', 'bo', 'co', 'iceberg'
        tag: str = 'marketmind',
        disclosed_quantity: int = 0,
        iceberg_legs: int = 2,
        iceberg_quantity: int = 0,
    ) -> Optional[str]:
        """
        Universal order placement method.
        Returns order_id string on success, None on failure.

        For bracket orders (variety='bo'):
            - stoploss, squareoff, trailing_stoploss are in absolute price points
        For cover orders (variety='co'):
            - trigger_price is mandatory
        """
        if not self.kite:
            logger.warning("KiteConnect not initialized")
            return None

        try:
            params = dict(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                validity=validity,
                variety=variety,
                tag=tag,
            )

            if price > 0:
                params['price'] = price
            if trigger_price > 0:
                params['trigger_price'] = trigger_price
            if disclosed_quantity > 0:
                params['disclosed_quantity'] = disclosed_quantity

            # Bracket order parameters
            if variety == 'bo':
                if stoploss > 0:
                    params['stoploss'] = stoploss
                if squareoff > 0:
                    params['squareoff'] = squareoff
                if trailing_stoploss > 0:
                    params['trailing_stoploss'] = trailing_stoploss

            # Iceberg order parameters
            if variety == 'iceberg':
                params['iceberg_legs'] = iceberg_legs
                if iceberg_quantity > 0:
                    params['iceberg_quantity'] = iceberg_quantity

            order_id = self.kite.place_order(**params)
            logger.info(f"Order placed: {order_id} | {transaction_type} {quantity} {tradingsymbol} @ {price}")
            return order_id

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

    def place_market_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = 'CNC',
        tag: str = 'marketmind'
    ) -> Optional[str]:
        """Place a market order."""
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type='MARKET',
            product=product,
            tag=tag,
        )

    def place_limit_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        product: str = 'CNC',
        tag: str = 'marketmind'
    ) -> Optional[str]:
        """Place a limit order."""
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type='LIMIT',
            product=product,
            price=price,
            tag=tag,
        )

    def place_sl_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        trigger_price: float,
        product: str = 'CNC',
        tag: str = 'marketmind'
    ) -> Optional[str]:
        """Place a stop-loss (SL) limit order."""
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type='SL',
            product=product,
            price=price,
            trigger_price=trigger_price,
            tag=tag,
        )

    def place_slm_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        trigger_price: float,
        product: str = 'CNC',
        tag: str = 'marketmind'
    ) -> Optional[str]:
        """Place a stop-loss market (SL-M) order."""
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type='SL-M',
            product=product,
            trigger_price=trigger_price,
            tag=tag,
        )

    def place_bracket_order_with_trailing_sl(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        stoploss_points: float,
        target_points: float,
        trailing_stoploss_points: float = 0,
        tag: str = 'marketmind_auto'
    ) -> Optional[str]:
        """
        Place a bracket order (BO) with optional trailing stop-loss.
        Bracket orders are MIS and auto square-off at end of day.

        stoploss_points:          absolute price points below entry for stop-loss
        target_points:            absolute price points above entry for target
        trailing_stoploss_points: trailing stop-loss in price points (0 = disabled)
        """
        return self.place_order(
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type='LIMIT',
            product='MIS',   # Bracket orders must be MIS
            price=price,
            stoploss=stoploss_points,
            squareoff=target_points,
            trailing_stoploss=trailing_stoploss_points,
            variety='bo',
            tag=tag,
        )

    def modify_order(
        self,
        order_id: str,
        variety: str = 'regular',
        quantity: int = None,
        price: float = None,
        order_type: str = None,
        trigger_price: float = None,
        validity: str = None,
        disclosed_quantity: int = None,
    ) -> Optional[str]:
        """Modify an existing pending order."""
        if not self.kite:
            return None
        try:
            params = {'order_id': order_id, 'variety': variety}
            if quantity is not None:
                params['quantity'] = quantity
            if price is not None:
                params['price'] = price
            if order_type is not None:
                params['order_type'] = order_type
            if trigger_price is not None:
                params['trigger_price'] = trigger_price
            if validity is not None:
                params['validity'] = validity
            if disclosed_quantity is not None:
                params['disclosed_quantity'] = disclosed_quantity
            return self.kite.modify_order(**params)
        except Exception as e:
            logger.error(f"Modify order error: {e}")
            return None

    def cancel_order(self, order_id: str, variety: str = 'regular') -> Optional[str]:
        """Cancel a pending order."""
        if not self.kite:
            return None
        try:
            return self.kite.cancel_order(variety=variety, order_id=order_id)
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return None

    def exit_bracket_order(self, order_id: str, parent_order_id: str = None) -> Optional[str]:
        """Exit a bracket order position."""
        if not self.kite:
            return None
        try:
            kwargs = {'variety': 'bo', 'order_id': order_id}
            if parent_order_id:
                kwargs['parent_order_id'] = parent_order_id
            return self.kite.cancel_order(**kwargs)
        except Exception as e:
            logger.error(f"Exit bracket order error: {e}")
            return None


# ============================================================
# AUTOMATED BUYER LOOP (BUYER LOOPING ONLY)
# ============================================================

class AutoTrader:
    """
    Automated trading module — BUYER LOOPING ONLY.

    This module:
    - Processes BUY signals from the RL engine
    - Places bracket orders with trailing stop-loss automatically
    - Monitors active orders for status changes
    - NEVER places SELL orders autonomously (manual sell only)

    Enable with: start_buyer_loop()
    Must be explicitly started; off by default.
    """

    def __init__(self, kite_client: KiteClient, config: KiteConfig):
        self.kite = kite_client
        self.config = config
        self.buyer_loop_active = False
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._active_orders: Dict[str, Dict] = {}   # order_id -> details
        self._orders_lock = threading.Lock()
        self._signals: List[Dict] = []
        self._signals_lock = threading.Lock()

        # Callbacks
        self._order_placed_callbacks: List[Callable] = []
        self._order_executed_callbacks: List[Callable] = []
        self._order_failed_callbacks: List[Callable] = []

    def on_order_placed(self, callback: Callable):
        """callback(order_id, signal_dict)"""
        self._order_placed_callbacks.append(callback)

    def on_order_executed(self, callback: Callable):
        """callback(order_id, order_dict, status)"""
        self._order_executed_callbacks.append(callback)

    def on_order_failed(self, callback: Callable):
        """callback(signal_dict, error)"""
        self._order_failed_callbacks.append(callback)

    def _fire(self, callbacks, *args):
        for cb in callbacks:
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"AutoTrader callback error: {e}")

    def start_buyer_loop(self):
        """Start the automated buyer loop (BUY signals only)."""
        if self.buyer_loop_active:
            logger.info("Buyer loop already running")
            return
        if not self.config.auto_trading_enabled:
            logger.warning("Auto trading is disabled in config. Enable auto_trading_enabled in local.json")
            return
        if not self.kite.is_connected:
            logger.warning("KiteConnect not authenticated - cannot start buyer loop")
            return

        self.buyer_loop_active = True
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, name="AutoTraderBuyerLoop", daemon=True)
        self._thread.start()
        logger.info("AutoTrader buyer loop STARTED")

    def stop_buyer_loop(self):
        """Stop the buyer loop."""
        self.buyer_loop_active = False
        self.is_running = False
        logger.info("AutoTrader buyer loop STOPPED")

    def add_buy_signal(
        self,
        symbol: str,
        exchange: str,
        price: float,
        quantity: int,
        confidence: float,
        reason: str = '',
        tag: str = 'marketmind_auto'
    ):
        """
        Enqueue a BUY signal for the buyer loop to process.
        Only signals with confidence >= min_confidence_threshold are acted on.
        """
        with self._signals_lock:
            self._signals.append({
                'symbol': symbol,
                'exchange': exchange,
                'price': price,
                'quantity': quantity,
                'confidence': confidence,
                'reason': reason,
                'tag': tag,
                'timestamp': datetime.now(),
            })

    def _loop(self):
        """Main buyer loop."""
        while self.buyer_loop_active:
            try:
                self._process_signals()
                self._monitor_orders()
            except Exception as e:
                logger.error(f"Buyer loop error: {e}")
            time.sleep(self.config.buyer_loop_interval)

    def _process_signals(self):
        """Process pending BUY signals."""
        with self._signals_lock:
            signals = self._signals[:]
            self._signals.clear()

        for sig in signals:
            # Confidence gate
            if sig['confidence'] < self.config.min_confidence:
                logger.debug(f"Skipping {sig['symbol']}: confidence {sig['confidence']:.2f} below threshold")
                continue

            # Max order value guard
            qty = sig['quantity']
            order_value = sig['price'] * qty
            if order_value > self.config.max_order_value:
                qty = max(1, int(self.config.max_order_value / sig['price']))
                logger.info(f"Quantity reduced to {qty} for {sig['symbol']} (max order value)")

            # Calculate SL and target in absolute points
            sl_points = round(sig['price'] * self.config.trailing_sl_percent / 100, 2)
            tgt_points = round(sig['price'] * self.config.target_percent / 100, 2)
            trailing_sl = round(sl_points * 0.5, 2)  # trail at 50% of SL

            order_id = self.kite.place_bracket_order_with_trailing_sl(
                tradingsymbol=sig['symbol'],
                exchange=sig['exchange'],
                transaction_type='BUY',
                quantity=qty,
                price=sig['price'],
                stoploss_points=sl_points,
                target_points=tgt_points,
                trailing_stoploss_points=trailing_sl,
                tag=sig.get('tag', 'marketmind_auto'),
            )

            if order_id:
                with self._orders_lock:
                    self._active_orders[order_id] = {
                        **sig,
                        'order_id': order_id,
                        'quantity': qty,
                        'sl_points': sl_points,
                        'target_points': tgt_points,
                        'status': 'OPEN',
                        'placed_at': datetime.now(),
                    }
                self._fire(self._order_placed_callbacks, order_id, sig)
                logger.info(f"Auto-buy placed: {order_id} | {qty} {sig['symbol']} @ {sig['price']}")
            else:
                self._fire(self._order_failed_callbacks, sig, "Order placement returned None")

    def _monitor_orders(self):
        """Poll active orders and move completed ones out."""
        with self._orders_lock:
            active = dict(self._active_orders)

        completed = []
        for oid, order in active.items():
            status = self.kite.get_order_status(oid)
            if status in ('COMPLETE', 'REJECTED', 'CANCELLED'):
                completed.append(oid)
                self._fire(self._order_executed_callbacks, oid, order, status)

        with self._orders_lock:
            for oid in completed:
                self._active_orders.pop(oid, None)

    @property
    def active_orders(self) -> Dict:
        with self._orders_lock:
            return dict(self._active_orders)

    @property
    def pending_signals_count(self) -> int:
        with self._signals_lock:
            return len(self._signals)
