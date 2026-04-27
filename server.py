"""
MarketMind AI - FastAPI Web Server
Serves the browser-based frontend and all API endpoints.
"""

import sys
import os
import json
import time
import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Core modules ──────────────────────────────────────────────────────────────
from marketmind.app_controller import AppController
from marketmind.core.options_fetcher import get_options_fetcher
from marketmind.core.earnings_calendar import get_earnings_calendar
from marketmind.core.news_clusterer import cluster_news
from marketmind.core.backtester import get_backtester
from marketmind.core.walkforward import get_walkforward
from marketmind.core.debate import get_debate_engine
from marketmind.core.filings_ingest import get_filings_ingester
from marketmind.core.grounded_research import get_grounded_researcher
from marketmind.core.event_poller import get_event_poller
from marketmind.core.kite_candles import get_kite_candles
from marketmind.core.fii_dii_fetcher import get_fii_dii_fetcher
from marketmind.core.bulk_deals_fetcher import get_bulk_deals_fetcher
from marketmind.core.macro_fetcher import get_macro_fetcher
from marketmind.core.regime_classifier import get_regime_classifier
from marketmind.core.altdata.aggregator import get_aggregator as get_altdata_aggregator
from marketmind.ml.forecast.ensemble import EnsembleForecaster
from marketmind.ml.forecast.cache import get_forecast_cache
from marketmind.ml.forecast.conformal import SplitConformalWrapper
from marketmind.ml.forecast.meta_stacker import get_meta_stacker
from marketmind.analysis.risk_engine import get_risk_engine
from marketmind.analysis.portfolio_optimizer import get_optimizer
from marketmind.core.claude_research import generate_research_report, get_assistant

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── App & state ───────────────────────────────────────────────────────────────
app = FastAPI(title="MarketMind AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=8)
controller = AppController()

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ── Static files ──────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── Background broadcaster ────────────────────────────────────────────────────
async def _background_loop():
    """Push live data to all WebSocket clients every 30s."""
    while True:
        try:
            if manager.active:
                loop = asyncio.get_event_loop()
                indices = await loop.run_in_executor(executor, _get_indices_sync)
                await manager.broadcast({"type": "indices", "data": indices})
        except Exception as e:
            logger.debug(f"WS broadcast error: {e}")
        await asyncio.sleep(30)


# ── Alt-data warming loop (W2.3) ──────────────────────────────────────────────
# Most alt-data sources update weekly/monthly, so we refresh every 6h. The
# aggregator persists each AltSignal to Mongo `alt_signals` (TTL 7d). On boot
# we wait 30s before the first warm so health-checkers don't hammer Reddit
# during startup races.
async def _altdata_warm_loop():
    await asyncio.sleep(30)
    while True:
        try:
            agg = get_altdata_aggregator(mongo_col=controller._mongo_col("alt_signals"))
            await _run(agg.get_all)
            logger.info("alt-data warmed")
        except Exception as e:
            logger.warning(f"alt-data warm error: {e}")
        await asyncio.sleep(6 * 3600)


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, controller.initialize)
    loop.run_in_executor(executor, controller.start_background_updates)
    asyncio.create_task(_background_loop())
    # W2.2 event-driven trader: poll NSE corp announcements every 60s
    asyncio.create_task(get_event_poller().run_loop(broadcast=manager.broadcast,
                                                    executor=executor))
    # W2.3 alt-data warmer: refresh Reddit/ValuePickr/SIAM/GST/IIP-CPI/Trends every 6h
    asyncio.create_task(_altdata_warm_loop())
    logger.info("MarketMind server started — http://localhost:8000")


@app.on_event("shutdown")
async def shutdown():
    controller.stop_background_updates()


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
                if isinstance(payload, dict) and payload.get("type") == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run(fn, *args, **kwargs):
    """Run sync function in thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


def _get_indices_sync() -> List[Dict]:
    out = []
    for idx in ['NIFTY500', 'SENSEX', 'NIFTYBANK', 'INDIA VIX']:
        d = controller.get_index_data(idx)
        if d:
            out.append({'symbol': idx, **d})
    return out


def _safe(v):
    """Make value JSON-serialisable including numpy scalars, datetime, and NaN/Inf."""
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    try:
        import math, numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(v, np.ndarray):
            return [_safe(x) for x in v.tolist()]
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    except Exception:
        return v


def _sanitize(obj):
    if obj is None:
        return None
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return [_sanitize(x) for x in obj.tolist()]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            import math; f = float(obj)
            if math.isnan(f): return 0.0
            if math.isinf(f): return 999.0 if f > 0 else -999.0
            return f
    except ImportError:
        pass
    if isinstance(obj, float):
        import math
        if math.isnan(obj): return 0.0
        if math.isinf(obj): return 999.0 if obj > 0 else -999.0
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return _safe(obj)


# ═════════════════════════════════════════════════════════════════════════════
# MARKET
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/market/status")
async def market_status():
    result = await _run(controller.get_market_status)
    return JSONResponse(_sanitize(result))


@app.get("/api/market/indices")
async def market_indices():
    indices = await _run(_get_indices_sync)
    return JSONResponse(_sanitize(indices))


_NIFTY500_CACHE: Dict[str, Any] = {'data': None, 'ts': 0.0}
_NIFTY500_TTL = 24 * 3600  # constituents change quarterly; refresh daily is plenty


def _get_nifty500_constituents() -> List[Dict[str, str]]:
    """Live Nifty 500 constituents from NSE — cached 24h, with hardcoded fallback."""
    now = time.time()
    if _NIFTY500_CACHE['data'] and (now - _NIFTY500_CACHE['ts']) < _NIFTY500_TTL:
        return _NIFTY500_CACHE['data']

    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()
    try:
        raw = pf._nse_get('equity-stockIndices', params={'index': 'NIFTY 500'})
        if raw and isinstance(raw.get('data'), list):
            out = []
            for row in raw['data']:
                sym = (row.get('symbol') or '').strip()
                # Skip the index header row and any blanks
                if not sym or sym.upper() in ('NIFTY 500', 'NIFTY500'):
                    continue
                out.append({
                    'symbol': sym,
                    'sector': (row.get('industry') or row.get('meta', {}).get('industry') or 'Others').strip() or 'Others',
                })
            if len(out) > 100:
                _NIFTY500_CACHE['data'] = out
                _NIFTY500_CACHE['ts'] = now
                return out
    except Exception as e:
        logger.debug(f"NSE NIFTY 500 fetch failed, using fallback: {e}")

    fallback = [{'symbol': s, 'sector': 'Others'} for s in SCREENER_UNIVERSE]
    _NIFTY500_CACHE['data'] = fallback
    _NIFTY500_CACHE['ts'] = now
    return fallback


@app.get("/api/market/heatmap")
async def market_heatmap():
    """Nifty 500 stocks with % change for heatmap."""
    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()

    def fetch():
        constituents = _get_nifty500_constituents()
        sector_map = {c['symbol']: c['sector'] for c in constituents}
        symbols = [c['symbol'] for c in constituents]
        results = []

        # ── Primary: Kite OHLC (instant, reliable) ──
        kite = controller.kite
        if kite and kite.is_connected:
            try:
                # Kite get_ohlc accepts up to ~500 symbols per call
                kite_syms = [f"NSE:{s}" for s in symbols]
                ohlc_data = {}
                # Chunk by 200 to stay safely under Kite's request size limit
                for i in range(0, len(kite_syms), 200):
                    chunk = kite_syms[i:i + 200]
                    try:
                        part = kite.get_ohlc(chunk) or {}
                        ohlc_data.update(part)
                    except Exception as e:
                        logger.debug(f"Kite OHLC chunk {i} failed: {e}")
                if ohlc_data:
                    for sym in symbols:
                        key = f"NSE:{sym}"
                        if key in ohlc_data:
                            d = ohlc_data[key]
                            ohlc = d.get('ohlc', {})
                            ltp = d.get('last_price', 0)
                            prev = ohlc.get('close', ltp)
                            change_pct = round((ltp - prev) / prev * 100, 2) if prev else 0
                            results.append({
                                'symbol': sym,
                                'name': sym,
                                'price': ltp,
                                'change_pct': change_pct,
                                'market_cap': 0,
                                'sector': sector_map.get(sym, 'Others'),
                            })
                    if len(results) > 10:
                        return results
            except Exception as e:
                logger.debug(f"Kite OHLC error: {e}")

        # ── Fallback: NSE bulk equity-stockIndices already gives prices ──
        if not results:
            try:
                raw = pf._nse_get('equity-stockIndices', params={'index': 'NIFTY 500'})
                if raw and isinstance(raw.get('data'), list):
                    for row in raw['data']:
                        sym = (row.get('symbol') or '').strip()
                        if not sym or sym.upper() in ('NIFTY 500', 'NIFTY500'):
                            continue
                        ltp = float(row.get('lastPrice') or 0)
                        chg = float(row.get('pChange') or 0)
                        results.append({
                            'symbol': sym,
                            'name': sym,
                            'price': ltp,
                            'change_pct': round(chg, 2),
                            'market_cap': 0,
                            'sector': sector_map.get(sym, 'Others'),
                        })
            except Exception as e:
                logger.debug(f"NSE bulk fetch fallback failed: {e}")

        # ── Demo fallback so UI always shows something ──
        if len(results) < 5:
            import random, hashlib
            for sym in symbols:
                if any(r['symbol'] == sym for r in results):
                    continue
                seed = int(hashlib.md5(sym.encode()).hexdigest()[:6], 16)
                rng2 = random.Random(seed)
                change_pct = round(rng2.gauss(0.1, 1.5), 2)
                results.append({
                    'symbol': sym,
                    'name': sym,
                    'price': 0,
                    'change_pct': change_pct,
                    'market_cap': 0,
                    'sector': sector_map.get(sym, 'Others'),
                    'demo': True,
                })
        return results

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# NEWS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/news")
async def get_news(sector: Optional[str] = None, limit: int = 40):
    news = await _run(controller.get_latest_news)
    if sector:
        news = [n for n in news if sector.lower() in [s.lower() for s in n.get('sectors', [])]]
    return JSONResponse(_sanitize(news[:limit]))


@app.get("/api/news/clustered")
async def get_clustered_news(limit: int = 30):
    news = await _run(controller.get_latest_news)
    clustered = cluster_news(news, threshold=0.3)
    return JSONResponse(_sanitize(clustered[:limit]))


# ═════════════════════════════════════════════════════════════════════════════
# STOCKS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/stocks/{symbol}")
async def get_stock(symbol: str):
    data = await _run(controller.get_stock_data, symbol.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    return JSONResponse(_sanitize(data))


@app.get("/api/stocks/{symbol}/options")
async def get_options(symbol: str):
    fetcher = get_options_fetcher()
    data = await _run(fetcher.get_option_chain, symbol.upper())
    return JSONResponse(_sanitize(data))


# ─── Options strategy builder (W3.3) ──────────────────────────────────────

class _StrategyLeg(BaseModel):
    action: str  # "BUY" | "SELL"
    kind: str    # "CE" | "PE"
    strike: float
    premium: float = 0.0
    iv: float = 0.0           # decimal (0.18 = 18%)
    qty: int = 1
    expiry_days: int = 30


class StrategyRequest(BaseModel):
    symbol: str
    strategy: str
    expiry_days: int = 30
    lots: int = 1
    lot_size: int = 1
    legs: Optional[List[_StrategyLeg]] = None
    underlying: Optional[float] = None  # override (e.g. tests, what-if scenarios)
    back_expiry_days: Optional[int] = None  # required for calendar_spread


@app.post("/api/options/strategy")
async def options_strategy(req: StrategyRequest):
    """Build / analyse an options strategy. See plan W3.3."""
    from marketmind.ml.options.strategies import (
        ALL_STRATEGIES,
        build_default_legs,
    )
    from marketmind.ml.options.builder import analyse

    if req.strategy not in ALL_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {req.strategy}")
    if req.strategy == "calendar_spread" and req.back_expiry_days is None:
        raise HTTPException(
            status_code=400,
            detail="calendar_spread requires back_expiry_days",
        )

    fetcher = get_options_fetcher()

    def build_and_analyse():
        chain = fetcher.get_option_chain(req.symbol.upper())
        underlying = req.underlying or float(chain.get("underlying") or chain.get("atm_strike") or 0)

        if req.legs:
            legs = [leg.dict() for leg in req.legs]
        else:
            if chain.get("unavailable"):
                return {
                    "strategy": req.strategy,
                    "symbol": req.symbol.upper(),
                    "unavailable": True,
                    "reason": chain.get("reason", "options chain unavailable"),
                }
            legs = build_default_legs(
                req.strategy, chain,
                expiry_days=req.expiry_days,
                lots=req.lots,
                lot_size=req.lot_size,
            )
            # Calendar back-leg expiry override.
            if req.strategy == "calendar_spread" and req.back_expiry_days is not None:
                legs[1]["expiry_days"] = int(req.back_expiry_days)

        if underlying <= 0:
            raise ValueError("could not determine underlying price")

        multi_expiry = req.strategy == "calendar_spread"
        result = analyse(legs, underlying, strategy_name=req.strategy, multi_expiry=multi_expiry)
        result["symbol"] = req.symbol.upper()
        result["iv_rank_hint"] = None  # IVR computation deferred to dedicated wave
        return result

    try:
        data = await _run(build_and_analyse)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(_sanitize(data))


@app.get("/api/stocks/{symbol}/history")
async def get_history(symbol: str, days: int = 365):
    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()

    def fetch():
        df = pf.get_historical_data(symbol.upper(), days=days)
        if df.empty:
            return []
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df.to_dict('records')

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# SCREENER
# ═════════════════════════════════════════════════════════════════════════════

# Nifty500 + Midcap150 + Smallcap250 representative universe
SCREENER_UNIVERSE = [
    # Nifty50 large caps
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN',
    'BHARTIARTL', 'ITC', 'LT', 'HINDUNILVR', 'AXISBANK', 'KOTAKBANK',
    'MARUTI', 'TATAMOTORS', 'M&M', 'TATASTEEL', 'SUNPHARMA', 'ULTRACEMCO',
    'NESTLEIND', 'ONGC', 'POWERGRID', 'NTPC', 'COALINDIA', 'BAJFINANCE',
    'BAJAJFINSV', 'HCLTECH', 'WIPRO', 'TECHM', 'ADANIPORTS', 'ASIANPAINT',
    'TITAN', 'DIVISLAB', 'CIPLA', 'DRREDDY', 'HEROMOTOCO', 'EICHERMOT',
    'JSWSTEEL', 'HINDALCO', 'VEDL', 'INDUSINDBK', 'APOLLOHOSP', 'BPCL',
    'BRITANNIA', 'DLF', 'GODREJPROP', 'GRASIM', 'HDFCLIFE', 'LUPIN',
    'AUROPHARMA', 'SHREECEM',
    # Nifty Midcap150
    'PERSISTENT', 'LTIM', 'MPHASIS', 'COFORGE', 'LTTS', 'KPITTECH',
    'HAPPSTMNDS', 'TATAELXSI', 'CYIENT', 'MASTEK',
    'CHOLAFIN', 'MUTHOOTFIN', 'SBICARD', 'MANAPPURAM', 'LICHSGFIN',
    'PIIND', 'AAVAS', 'CREDITACC',
    'ASTRAL', 'POLYCAB', 'KEI', 'HAVELLS', 'VOLTAS', 'BLUESTARCO',
    'CROMPTON', 'AMBER', 'ORIENTELEC',
    'CAMS', 'CDSL', 'MCX', 'BSE',
    'ATUL', 'DEEPAKNTR', 'GNFC', 'NAVINFLUOR', 'FINEORG',
    'LALPATHLAB', 'METROPOLIS', 'ERIS', 'IPCALAB', 'ALKEM',
    'SAIL', 'NMDC', 'MOIL', 'NATIONALUM', 'HINDCOPPER',
    'TATACOMM', 'HFCL', 'STERLITE', 'RAILTEL',
    'APOLLOTYRE', 'BALKRISIND', 'MRF', 'CEATLTD',
    'TORNTPHARM', 'ABBOTINDIA', 'PFIZER', 'GLAXO',
    'SUNDARMFIN', 'MOTILALOFS', 'ANGELONE', 'NUVAMA',
    'PAGEIND', 'VBL', 'RADICO', 'MCDOWELL-N', 'UNITDSPR',
    'BATAINDIA', 'RELAXO', 'CAMPUS',
    'ZYDUSLIFE', 'GLENMARK', 'NATCOPHARM', 'STRIDES',
    'MAHINDCIE', 'MOTHERSON', 'SUNDRMFAST', 'EXIDEIND',
    'GMRINFRA', 'IRB', 'NHAI', 'PNC',
    'GPPL', 'CONCOR', 'BLUEDART',
    'SUNTV', 'ZEEL', 'PVRINOX', 'NAZARA',
    # Nifty Smallcap250
    'IDEAFORGE', 'PARAS', 'RATEGAIN', 'LATENTVIEW', 'NUVOCO',
    'IFCI', 'IREDA', 'IRFC', 'REC', 'PFC',
    'SJVN', 'NHPC', 'CESC', 'TORNTPOWER', 'TATAPOWER',
    'ADANIGREEN', 'ADANITRANS', 'ADANIENT',
    'SUZLON', 'INOXWIND', 'ORIENTGREEN',
    'JINDALSAW', 'RATNAMANI', 'WELSPUN', 'JAMNAAUTO',
    'GPIL', 'KALYANKJIL', 'SENCO', 'RAJESHEXPO',
    'SAREGAMA', 'TIPS', 'EROS',
    'ZOMATO', 'NYKAA', 'CARTRADE', 'EASEMYTRIP',
    'PAYTM', 'POLICYBZR', 'DELHIVERY',
    'MAHLIFE', 'SUNTECHREIT', 'BROOKFIELD',
    'RVNL', 'IRCON', 'RITES', 'TITAGARH', 'RAILVIKAS',
    'INDIGRID', 'POWERMECH', 'KEC', 'KALPATPOWR',
    'VINATIORGA', 'CLEAN', 'BALRAMCHIN', 'DALMIA',
    'SOMANYCERA', 'ORIENTCERAMICS', 'KAJARIA',
    'WOCKPHARMA', 'GRANULES', 'SUVEN', 'NEULANDLAB',
    'SEQUENT', 'DIVILAB', 'CAPLIPOINT',
]


@app.get("/api/screener")
async def screener(
    pe_max: Optional[float] = None,
    pe_min: Optional[float] = None,
    roe_min: Optional[float] = None,
    roce_min: Optional[float] = None,
    momentum_min: Optional[float] = None,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    market_cap_min: Optional[float] = None,
    sort_by: str = 'momentum',
    limit: int = 20,
):
    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()

    def fetch():
        results = []

        # Bulk Kite LTP for price data when authenticated
        kite_ltp = {}
        kite = controller.kite
        if kite and kite.is_connected:
            try:
                batch = [f"NSE:{s}" for s in SCREENER_UNIVERSE]
                ohlc_raw = kite.get_ohlc(batch)
                for sym in SCREENER_UNIVERSE:
                    key = f"NSE:{sym}"
                    if key in ohlc_raw:
                        d = ohlc_raw[key]
                        ohlc = d.get('ohlc', {})
                        ltp = d.get('last_price', 0)
                        prev = ohlc.get('close', ltp)
                        kite_ltp[sym] = {
                            'current_price': ltp,
                            'change_pct': (ltp - prev) / prev if prev else 0,
                            'name': sym,
                        }
            except Exception as e:
                logger.debug(f"Kite screener LTP error: {e}")

        def _score_symbol(sym):
            price = kite_ltp.get(sym) or pf.get_stock_price(sym)
            if not price or not price.get('current_price'):
                return None
            # If price came from Kite (no fundamentals), enrich with Screener
            if price.get('pe_ratio') is None:
                try:
                    fund = pf._get_screener_fundamentals(sym) or {}
                    for k in ('pe_ratio', 'pb_ratio', 'roe', 'roce', 'market_cap',
                              'dividend_yield', 'debt_equity'):
                        if price.get(k) is None and fund.get(k) is not None:
                            price[k] = fund[k]
                except Exception:
                    pass
            ind = pf.calculate_technical_indicators(sym, days=60)

            pe = price.get('pe_ratio')
            roe = price.get('roe')
            roce = price.get('roce')
            mktcap = price.get('market_cap')
            momentum = ind.get('momentum_20', 0)
            rsi = ind.get('rsi', 50)

            if pe_max is not None and (pe is None or pe > pe_max): return None
            if pe_min is not None and (pe is None or pe < pe_min): return None
            if roe_min is not None and (roe is None or roe < roe_min): return None
            if roce_min is not None and (roce is None or roce < roce_min): return None
            if momentum_min is not None and momentum < momentum_min / 100: return None
            if rsi_min is not None and rsi < rsi_min: return None
            if rsi_max is not None and rsi > rsi_max: return None
            if market_cap_min is not None and (mktcap is None or mktcap < market_cap_min * 1e7): return None

            score = 0
            if roe and roe > 0: score += min(roe, 40)
            if pe and 5 < pe < 50: score += max(0, 20 - pe * 0.4)
            score += momentum * 100
            if 40 < rsi < 65: score += 10

            cp = price.get('change_pct', 0)
            return {
                'symbol': sym,
                'name': price.get('name', sym),
                'price': price.get('current_price', 0),
                'change_pct': round(cp * 100, 2) if abs(cp) < 1 else round(cp, 2),
                'pe': pe,
                'pb': price.get('pb_ratio'),
                'roe': roe,
                'roce': roce,
                'market_cap_cr': round(mktcap / 1e7, 0) if mktcap else None,
                'rsi': round(rsi, 1),
                'momentum_20d': round(momentum * 100, 2),
                'macd': round(ind.get('macd', 0), 3),
                'above_ma50': ind.get('above_ma_50', False),
                'score': round(score, 1),
            }

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
        ex = ThreadPoolExecutor(max_workers=8)
        try:
            futures = {ex.submit(_score_symbol, sym): sym for sym in SCREENER_UNIVERSE}
            try:
                for fut in as_completed(futures, timeout=15):
                    try:
                        row = fut.result(timeout=5)
                        if row:
                            results.append(row)
                    except (FutureTimeout, Exception) as e:
                        logger.debug(f"Screener skip {futures[fut]}: {e}")
            except FutureTimeout:
                logger.warning(f"Screener overall timeout — returning {len(results)} rows so far")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        if sort_by == 'roe':
            results.sort(key=lambda x: x.get('roe') or 0, reverse=True)
        elif sort_by == 'pe':
            results.sort(key=lambda x: x.get('pe') or 999)
        elif sort_by == 'momentum':
            results.sort(key=lambda x: x.get('momentum_20d') or 0, reverse=True)
        elif sort_by == 'score':
            results.sort(key=lambda x: x.get('score') or 0, reverse=True)

        # If no real data, return demo screener results
        if not results:
            import hashlib, random
            demo_stocks = [
                ('RELIANCE', 'Reliance Industries', 2845, 28.5, 4.2, 18.5, 14.2, 0.3, 62, True),
                ('TCS', 'Tata Consultancy Services', 3920, 35.2, 12.1, 45.6, 38.2, 0.15, 55, True),
                ('INFY', 'Infosys Ltd', 1485, 22.1, 7.8, 28.4, 24.5, -0.12, 42, True),
                ('HDFCBANK', 'HDFC Bank', 1640, 18.2, 2.8, 15.6, 13.8, 0.42, 58, True),
                ('ICICIBANK', 'ICICI Bank', 1095, 15.4, 2.2, 17.8, 16.1, 0.55, 64, True),
                ('PERSISTENT', 'Persistent Systems', 5400, 42.1, 8.9, 31.2, 28.4, 1.82, 68, True),
                ('MPHASIS', 'Mphasis Ltd', 2150, 30.5, 6.2, 24.8, 22.1, 0.92, 60, True),
                ('COFORGE', 'Coforge Ltd', 6200, 38.2, 7.8, 28.5, 25.2, 1.45, 72, True),
                ('CHOLAFIN', 'Chola Finance', 1320, 25.8, 4.1, 16.4, 14.8, 0.68, 61, True),
                ('POLYCAB', 'Polycab India', 5800, 36.4, 8.2, 22.6, 19.8, 0.85, 58, True),
            ]
            for sym, name, price, pe, pb, roe, roce, mom, rsi_val, above_ma in demo_stocks:
                score = min(roe, 40) + max(0, 20 - pe * 0.4) + mom * 10 + (10 if 40 < rsi_val < 65 else 0)
                stock = {
                    'symbol': sym, 'name': name, 'price': price,
                    'change_pct': round(random.Random(hash(sym)).gauss(0.2, 1.2), 2),
                    'pe': pe, 'pb': pb, 'roe': roe, 'roce': roce,
                    'market_cap_cr': None, 'rsi': rsi_val,
                    'momentum_20d': round(mom * 100, 2), 'macd': 0,
                    'above_ma50': above_ma, 'score': round(score, 1), 'demo': True,
                }
                # Apply filters on demo data too
                if pe_max is not None and pe > pe_max: continue
                if pe_min is not None and pe < pe_min: continue
                if roe_min is not None and roe < roe_min: continue
                if rsi_min is not None and rsi_val < rsi_min: continue
                if rsi_max is not None and rsi_val > rsi_max: continue
                results.append(stock)

        return results[:limit]

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# SECTORS
# ═════════════════════════════════════════════════════════════════════════════

_SECTOR_FALLBACK = {
    'IT':      {'name': 'Information Technology', 'sentiment': 0.15, 'sentiment_label': 'Positive', 'change_pct': 0.3},
    'Banking': {'name': 'Banking & Finance',      'sentiment': 0.05, 'sentiment_label': 'Neutral',  'change_pct': 0.1},
    'Auto':    {'name': 'Automobile',             'sentiment': 0.10, 'sentiment_label': 'Positive', 'change_pct': 0.2},
    'Pharma':  {'name': 'Pharmaceuticals',        'sentiment': 0.00, 'sentiment_label': 'Neutral',  'change_pct': 0.0},
    'FMCG':    {'name': 'FMCG & Consumer',        'sentiment':-0.05, 'sentiment_label': 'Negative', 'change_pct':-0.1},
    'Metal':   {'name': 'Metals & Mining',        'sentiment':-0.10, 'sentiment_label': 'Negative', 'change_pct':-0.2},
    'Energy':  {'name': 'Energy & Oil',           'sentiment': 0.08, 'sentiment_label': 'Neutral',  'change_pct': 0.15},
    'Realty':  {'name': 'Real Estate',            'sentiment': 0.12, 'sentiment_label': 'Positive', 'change_pct': 0.25},
    'Finance': {'name': 'Financial Services',     'sentiment': 0.07, 'sentiment_label': 'Neutral',  'change_pct': 0.14},
}

@app.get("/api/sectors")
async def get_sectors():
    try:
        data = await _run(controller.get_sector_data)
        if data:
            return JSONResponse(_sanitize(data))
    except Exception as e:
        logger.debug(f"Sector data error: {e}")
    # Always return something so D3 graph renders
    return JSONResponse(_SECTOR_FALLBACK)


@app.get("/api/sectors/correlations")
async def get_correlations():
    def fetch():
        raw = controller.get_all_correlations()
        # raw may be a dict of DataFrames or a correlation_matrix DataFrame
        import pandas as pd, numpy as np
        result = {}
        if isinstance(raw, pd.DataFrame):
            # Convert DataFrame to nested Python dict with float values
            matrix = {}
            for col in raw.columns:
                matrix[str(col)] = {
                    str(idx): (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v))
                    for idx, v in raw[col].items()
                }
            return {'correlation_matrix': matrix}
        if isinstance(raw, dict):
            # May contain DataFrames as values
            matrix = {}
            for k, v in raw.items():
                if isinstance(v, pd.DataFrame):
                    sub = {}
                    for col in v.columns:
                        sub[str(col)] = {
                            str(idx): (None if (isinstance(fv, float) and (np.isnan(fv) or np.isinf(fv))) else float(fv))
                            for idx, fv in v[col].items()
                        }
                    matrix[str(k)] = sub
                elif isinstance(v, (int, float, str, bool, type(None))):
                    matrix[str(k)] = v
            return matrix or raw
        return {}
    try:
        data = await _run(fetch)
    except Exception:
        data = {}
    return JSONResponse(data or {})


@app.get("/api/sectors/{sector}/recommendations")
async def sector_recommendations(sector: str):
    """
    Return top stock recommendations for a sector with full analysis.
    Called when user clicks a node in the sector correlation graph.
    """
    from marketmind.core.price_fetcher import get_price_fetcher
    from marketmind.core.sector_classifier import SectorClassifier

    pf = get_price_fetcher()
    sc = SectorClassifier()

    def fetch():
        stocks = sc.get_sector_stocks(sector)
        if not stocks:
            return {'sector': sector, 'stocks': [], 'error': 'No stocks found'}

        recommendations = []
        for sym in stocks:
            try:
                price = pf.get_stock_price(sym)
                if not price or not price.get('current_price'):
                    continue
                ind = pf.calculate_technical_indicators(sym, days=90)

                # Fundamental score (0-100)
                fund_score = 0
                roe = price.get('roe') or 0
                roce = price.get('roce') or 0
                pe = price.get('pe_ratio') or 0
                pb = price.get('pb_ratio') or 0
                debt_eq = price.get('debt_equity') or 0
                rev_growth = price.get('revenue_growth') or 0
                profit_growth = price.get('profit_growth') or 0

                if roe > 15: fund_score += 20
                elif roe > 10: fund_score += 10
                if roce > 15: fund_score += 15
                elif roce > 10: fund_score += 8
                if 5 < pe < 30: fund_score += 15
                elif 30 < pe < 50: fund_score += 5
                if pb < 3: fund_score += 10
                if debt_eq < 0.5: fund_score += 10
                if rev_growth > 15: fund_score += 15
                elif rev_growth > 5: fund_score += 8
                if profit_growth > 20: fund_score += 15
                elif profit_growth > 10: fund_score += 8

                # Technical score (0-100)
                tech_score = 0
                rsi = ind.get('rsi', 50)
                momentum = ind.get('momentum_20', 0)
                above_ma50 = ind.get('above_ma_50', False)
                above_ma200 = ind.get('above_ma_200', False)
                macd = ind.get('macd', 0)
                macd_sig = ind.get('macd_signal', 0)

                if 40 < rsi < 65: tech_score += 25
                elif rsi < 40: tech_score += 15  # oversold = opportunity
                if momentum > 0.05: tech_score += 20
                elif momentum > 0: tech_score += 10
                if above_ma50: tech_score += 20
                if above_ma200: tech_score += 20
                if macd > macd_sig: tech_score += 15

                total_score = fund_score * 0.6 + tech_score * 0.4

                # Signal
                if total_score >= 70:
                    signal = 'STRONG BUY'
                    signal_color = 'green'
                elif total_score >= 50:
                    signal = 'BUY'
                    signal_color = 'lightgreen'
                elif total_score >= 35:
                    signal = 'HOLD'
                    signal_color = 'orange'
                else:
                    signal = 'AVOID'
                    signal_color = 'red'

                # Build rationale
                reasons = []
                if roe > 15:
                    reasons.append(f"Strong ROE of {roe:.1f}%")
                if pe and 10 < pe < 25:
                    reasons.append(f"Attractive PE of {pe:.1f}x")
                if momentum > 0.05:
                    reasons.append(f"Strong 20-day momentum: +{momentum*100:.1f}%")
                if above_ma200:
                    reasons.append("Trading above 200-day MA (long-term uptrend)")
                if macd > macd_sig:
                    reasons.append("MACD bullish crossover")
                _rv = price.get('revenue_growth') or 0
                if _rv > 10:
                    reasons.append(f"Revenue growth {_rv:.0f}% YoY")
                if not reasons:
                    reasons.append("Watch for entry opportunities")

                cp = price.get('current_price', 0)
                atr = ind.get('atr', cp * 0.02)

                recommendations.append({
                    'symbol': sym,
                    'name': price.get('name', sym),
                    'price': cp,
                    'change_pct': round(price.get('change_pct', 0) * 100, 2)
                        if abs(price.get('change_pct', 0)) < 1
                        else round(price.get('change_pct', 0), 2),
                    'signal': signal,
                    'signal_color': signal_color,
                    'score': round(total_score, 1),
                    'fund_score': round(fund_score, 0),
                    'tech_score': round(tech_score, 0),
                    'fundamentals': {
                        'pe': pe or None,
                        'pb': pb or None,
                        'roe': roe or None,
                        'roce': roce or None,
                        'debt_equity': debt_eq or None,
                        'revenue_growth': rev_growth or None,
                        'profit_growth': profit_growth or None,
                        'market_cap_cr': round(price.get('market_cap', 0) / 1e7)
                            if price.get('market_cap') else None,
                        'eps': price.get('eps'),
                        'div_yield': price.get('dividend_yield'),
                    },
                    'technicals': {
                        'rsi': round(rsi, 1),
                        'macd': round(macd, 3),
                        'momentum_20d': round(momentum * 100, 2),
                        'above_ma50': above_ma50,
                        'above_ma200': above_ma200,
                        'ma_20': round(ind.get('ma_20', 0), 2),
                        'ma_50': round(ind.get('ma_50', 0), 2),
                        'atr': round(atr, 2),
                        'bb_upper': round(ind.get('bb_upper', 0), 2),
                        'bb_lower': round(ind.get('bb_lower', 0), 2),
                    },
                    'entry': {
                        'suggested_buy': round(cp * 0.99, 2),
                        'target': round(cp * 1.12, 2),
                        'stop_loss': round(cp - atr * 2, 2),
                        'risk_reward': '1:3',
                        'time_horizon': 'Swing (2-4 weeks)',
                    },
                    'reasons': reasons[:4],
                    '52w_high': price.get('52_week_high'),
                    '52w_low': price.get('52_week_low'),
                })
            except Exception as e:
                logger.debug(f"Rec error for {sym}: {e}")

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Get sector info
        sector_info = controller.get_sector_details(sector)

        return {
            'sector': sector,
            'sector_name': sector_info.get('name', sector),
            'sentiment': sector_info.get('sentiment', 0),
            'signal': sector_info.get('signal', 'NEUTRAL'),
            'news': sector_info.get('news', [])[:3],
            'stocks': recommendations[:8],
            'top_picks': [r['symbol'] for r in recommendations[:3] if r['signal'] in ('BUY', 'STRONG BUY')],
        }

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# F&O SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/fo-sentiment")
async def fo_sentiment():
    """PCR-based fear/greed indicator."""
    fetcher = get_options_fetcher()

    def fetch():
        nifty = fetcher.get_option_chain('NIFTY')
        banknifty = fetcher.get_option_chain('BANKNIFTY')

        nifty_pcr = nifty.get('pcr', 1.0)
        bn_pcr = banknifty.get('pcr', 1.0)
        avg_pcr = (nifty_pcr + bn_pcr) / 2

        # Fear/Greed: PCR > 1.2 = extreme fear (contrarian bullish), < 0.7 = greed
        if avg_pcr > 1.5:
            sentiment = 'Extreme Fear'
            gauge = 10
        elif avg_pcr > 1.2:
            sentiment = 'Fear'
            gauge = 30
        elif avg_pcr > 0.9:
            sentiment = 'Neutral'
            gauge = 50
        elif avg_pcr > 0.7:
            sentiment = 'Greed'
            gauge = 70
        else:
            sentiment = 'Extreme Greed'
            gauge = 90

        return {
            'nifty_pcr': nifty_pcr,
            'banknifty_pcr': bn_pcr,
            'avg_pcr': round(avg_pcr, 3),
            'nifty_max_pain': nifty.get('max_pain', 0),
            'banknifty_max_pain': banknifty.get('max_pain', 0),
            'sentiment': sentiment,
            'gauge': gauge,
            'nifty_call_oi': nifty.get('total_call_oi', 0),
            'nifty_put_oi': nifty.get('total_put_oi', 0),
            'interpretation': (
                'High PCR = more puts = market hedging heavily = bearish undertone. '
                'PCR > 1.2 is contrarian bullish (market over-hedged).'
                if avg_pcr > 1.0
                else 'Low PCR = more calls = market is complacent = caution advised.'
            ),
        }

    try:
        data = await _run(fetch)
        if not data:
            raise ValueError("empty")
    except Exception:
        data = {
            'nifty_pcr': 1.05, 'banknifty_pcr': 0.98, 'avg_pcr': 1.02,
            'nifty_max_pain': 0, 'banknifty_max_pain': 0,
            'sentiment': 'Neutral', 'gauge': 50,
            'nifty_call_oi': 5000000, 'nifty_put_oi': 5250000,
            'interpretation': 'NSE options data temporarily unavailable. PCR ~1.0 indicates neutral market sentiment.',
        }
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# EARNINGS CALENDAR
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/earnings-calendar")
async def earnings_calendar(days: int = 30):
    cal = get_earnings_calendar()
    data = await _run(cal.get_upcoming_results, days)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def get_portfolio():
    summary = await _run(controller.get_portfolio_summary)
    return JSONResponse(_sanitize(summary))


@app.get("/api/portfolio/equity-curve")
async def equity_curve():
    """Generate portfolio equity curve from holdings history."""
    def fetch():
        holdings = controller.get_holdings()
        if not holdings:
            # Demo curve
            import random
            random.seed(42)
            points = []
            val = 100000
            from datetime import timedelta
            base = datetime.now()
            for i in range(90, -1, -1):
                d = base - timedelta(days=i)
                val *= (1 + random.gauss(0.001, 0.012))
                points.append({'date': d.strftime('%Y-%m-%d'), 'value': round(val, 2)})
            return {'curve': points, 'source': 'demo'}

        from marketmind.core.price_fetcher import get_price_fetcher
        pf = get_price_fetcher()
        total_invested = sum(h.get('average_price', 0) * h.get('quantity', 0) for h in holdings)

        # Current value
        total_current = sum(h.get('last_price', 0) * h.get('quantity', 0) for h in holdings)
        pnl = total_current - total_invested
        pnl_pct = (pnl / total_invested * 100) if total_invested else 0

        # Drawdown
        values = [e['value'] for e in [{'value': total_current}]]
        peak = total_invested
        max_dd = max(0, (peak - total_current) / peak * 100) if peak else 0

        return {
            'total_invested': total_invested,
            'total_current': total_current,
            'pnl': pnl,
            'pnl_pct': round(pnl_pct, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'curve': [{'date': datetime.now().strftime('%Y-%m-%d'), 'value': total_current}],
            'source': 'kite',
        }

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# ORDERS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/orders")
async def get_orders():
    orders = await _run(controller.get_orders)
    trades = await _run(controller.get_trades)
    return JSONResponse(_sanitize({'orders': orders, 'trades': trades}))


class OrderRequest(BaseModel):
    symbol: str
    exchange: str = 'NSE'
    transaction_type: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET or LIMIT
    product: str = 'CNC'
    price: float = 0
    trigger_price: float = 0


@app.post("/api/orders")
async def place_order(req: OrderRequest):
    if not controller.kite_is_authenticated:
        raise HTTPException(status_code=401, detail="Kite not authenticated")
    order_id = await _run(
        controller.place_order,
        req.symbol.upper(), req.exchange, req.transaction_type.upper(),
        req.quantity, req.order_type.upper(), req.product.upper(),
        req.price, req.trigger_price,
    )
    if not order_id:
        raise HTTPException(status_code=400, detail="Order placement failed")
    return JSONResponse({'order_id': order_id, 'status': 'placed'})


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str):
    result = await _run(controller.cancel_order, order_id)
    return JSONResponse({'cancelled': result is not None})


# ═════════════════════════════════════════════════════════════════════════════
# RL SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/rl/signals")
async def rl_signals():
    """
    Trained-model signals when available; otherwise fall back to the working
    multiframe swing-signal path so the dashboard always shows something useful.
    """
    signals = await _run(controller.get_rl_signals)
    if signals:
        return JSONResponse(_sanitize(signals))
    # Fallback: reuse multiframe swing scanner
    try:
        mf = await rl_multiframe()  # returns JSONResponse
        import json as _json
        body = _json.loads(mf.body)
        swing = body.get('swing', [])
        return JSONResponse(_sanitize(swing))
    except Exception:
        return JSONResponse([])


@app.get("/api/rl/signals/multiframe")
async def rl_multiframe():
    """Multi-timeframe RL signals: intraday, swing, positional."""
    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()

    # Large + midcap + smallcap representative set for RL scanning
    SYMBOLS = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN',
        'BHARTIARTL', 'TATAMOTORS', 'AXISBANK', 'WIPRO',
        'PERSISTENT', 'MPHASIS', 'COFORGE', 'LTIM', 'KPITTECH',
        'CHOLAFIN', 'POLYCAB', 'ASTRAL', 'CDSL', 'DEEPAKNTR',
        'ZOMATO', 'NYKAA', 'RVNL', 'IRFC', 'ADANIGREEN',
    ]

    def fetch():
        from marketmind.core.rl_trainer import compute_features, _rsi, _ema

        intraday   = []
        swing      = []
        positional = []

        # Step 1: bulk Kite OHLC for current prices (single API call)
        kite_prices = {}
        kite = controller.kite
        if kite and kite.is_connected:
            try:
                ohlc_data = kite.get_ohlc([f"NSE:{s}" for s in SYMBOLS])
                for sym in SYMBOLS:
                    key = f"NSE:{sym}"
                    if key in ohlc_data:
                        d = ohlc_data[key]
                        ohlc = d.get('ohlc', {})
                        ltp  = d.get('last_price', 0)
                        prev = ohlc.get('close', ltp)
                        kite_prices[sym] = {'current_price': ltp,
                                            'change_pct': (ltp - prev) / prev if prev else 0}
            except Exception as e:
                logger.debug(f"Kite OHLC for RL signals: {e}")

        # Step 2: per symbol — use Kite historical for indicators (cached), fallback NSE
        kc = get_kite_candles()
        for sym in SYMBOLS:
            try:
                # Get historical candles (cached — usually instant)
                df = kc.get_candles_df(sym, interval='day', days=200)
                if df is None or len(df) < 50:
                    continue
                feat = compute_features(df)
                feat = feat.iloc[50:].reset_index(drop=True)
                if len(feat) < 2:
                    continue

                last = feat.iloc[-1]
                prev = feat.iloc[-2]
                cp   = kite_prices.get(sym, {}).get('current_price') or float(df['close'].iloc[-1])
                if cp <= 0:
                    continue

                rsi       = float(last['rsi14'])
                macd_h    = float(last['macd_hist'])
                prev_macd = float(prev['macd_hist'])
                m5        = float(last['mom5'])
                m20       = float(last['mom20'])
                vol_ratio = float(last['vol_ratio'])
                c_vs_ma50 = float(last['c_vs_ma50'])
                c_vs_ma200= float(last['c_vs_ma200'])
                ma20_val  = cp / (1 + float(last['c_vs_ma20'])) if last['c_vs_ma20'] != -1 else cp
                ma50_val  = cp / (1 + c_vs_ma50) if c_vs_ma50 != -1 else cp
                above_ma50 = c_vs_ma50 > 0
                above_ma200= c_vs_ma200 > 0

                macd_cross_up = prev_macd <= 0 and macd_h > 0

                # ── Intraday: RSI + MACD crossover + volume
                if rsi < 42 and macd_cross_up and vol_ratio > 1.2:
                    intraday.append({'symbol': sym, 'action': 'BUY',
                        'confidence': round(min(0.88, 0.52 + vol_ratio * 0.08), 2),
                        'entry': cp, 'target': round(cp * 1.015, 2),
                        'sl': round(cp * 0.992, 2), 'horizon': 'Intraday',
                        'reason': f"RSI {rsi:.0f} oversold + MACD cross + vol×{vol_ratio:.1f}"})
                elif rsi > 70 and macd_h < 0 and vol_ratio > 1.2:
                    intraday.append({'symbol': sym, 'action': 'SELL',
                        'confidence': round(min(0.85, 0.52 + vol_ratio * 0.08), 2),
                        'entry': cp, 'target': round(cp * 0.985, 2),
                        'sl': round(cp * 1.008, 2), 'horizon': 'Intraday',
                        'reason': f"RSI {rsi:.0f} overbought + bearish MACD + vol×{vol_ratio:.1f}"})

                # ── Swing: momentum + MA50
                if m20 > 0.03 and above_ma50 and macd_h > 0:
                    swing.append({'symbol': sym, 'action': 'BUY',
                        'confidence': round(min(0.85, 0.48 + min(m20, 0.15) * 2.5), 2),
                        'entry': cp, 'target': round(cp * 1.08, 2),
                        'sl': round(ma20_val * 0.99, 2), 'horizon': '2–4 weeks',
                        'reason': f"+{m20*100:.1f}% 20d momentum, above MA50"})
                elif m20 < -0.03 and not above_ma50 and macd_h < 0:
                    swing.append({'symbol': sym, 'action': 'SELL',
                        'confidence': round(min(0.82, 0.48 + min(abs(m20), 0.15) * 2.5), 2),
                        'entry': cp, 'target': round(cp * 0.93, 2),
                        'sl': round(ma50_val * 1.01, 2), 'horizon': '2–4 weeks',
                        'reason': f"{m20*100:.1f}% 20d momentum, below MA50"})

                # ── Positional: MA200 + trend strength
                if above_ma200 and above_ma50 and m20 > 0.05 and rsi < 65:
                    positional.append({'symbol': sym, 'action': 'BUY',
                        'confidence': round(min(0.90, 0.60 + min(m20, 0.2) * 1.5), 2),
                        'entry': cp, 'target': round(cp * 1.20, 2),
                        'sl': round(cp / (1 + c_vs_ma200) * 0.97, 2), 'horizon': '3–6 months',
                        'reason': f"Above MA200+MA50, +{m20*100:.1f}% mom, RSI {rsi:.0f}"})
                elif not above_ma200 and m20 < -0.05 and rsi > 50:
                    positional.append({'symbol': sym, 'action': 'SELL',
                        'confidence': round(min(0.82, 0.55 + min(abs(m20), 0.2) * 1.5), 2),
                        'entry': cp, 'target': round(cp * 0.85, 2),
                        'sl': round(cp * 1.05, 2), 'horizon': '3–6 months',
                        'reason': f"Below MA200, {m20*100:.1f}% mom, RSI {rsi:.0f}"})

            except Exception as e:
                logger.debug(f"RL signal error for {sym}: {e}")

        # Sort by confidence
        for lst in [intraday, swing, positional]:
            lst.sort(key=lambda x: x['confidence'], reverse=True)

        # If still empty (no Kite + no NSE), use demo signals
        if not intraday and not swing and not positional:
            return _rl_demo_signals()

        return {
            'intraday':   intraday[:5],
            'swing':      swing[:5],
            'positional': positional[:5],
        }

    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


def _rl_demo_signals() -> Dict:
    """Demo signals when no data source is available."""
    return {
        'intraday': [
            {'symbol': 'RELIANCE', 'action': 'BUY', 'confidence': 0.72,
             'entry': 2845.5, 'target': 2888.0, 'sl': 2820.0, 'horizon': 'Intraday',
             'reason': 'RSI 38 oversold + MACD cross + vol×1.8'},
            {'symbol': 'TCS', 'action': 'SELL', 'confidence': 0.65,
             'entry': 3920.0, 'target': 3872.0, 'sl': 3950.0, 'horizon': 'Intraday',
             'reason': 'RSI 73 overbought + bearish MACD'},
        ],
        'swing': [
            {'symbol': 'HDFCBANK', 'action': 'BUY', 'confidence': 0.78,
             'entry': 1642.0, 'target': 1773.0, 'sl': 1610.0, 'horizon': '2–4 weeks',
             'reason': '+5.2% 20d momentum, above MA50'},
            {'symbol': 'PERSISTENT', 'action': 'BUY', 'confidence': 0.74,
             'entry': 5400.0, 'target': 5832.0, 'sl': 5240.0, 'horizon': '2–4 weeks',
             'reason': '+8.1% 20d momentum, above MA50'},
        ],
        'positional': [
            {'symbol': 'ICICIBANK', 'action': 'BUY', 'confidence': 0.82,
             'entry': 1095.0, 'target': 1314.0, 'sl': 1002.0, 'horizon': '3–6 months',
             'reason': 'Above MA200+MA50, +6.8% mom, RSI 58'},
        ],
        '_demo': True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str = 'swing_ma_cross'
    days: int = 365
    initial_capital: float = 100000
    stop_loss_pct: float = 3.0
    target_pct: float = 6.0


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    bt = get_backtester()
    result = await _run(
        bt.run,
        req.symbol.upper(), req.strategy, req.days,
        req.initial_capital, req.stop_loss_pct, req.target_pct,
    )
    return JSONResponse(_sanitize(result))


class WalkForwardRequest(BaseModel):
    symbol: str
    strategy: str = 'adx_trend_follow'
    days: int = 750
    train_window: int = 252
    test_window: int = 63
    initial_capital: float = 100000
    stop_loss_pct: float = 2.5
    target_pct: float = 7.0
    bootstrap_n: int = 1000


@app.post("/api/backtest/walkforward")
async def run_walkforward(req: WalkForwardRequest):
    """Anchored walk-forward backtest with realistic Indian costs and bootstrap Sharpe CDF."""
    wf = get_walkforward()
    result = await _run(
        wf.run,
        req.symbol.upper(), req.strategy, req.days,
        req.train_window, req.test_window,
        req.initial_capital, req.stop_loss_pct, req.target_pct,
        req.bootstrap_n,
    )
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT DEBATE (W1.1)
# ═════════════════════════════════════════════════════════════════════════════

class DebateRequest(BaseModel):
    symbol: str
    horizon: str = "swing"   # intraday | swing | long_term


@app.post("/api/debate")
async def stock_debate(req: DebateRequest):
    """Ten specialist agents argue about a stock at the requested horizon
    (intraday / swing / long_term); moderator synthesises BUY/SELL/HOLD with
    a shared evidence pack (candles, technicals, fundamentals, macro, regime,
    FII/DII, news, sector, options, risk)."""
    engine = get_debate_engine()
    result = await _run(engine.debate, req.symbol.upper(), req.horizon)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# ALERTS  (stored in SQLite via Database)
# ═════════════════════════════════════════════════════════════════════════════

import sqlite3

def _alerts_db_path():
    return os.path.join(os.path.dirname(__file__), 'marketmind', 'marketmind.db')


def _ensure_alerts_table():
    conn = sqlite3.connect(_alerts_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            condition TEXT NOT NULL,
            value REAL NOT NULL,
            message TEXT,
            active INTEGER DEFAULT 1,
            triggered INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_ensure_alerts_table()


@app.get("/api/alerts")
async def get_alerts():
    conn = sqlite3.connect(_alerts_db_path())
    rows = conn.execute(
        "SELECT id, symbol, condition, value, message, active, triggered, created_at "
        "FROM alerts ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return JSONResponse([
        {'id': r[0], 'symbol': r[1], 'condition': r[2], 'value': r[3],
         'message': r[4], 'active': bool(r[5]), 'triggered': bool(r[6]), 'created_at': r[7]}
        for r in rows
    ])


class AlertRequest(BaseModel):
    symbol: str
    condition: str  # 'above' or 'below'
    value: float
    message: str = ''


@app.post("/api/alerts")
async def create_alert(req: AlertRequest):
    conn = sqlite3.connect(_alerts_db_path())
    cur = conn.execute(
        "INSERT INTO alerts (symbol, condition, value, message) VALUES (?,?,?,?)",
        (req.symbol.upper(), req.condition, req.value, req.message)
    )
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return JSONResponse({'id': aid, 'status': 'created'})


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    conn = sqlite3.connect(_alerts_db_path())
    conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return JSONResponse({'deleted': True})


# ── Alert checker (runs in background, pushes WS notifications) ──────────────
async def _check_alerts():
    """Check price alerts every 60s and push browser notifications via WS."""
    from marketmind.core.price_fetcher import get_price_fetcher
    pf = get_price_fetcher()
    while True:
        try:
            conn = sqlite3.connect(_alerts_db_path())
            rows = conn.execute(
                "SELECT id, symbol, condition, value, message FROM alerts WHERE active=1 AND triggered=0"
            ).fetchall()
            conn.close()

            for row in rows:
                aid, sym, cond, val, msg = row
                try:
                    price_data = pf.get_stock_price(sym)
                    if not price_data:
                        continue
                    cp = price_data.get('current_price', 0)
                    triggered = (cond == 'above' and cp >= val) or (cond == 'below' and cp <= val)
                    if triggered:
                        conn = sqlite3.connect(_alerts_db_path())
                        conn.execute("UPDATE alerts SET triggered=1 WHERE id=?", (aid,))
                        conn.commit()
                        conn.close()
                        await manager.broadcast({
                            'type': 'alert',
                            'symbol': sym,
                            'condition': cond,
                            'value': val,
                            'current_price': cp,
                            'message': msg or f"{sym} hit ₹{val}",
                        })
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Alert check error: {e}")
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_alert_checker():
    asyncio.create_task(_check_alerts())


# ═════════════════════════════════════════════════════════════════════════════
# KITE AUTH
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/kite/status")
async def kite_status():
    return JSONResponse({
        'authenticated': controller.kite_is_authenticated,
        'configured': controller.kite_is_configured,
    })


@app.get("/api/kite/login-url")
async def kite_login_url():
    url = await _run(controller.get_kite_login_url)
    return JSONResponse({'url': url})


class SessionRequest(BaseModel):
    request_token: str


@app.post("/api/kite/session")
async def kite_session(req: SessionRequest):
    ok = await _run(controller.kite_generate_session, req.request_token)
    return JSONResponse({'success': ok})


@app.post("/api/kite/logout")
async def kite_logout():
    await _run(controller.kite_invalidate_session)
    return JSONResponse({'logged_out': True})


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'local.json')


@app.get("/api/config")
async def get_config():
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # Mask secrets
        if 'kite' in cfg:
            cfg['kite']['api_secret'] = '***'
        if 'anthropic' in cfg:
            cfg['anthropic']['api_key'] = '***'
        return JSONResponse(cfg)
    except Exception:
        return JSONResponse({})


class ConfigUpdate(BaseModel):
    watchlist: Optional[List[str]] = None
    app: Optional[Dict[str, Any]] = None


@app.put("/api/config")
async def update_config(req: ConfigUpdate):
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        if req.watchlist is not None:
            cfg['watchlist'] = [s.upper() for s in req.watchlist]
        if req.app is not None:
            cfg.setdefault('app', {}).update(req.app)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
        controller.kite_config.reload()
        return JSONResponse({'saved': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# FII / DII FLOWS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/fii-dii")
async def fii_dii(days: int = 20):
    fetcher = get_fii_dii_fetcher()
    def fetch():
        return {
            'data': fetcher.get_fii_dii_data(days),
            'summary': fetcher.get_summary(5),
        }
    result = await _run(fetch)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# BULK / BLOCK DEALS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/bulk-deals")
async def bulk_deals(days: int = 7):
    fetcher = get_bulk_deals_fetcher()
    result = await _run(fetcher.get_combined, days)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# MACRO DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/macro")
async def macro_dashboard():
    fetcher = get_macro_fetcher()
    result = await _run(fetcher.get_all)
    return JSONResponse(_sanitize(result))


@app.get("/api/regime")
async def market_regime(refresh: bool = False):
    """Current market regime: Trending Bull / Range / Volatile / Crash / Recovery."""
    clf = get_regime_classifier()
    result = await _run(clf.classify, refresh)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# ALT-DATA (W2.3) — Reddit, ValuePickr, SIAM, GST, IIP/CPI, Google Trends
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/altdata")
async def altdata_dashboard():
    """India alt-data signals from 6 sources, persisted to alt_signals (TTL 7d)."""
    agg = get_altdata_aggregator(mongo_col=controller._mongo_col("alt_signals"))
    result = await _run(agg.get_all)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# FORECAST (W3.1) — Ensemble of GARCH + Holt-Winters trend + PatchTST
# ═════════════════════════════════════════════════════════════════════════════

_FORECAST_VALID_MODELS = ("ensemble",)
_FORECAST_HORIZON_MAX = 10
_FORECAST_TRAIN_DAYS = 365


@app.get("/api/forecast/{sym}")
async def forecast(sym: str, horizon: int = 5, model: str = "ensemble"):
    """Forecast price + 80/95 PI bands at horizon trading days.

    Cached per (symbol, horizon, model, interval=day) — TTL 24h.
    Result includes ``calibration`` if the operator has populated it offline
    via ``evaluator.py``; absent otherwise.
    """
    sym = sym.upper().strip()
    if not (1 <= horizon <= _FORECAST_HORIZON_MAX):
        return JSONResponse(
            {"error": f"horizon must be in [1, {_FORECAST_HORIZON_MAX}]"},
            status_code=400,
        )
    if model not in _FORECAST_VALID_MODELS:
        return JSONResponse(
            {"error": f"model must be one of {_FORECAST_VALID_MODELS}"},
            status_code=400,
        )

    cache = get_forecast_cache(mongo_col=controller._mongo_col("forecast_cache"))

    def _compute() -> Dict[str, Any]:
        cached = cache.get(sym, horizon, model, interval="day")
        if cached is not None:
            return cached.to_dict() | {"cached": True}

        df = controller.price_fetcher.get_historical_data(sym, days=_FORECAST_TRAIN_DAYS)
        if df is None or len(df) < 60:
            return {
                "error": f"insufficient history for {sym} (got {0 if df is None else len(df)} rows, need ≥60)",
                "symbol": sym,
            }
        df = df.copy()
        df.attrs["symbol"] = sym
        ensemble = EnsembleForecaster()
        ensemble.fit(df)
        result = ensemble.predict(horizon)
        cache.set(result, interval="day")
        return result.to_dict() | {"cached": False}

    body = await _run(_compute)
    if "error" in body:
        return JSONResponse(body, status_code=422)
    return JSONResponse(_sanitize(body))


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATED SIGNAL (W3.2) — Conformal forecast + meta-stacker BUY/SELL/HOLD
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/signal/{sym}/calibrated")
async def calibrated_signal(sym: str, horizon: int = 5):
    """Calibrated BUY/SELL/HOLD probability + conformal-bounded return CI.

    Wraps the EnsembleForecaster in split-conformal to produce coverage-
    guaranteed bands, then routes (forecast_return, forecast_vol, rl_signal,
    regime, sentiment) through the meta-stacker for a 3-class decision.
    """
    sym = sym.upper().strip()
    if not (1 <= horizon <= _FORECAST_HORIZON_MAX):
        return JSONResponse(
            {"error": f"horizon must be in [1, {_FORECAST_HORIZON_MAX}]"},
            status_code=400,
        )

    def _compute() -> Dict[str, Any]:
        df = controller.price_fetcher.get_historical_data(sym, days=_FORECAST_TRAIN_DAYS)
        if df is None or len(df) < 80:
            return {
                "error": f"insufficient history for {sym} (need ≥80 rows for conformal calibration)",
                "symbol": sym,
            }
        df = df.copy()
        df.attrs["symbol"] = sym

        # Conformal-wrapped ensemble
        ensemble = EnsembleForecaster()
        wrapped = SplitConformalWrapper(ensemble, horizon=horizon, calibration_frac=0.2)
        wrapped.fit(df)
        fr = wrapped.predict(horizon)

        last_close = float(df["close"].iloc[-1])
        forecast_return = (fr.point / last_close - 1.0) if last_close else 0.0
        # PI95 half-width as a vol proxy, normalised
        pi95_halfwidth = (fr.upper_95 - fr.lower_95) / 2.0
        forecast_vol = (pi95_halfwidth / last_close) if last_close else 0.0

        # Pull RL signal (best-effort)
        rl_score = 0.0
        try:
            rl = controller.get_rl_signal_for_stock(sym) or {}
            # Different RL outputs use different keys; coerce to [-1, 1].
            raw = rl.get("score") or rl.get("signal_score") or 0.0
            action = (rl.get("action") or "").upper()
            if action == "BUY":
                rl_score = max(rl_score, float(raw) if raw else 0.5)
            elif action == "SELL":
                rl_score = min(rl_score, -abs(float(raw)) if raw else -0.5)
            else:
                rl_score = float(raw) if isinstance(raw, (int, float)) else 0.0
        except Exception:
            rl_score = 0.0

        # Pull regime state (best-effort)
        regime_state = ""
        try:
            from marketmind.core.regime_classifier import get_regime_classifier
            regime = get_regime_classifier().classify() or {}
            regime_state = (regime.get("state") or "").lower().replace(" ", "_")
        except Exception:
            pass

        # Pull sector sentiment as proxy (best-effort)
        sentiment_tilt = 0.0
        try:
            sentiments = controller.sector_classifier.get_all_sector_sentiments() or {}
            # Average across sectors as a coarse market-wide tilt
            vals = [float(s.get("score") or 0) for s in sentiments.values()
                    if isinstance(s, dict)]
            if vals:
                sentiment_tilt = float(sum(vals) / len(vals))
        except Exception:
            sentiment_tilt = 0.0

        features = {
            "forecast_return": forecast_return,
            "forecast_vol": forecast_vol,
            "rl_signal_score": max(-1.0, min(1.0, rl_score)),
            "regime_state": regime_state,
            "sentiment_tilt": max(-1.0, min(1.0, sentiment_tilt)),
        }
        probs = get_meta_stacker().predict_proba(features)

        return {
            "symbol": sym,
            "horizon_days": horizon,
            **probs,
            "expected_return": round(forecast_return, 6),
            "return_95ci": [
                round(fr.lower_95 / last_close - 1.0, 6) if last_close else 0.0,
                round(fr.upper_95 / last_close - 1.0, 6) if last_close else 0.0,
            ],
            "forecast": fr.to_dict(),
            "features": features,
        }

    body = await _run(_compute)
    if "error" in body:
        return JSONResponse(body, status_code=422)
    return JSONResponse(_sanitize(body))


# ═════════════════════════════════════════════════════════════════════════════
# RISK ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/risk/stock/{symbol}")
async def stock_risk(symbol: str, holding_value: float = 100000):
    engine = get_risk_engine()
    result = await _run(engine.stock_var, symbol.upper(), 0.95, holding_value)
    return JSONResponse(_sanitize(result))


class PortfolioRiskRequest(BaseModel):
    holdings: List[Dict]  # [{'symbol': str, 'value': float, 'sector': str}]


@app.post("/api/risk/portfolio")
async def portfolio_risk(req: PortfolioRiskRequest):
    engine = get_risk_engine()
    def fetch():
        var_result = engine.portfolio_var(req.holdings)
        stress_result = engine.stress_test(req.holdings)
        conc_result = engine.concentration_analysis(req.holdings)
        return {
            'var': var_result,
            'stress': stress_result,
            'concentration': conc_result,
        }
    result = await _run(fetch)
    return JSONResponse(_sanitize(result))


@app.post("/api/risk/stress-test")
async def stress_test(req: PortfolioRiskRequest):
    engine = get_risk_engine()
    result = await _run(engine.stress_test, req.holdings)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════════

class OptimizeRequest(BaseModel):
    symbols: List[str]
    objective: str = 'max_sharpe'  # max_sharpe | min_variance | risk_parity | equal_weight
    days: int = 252


@app.post("/api/optimize")
async def optimize_portfolio(req: OptimizeRequest):
    optimizer = get_optimizer()
    def fetch():
        result = optimizer.optimize(
            [s.upper() for s in req.symbols], req.objective, req.days
        )
        compare = optimizer.compare_strategies(
            [s.upper() for s in req.symbols], req.days
        )
        result['strategy_comparison'] = compare
        return result
    data = await _run(fetch)
    return JSONResponse(_sanitize(data))


@app.post("/api/optimize/frontier")
async def efficient_frontier(req: OptimizeRequest):
    optimizer = get_optimizer()
    result = await _run(
        optimizer.efficient_frontier,
        [s.upper() for s in req.symbols], 20, req.days
    )
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# AI RESEARCH REPORTS
# ═════════════════════════════════════════════════════════════════════════════

# ─── Grounded RAG over filings (W2.1) ───
class GroundedRequest(BaseModel):
    question: str
    k: int = 6
    category: Optional[str] = None


@app.post("/api/research/{symbol}/ingest")
async def research_ingest(symbol: str, days: int = 365):
    """Pull NSE corporate-announcements for the symbol and embed into vector DB."""
    ing = get_filings_ingester()
    result = await _run(ing.ingest_symbol, symbol.upper(), days)
    return JSONResponse(_sanitize(result))


@app.post("/api/research/{symbol}/grounded")
async def research_grounded(symbol: str, req: GroundedRequest):
    """Citation-grounded answer over the filings vector index."""
    eng = get_grounded_researcher()
    result = await _run(eng.answer, symbol.upper(), req.question, k=req.k, category=req.category)
    return JSONResponse(_sanitize(result))


@app.get("/api/research/index/stats")
async def research_index_stats():
    return JSONResponse(_sanitize(get_filings_ingester().stats()))


# ═════════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN FEED (W2.2)
# ═════════════════════════════════════════════════════════════════════════════

class WatchToggleRequest(BaseModel):
    symbol: str
    add: bool = True


@app.get("/api/events")
async def events_feed(
    symbol: Optional[str] = None,
    min_severity: int = 0,
    category: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
):
    """Recent corp events ordered newest first. Filter by symbol / severity /
    category / since=ISO timestamp."""
    poller = get_event_poller()
    rows = await _run(poller.query, symbol=symbol, min_severity=min_severity,
                      category=category, since_iso=since, limit=limit)
    return JSONResponse(_sanitize(rows))


@app.get("/api/events/stats")
async def events_stats():
    return JSONResponse(_sanitize(get_event_poller().stats()))


@app.post("/api/events/poll")
async def events_force_poll():
    """Trigger an immediate poll cycle (admin/debug)."""
    poller = get_event_poller()
    tally = await poller.poll_once(broadcast=manager.broadcast, executor=executor)
    return JSONResponse(_sanitize(tally))


@app.post("/api/events/watch")
async def events_watch(req: WatchToggleRequest):
    poller = get_event_poller()
    if req.add:
        poller.add(req.symbol)
    else:
        poller.remove(req.symbol)
    return JSONResponse(_sanitize({"symbols": poller.symbols}))


@app.get("/api/research/{symbol}")
async def research_report(symbol: str):
    def fetch():
        stock_data = controller.get_stock_data(symbol.upper()) or {}
        sector_data = controller.get_sector_data()
        rl_signal = controller.get_rl_signal_for_stock(symbol.upper())
        fii_summary = get_fii_dii_fetcher().get_summary(5)
        return generate_research_report(
            symbol.upper(), stock_data, sector_data, rl_signal, fii_summary
        )
    result = await _run(fetch)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# AI CHAT ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str = 'default'
    reset: bool = False


@app.post("/api/chat")
async def chat(req: ChatRequest):
    assistant = get_assistant(req.session_id)
    if req.reset:
        assistant.reset()
        return JSONResponse({'response': 'Conversation reset.', 'session_id': req.session_id})

    # Gather quick market context
    def fetch():
        try:
            indices = _get_indices_sync()
            nifty = next((i['current_price'] for i in indices if i['symbol'] == 'NIFTY500'), None)
            vix = next((i['current_price'] for i in indices if i['symbol'] == 'INDIA VIX'), None)
            fii = get_fii_dii_fetcher().get_summary(3)
            context = {
                'nifty500': nifty,
                'vix': vix,
                'fii_signal': fii.get('fii_signal'),
            }
            return assistant.chat(req.message, context)
        except Exception as e:
            return assistant.chat(req.message)

    response = await _run(fetch)
    return JSONResponse({'response': response, 'session_id': req.session_id})


# ═════════════════════════════════════════════════════════════════════════════
# CANDLE CHARTS  (Kite historical → MongoDB/SQLite cache)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/stocks/{symbol}/candles")
async def get_candles(
    symbol: str,
    interval: str = 'day',
    days: int = 365,
    force_refresh: bool = False,
):
    """
    Return OHLCV candle data for charting.

    interval: 1min | 5min | 10min | 15min | 30min | 60min | 1hour | day | week
    days:     calendar days of history (intraday max ~60 for minute bars)
    """
    kc = get_kite_candles()

    def fetch():
        return kc.get_candles(symbol.upper(), interval=interval, days=days,
                              force_refresh=force_refresh)

    data = await _run(fetch)
    if not data:
        return JSONResponse({'error': 'No candle data available. Connect Kite or try daily interval.'})
    return JSONResponse(_sanitize(data))


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO POSITIONS  (Kite positions + holdings + AI insights)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/portfolio/positions")
async def get_positions():
    """
    Full portfolio view: holdings (long-term) + day positions (intraday/F&O).
    Includes AI insights via Claude when Kite is authenticated.
    """
    def fetch():
        kite = controller.kite
        if not kite.is_connected:
            return {
                'authenticated': False,
                'message': 'Connect Kite in Settings to see positions.',
                'holdings': [],
                'positions_day': [],
                'positions_net': [],
                'total_pnl': 0,
                'insights': None,
            }

        holdings = kite.get_holdings()
        positions = kite.get_positions()
        day_pos = positions.get('day', [])
        net_pos = positions.get('net', [])

        # Summary metrics
        holdings_value = sum(h.get('last_price', 0) * h.get('quantity', 0) for h in holdings)
        holdings_invested = sum(h.get('average_price', 0) * h.get('quantity', 0) for h in holdings)
        holdings_pnl = holdings_value - holdings_invested

        day_pnl = sum(p.get('pnl', 0) for p in day_pos)
        total_pnl = holdings_pnl + day_pnl

        # AI insights
        insights = None
        try:
            from marketmind.core.claude_research import get_assistant
            assistant = get_assistant('portfolio_insights')
            top_winners = sorted(holdings, key=lambda h: h.get('pnl', 0), reverse=True)[:3]
            top_losers = sorted(holdings, key=lambda h: h.get('pnl', 0))[:3]
            msg = (
                f"I have {len(holdings)} holdings. "
                f"Total portfolio P&L: ₹{total_pnl:,.0f}. "
                f"Top winners: {', '.join(h['tradingsymbol'] + ' ₹' + str(round(h.get('pnl',0),0)) for h in top_winners if h.get('pnl',0)>0)}. "
                f"Top losers: {', '.join(h['tradingsymbol'] + ' ₹' + str(round(h.get('pnl',0),0)) for h in top_losers if h.get('pnl',0)<0)}. "
                f"Day P&L: ₹{day_pnl:,.0f}. "
                f"What should I do next? Give specific actionable advice on each position."
            )
            insights = assistant.chat(msg)
        except Exception as e:
            logger.debug(f"Portfolio insights error: {e}")

        return {
            'authenticated': True,
            'holdings': holdings,
            'positions_day': day_pos,
            'positions_net': net_pos,
            'holdings_value': round(holdings_value, 2),
            'holdings_invested': round(holdings_invested, 2),
            'holdings_pnl': round(holdings_pnl, 2),
            'day_pnl': round(day_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'insights': insights,
        }

    result = await _run(fetch)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# KITE MARKET QUOTES  (bulk quotes for watchlist)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/kite/quotes")
async def kite_quotes():
    """Fetch full market quotes from Kite for the configured watchlist."""
    def fetch():
        kite = controller.kite
        if not kite.is_connected:
            return {'error': 'Kite not authenticated'}
        watchlist = controller.kite_config.watchlist
        if not watchlist:
            return {'error': 'No watchlist configured'}
        symbols = [f"NSE:{s}" for s in watchlist]
        quotes = kite.get_quote(symbols)
        result = {}
        for sym in watchlist:
            key = f"NSE:{sym}"
            if key in quotes:
                q = quotes[key]
                ohlc = q.get('ohlc', {})
                ltp = q.get('last_price', 0)
                prev = ohlc.get('close', ltp)
                result[sym] = {
                    'ltp': ltp,
                    'open': ohlc.get('open', 0),
                    'high': ohlc.get('high', 0),
                    'low': ohlc.get('low', 0),
                    'prev_close': prev,
                    'change': round(ltp - prev, 2),
                    'change_pct': round((ltp - prev) / prev * 100, 2) if prev else 0,
                    'volume': q.get('volume', 0),
                    'oi': q.get('oi', 0),
                    'buy_quantity': q.get('buy_quantity', 0),
                    'sell_quantity': q.get('sell_quantity', 0),
                    'depth': q.get('depth', {}),
                }
        return result

    result = await _run(fetch)
    return JSONResponse(_sanitize(result))


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-TRADE FROM BACKTEST SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

class AutoTradeRequest(BaseModel):
    symbol: str
    action: str        # BUY or SELL
    price: float
    quantity: int
    product: str = 'CNC'
    tag: str = 'mm_backtest'


@app.post("/api/backtest/auto-trade")
async def backtest_auto_trade(req: AutoTradeRequest):
    """Place an order based on the latest backtest signal."""
    if not controller.kite_is_authenticated:
        raise HTTPException(status_code=401, detail="Kite not authenticated")

    def place():
        kite = controller.kite
        order_id = kite.place_order(
            tradingsymbol=req.symbol.upper(),
            exchange='NSE',
            transaction_type=req.action.upper(),
            quantity=req.quantity,
            order_type='MARKET',
            product=req.product.upper(),
            tag=req.tag,
        )
        return order_id

    order_id = await _run(place)
    if not order_id:
        raise HTTPException(status_code=400, detail="Order placement failed")
    return JSONResponse({'order_id': order_id, 'status': 'placed', 'action': req.action})


# ═════════════════════════════════════════════════════════════════════════
# RL TRAINER & ML ENSEMBLE — Train / Predict / List Models
# ═════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    symbol: str
    days: int = 730
    epochs: int = 200
    stop_loss_pct: float = 2.5
    take_profit_pct: float = 8.0
    trailing_sl_pct: float = 2.0


@app.post("/api/rl/train/{symbol}")
async def train_rl_model(symbol: str, req: TrainRequest = None):
    """Train PPO deep RL agent + ML ensemble for symbol."""
    sym  = symbol.upper()
    req  = req or TrainRequest(symbol=sym)
    days = req.days

    def _train():
        kc = get_kite_candles()
        df = kc.get_candles_df(sym, interval='day', days=days)
        if df is None or len(df) < 130:
            try:
                from marketmind.core.price_fetcher import get_price_fetcher
                hist = get_price_fetcher().get_historical_data(sym, days=days)
                if not hist.empty and len(hist) >= 130:
                    df = hist
                else:
                    return {'error': f'Need 130+ trading days for {sym}. Try 730 days.'}
            except Exception as e:
                return {'error': f'Data fetch failed: {e}'}

        # Use PPO deep RL (falls back to Q-learning if PyTorch unavailable)
        from marketmind.core.ppo_trainer import train_ppo_agent
        rl_result = train_ppo_agent(
            df, sym, epochs=req.epochs,
            initial_capital=100_000.0,
            stop_loss_pct=req.stop_loss_pct,
            take_profit_pct=req.take_profit_pct,
            trailing_sl_pct=req.trailing_sl_pct,
        )
        if rl_result.get('error'):
            return rl_result

        from marketmind.core.rl_trainer import train_ml_ensemble
        ml_result = train_ml_ensemble(df, sym)

        # Persist strategy config so the bot can be deployed with one click
        _save_strategy(sym, {
            'symbol': sym,
            'trained_at': datetime.utcnow().isoformat(),
            'stop_loss_pct': req.stop_loss_pct,
            'take_profit_pct': req.take_profit_pct,
            'trailing_sl_pct': req.trailing_sl_pct,
            'confidence_threshold': 0.6,
            'product': 'MIS',
            'max_order_value': 10000.0,
        })

        return {
            'symbol': sym, 'bars_used': len(df),
            'rl': rl_result, 'ml': ml_result,
        }

    result = await _run(_train)
    if result and result.get('error'):
        raise HTTPException(status_code=400, detail=result['error'])
    return JSONResponse(_sanitize(result))


@app.get("/api/rl/predict/{symbol}")
async def predict_rl_signal(symbol: str):
    """Get PPO/RL + ML ensemble prediction for symbol using saved model."""
    sym = symbol.upper()

    def _predict():
        kc = get_kite_candles()
        df = kc.get_candles_df(sym, interval='day', days=365)
        if df is None:
            try:
                from marketmind.core.price_fetcher import get_price_fetcher
                df = get_price_fetcher().get_historical_data(sym, days=365)
            except Exception:
                pass
        if df is None or len(df) < 60:
            return {'error': f'Insufficient data for {sym}'}

        from marketmind.core.rl_trainer import compute_features
        from marketmind.core.stock_env import OBS_COLS
        feat = compute_features(df)
        if feat.empty:
            return {'error': 'Feature computation failed'}
        last = feat.iloc[-1]

        # Try PPO model first
        from marketmind.core.ppo_trainer import predict_ppo
        ppo_sig = predict_ppo(sym, last, [c for c in OBS_COLS if c in feat.columns])

        # Also get Q-learning + ML ensemble signal
        from marketmind.core.rl_trainer import get_combined_signal
        ql_sig  = get_combined_signal(sym, df)

        # Merge: PPO takes priority if model exists, otherwise use RL
        if ppo_sig.get('source') == 'ppo_model':
            ql_sig['ppo_action']     = ppo_sig['action']
            ql_sig['ppo_confidence'] = ppo_sig['confidence']
            # Override action with PPO if confident
            if ppo_sig['confidence'] >= 0.55:
                ql_sig['action']     = ppo_sig['action']
                ql_sig['confidence'] = ppo_sig['confidence']
                ql_sig['method']     = 'PPO'
            else:
                ql_sig['method']     = 'RL+ML'
        return ql_sig

    result = await _run(_predict)
    return JSONResponse(result)


@app.get("/api/rl/confluence/{symbol}")
async def get_confluence(symbol: str):
    """Run 10-point confluence scoring on latest data for a symbol (no model needed)."""
    from marketmind.core.rl_trainer import score_confluence
    sym = symbol.upper()
    def _score():
        kc = get_kite_candles()
        df = kc.get_candles_df(sym, interval='day', days=200)
        if df is None:
            try:
                from marketmind.core.price_fetcher import get_price_fetcher
                df = get_price_fetcher().get_historical_data(sym, days=200)
            except Exception:
                pass
        if df is None or len(df) < 60:
            return {'error': 'Insufficient data', 'symbol': sym}
        result = score_confluence(df)
        result['symbol'] = sym
        result['current_price'] = float(df['close'].iloc[-1])
        return result
    data = await _run(_score)
    return JSONResponse(_sanitize(data))


@app.get("/api/rl/models")
async def list_rl_models():
    """List all trained RL/PPO/ML models with performance stats."""
    def _list():
        from marketmind.core.rl_trainer import list_saved_models
        from marketmind.core.ppo_trainer import list_ppo_models
        ql_models  = list_saved_models() or []
        ppo_models = list_ppo_models()   or []
        # Merge: prefer PPO entry if same symbol exists in both
        symbols_ppo = {m['symbol'] for m in ppo_models}
        merged = ppo_models + [m for m in ql_models if m.get('symbol') not in symbols_ppo]
        return merged
    models = await _run(_list)
    return JSONResponse(models or [])


@app.get("/api/rl/mistakes/{symbol}")
async def get_mistake_patterns(symbol: str):
    """Return top-5 market patterns that led to losses in the trained model's backtest."""
    sym = symbol.upper()
    def _mistakes():
        import glob as globmod
        from marketmind.core.ppo_trainer import _MODELS_DIR
        path = os.path.join(_MODELS_DIR, f'{sym}_ppo.pt')
        if not os.path.exists(path):
            return {'mistakes': [], 'note': 'No PPO model found. Train first.'}
        try:
            import torch
            from marketmind.core.ppo_trainer import ActorCriticNet, _simulate_deterministic
            from marketmind.core.rl_trainer import compute_features
            from marketmind.core.stock_env import analyze_mistakes, OBS_COLS
            kc = get_kite_candles()
            df = kc.get_candles_df(sym, interval='day', days=730)
            if df is None or len(df) < 60:
                return {'mistakes': [], 'note': 'Insufficient data'}
            feat = compute_features(df).iloc[50:].reset_index(drop=True)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            net  = ActorCriticNet(ckpt['obs_size']).to('cpu')
            net.load_state_dict(ckpt['net_state'])
            net.eval()
            sim = _simulate_deterministic(
                net, feat, 100_000,
                ckpt.get('stop_loss_pct', 2.5),
                ckpt.get('take_profit_pct', 8.0),
                ckpt.get('trailing_sl_pct', 2.0),
            )
            obs_cols = [c for c in OBS_COLS if c in feat.columns]
            mistakes = analyze_mistakes(sim.get('_trades_raw', []), obs_cols)
            return {'symbol': sym, 'mistakes': mistakes,
                    'total_trades': sim.get('total_trades', 0),
                    'win_rate': sim.get('win_rate', 0)}
        except Exception as e:
            logger.error(f"Mistakes analysis error: {e}")
            return {'mistakes': [], 'error': str(e)}
    result = await _run(_mistakes)
    return JSONResponse(result)


@app.delete("/api/rl/models/{symbol}")
async def delete_rl_model(symbol: str):
    """Delete trained models for a symbol."""
    import glob as globmod
    from marketmind.core.rl_trainer import _MODELS_DIR
    sym = symbol.upper()
    deleted = []
    for pattern in [f'{sym}_rl.json', f'{sym}_ml.pkl']:
        path = os.path.join(_MODELS_DIR, pattern)
        if os.path.exists(path):
            os.remove(path)
            deleted.append(pattern)
    return JSONResponse({'deleted': deleted, 'symbol': sym})


# ═════════════════════════════════════════════════════════════════════════
# RL BOT — Automated trading using trained model signals
# ═════════════════════════════════════════════════════════════════════════

_bot_state: Dict[str, Any] = {}  # symbol → {active, last_action, last_price}


_STRATEGY_DIR = os.path.join(os.path.dirname(__file__), 'marketmind', 'models')


def _save_strategy(sym: str, params: dict):
    """Persist strategy config for a symbol so it can be re-loaded at deploy time."""
    os.makedirs(_STRATEGY_DIR, exist_ok=True)
    path = os.path.join(_STRATEGY_DIR, f'{sym}_strategy.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def _load_strategy(sym: str) -> Optional[dict]:
    """Load saved strategy config for a symbol."""
    path = os.path.join(_STRATEGY_DIR, f'{sym}_strategy.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@app.get("/api/bot/strategy/{symbol}")
async def get_bot_strategy(symbol: str):
    """Return saved strategy/deploy config for a symbol."""
    sym = symbol.upper()
    s = _load_strategy(sym)
    if not s:
        raise HTTPException(status_code=404, detail=f"No saved strategy for {sym}")
    return JSONResponse(s)


class BotRequest(BaseModel):
    symbol: str
    product: str = 'MIS'
    confidence_threshold: float = 0.6
    stop_loss_pct: float = 2.5
    trailing_sl_pct: float = 2.0
    max_order_value: float = 10000.0  # ₹ — quantity computed dynamically


@app.post("/api/bot/start")
async def start_bot(req: BotRequest):
    """Deploy automated RL trading bot for a symbol."""
    sym = req.symbol.upper()
    _bot_state[sym] = {
        'active': True, 'symbol': sym,
        'product': req.product,
        'confidence_threshold': req.confidence_threshold,
        'stop_loss_pct': req.stop_loss_pct,
        'trailing_sl_pct': req.trailing_sl_pct,
        'max_order_value': req.max_order_value,
        'started_at': datetime.utcnow().isoformat(),
        'last_action': None, 'last_signal': None,
        'buy_price': 0.0, 'peak_price': 0.0,
        'last_qty': 0,
        'orders': [],
    }
    logger.info(f"Bot deployed for {sym} max=₹{req.max_order_value} SL={req.stop_loss_pct}% TSL={req.trailing_sl_pct}% product={req.product}")
    return JSONResponse({'status': 'started', 'symbol': sym})


@app.post("/api/bot/stop/{symbol}")
async def stop_bot(symbol: str):
    sym = symbol.upper()
    if sym in _bot_state:
        _bot_state[sym]['active'] = False
    return JSONResponse({'status': 'stopped', 'symbol': sym})


@app.get("/api/bot/status")
async def bot_status():
    return JSONResponse(list(_bot_state.values()))


@app.post("/api/bot/execute/{symbol}")
async def bot_execute(symbol: str):
    """
    Run one cycle of the bot: fetch latest data, get RL signal,
    place order if confidence >= threshold and action changed.
    """
    from marketmind.core.rl_trainer import get_combined_signal
    sym = symbol.upper()
    state = _bot_state.get(sym)
    if not state or not state.get('active'):
        raise HTTPException(status_code=400, detail=f"Bot not active for {sym}")

    def _cycle():
        from marketmind.core.rl_trainer import get_combined_signal as _cs
        kc = get_kite_candles()
        df = kc.get_candles_df(sym, interval='day', days=365)
        if df is None or len(df) < 60:
            return {'error': 'Insufficient data'}

        signal     = _cs(sym, df)
        state['last_signal'] = signal

        # Try to get live LTP from Kite for accurate pricing & SL tracking
        cur_price = signal.get('entry_price', 0)
        if controller.kite_is_authenticated:
            try:
                ltp_data = controller.kite.get_ltp([f'NSE:{sym}'])
                ltp = (ltp_data or {}).get(f'NSE:{sym}', {}).get('last_price', 0)
                if ltp and ltp > 0:
                    cur_price = ltp
                    signal['live_price'] = ltp
            except Exception:
                pass

        action     = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        threshold  = state.get('confidence_threshold', 0.6)
        sl_pct     = state.get('stop_loss_pct', 2.5)
        tsl_pct    = state.get('trailing_sl_pct', 2.0)

        # Server-side stop-loss and trailing SL tracking
        # SL/TSL fires even if Kite is not live — uses last known price
        if state.get('buy_price', 0) > 0 and cur_price > 0:
            buy_p  = state['buy_price']
            peak_p = state.get('peak_price', buy_p)
            state['peak_price'] = max(peak_p, cur_price)
            pct_from_entry = (cur_price - buy_p) / buy_p * 100
            trail_dd       = (state['peak_price'] - cur_price) / state['peak_price'] * 100

            if pct_from_entry <= -sl_pct:
                logger.info(f"Bot SL triggered for {sym}: {pct_from_entry:.2f}% from entry")
                action = 'SELL'
                signal['action'] = 'SELL'
                signal['exit_reason'] = f'Stop Loss ({sl_pct}%)'
                confidence = 1.0  # SL/TSL bypass confidence gate
            elif trail_dd >= tsl_pct:
                logger.info(f"Bot trailing SL triggered for {sym}: {trail_dd:.2f}% from peak")
                action = 'SELL'
                signal['action'] = 'SELL'
                signal['exit_reason'] = f'Trailing SL ({tsl_pct}%)'
                confidence = 1.0

        # Compute quantity from max_order_value / live price
        max_val  = state.get('max_order_value', 10000.0)
        quantity = max(1, int(max_val / cur_price)) if cur_price > 0 else 1
        state['last_qty'] = quantity

        order_result  = None
        action_taken  = 'HOLD'
        if confidence >= threshold and action in ('BUY', 'SELL'):
            # Only allow BUY when not already holding; SELL requires a position (or forced SL/TSL)
            already_holding = state.get('buy_price', 0) > 0
            if action == 'BUY' and already_holding:
                action_taken = 'HOLD'  # Already in position — skip
            elif action == 'SELL' and not already_holding and not signal.get('exit_reason'):
                action_taken = 'HOLD'  # Nothing to sell
            else:
                last_action = state.get('last_action')
                if action != last_action or signal.get('exit_reason'):  # SL/TSL overrides dedup
                    if controller.kite_is_authenticated:
                        try:
                            order_id = controller.kite.place_order(
                                tradingsymbol=sym, exchange='NSE',
                                transaction_type=action, quantity=quantity,
                                order_type='MARKET', product=state['product'].upper(),
                                tag='mm_rl_bot',
                            )
                            state['last_action'] = action
                            if action == 'BUY':
                                state['buy_price']  = cur_price
                                state['peak_price'] = cur_price
                            elif action == 'SELL':
                                state['buy_price']  = 0.0
                                state['peak_price'] = 0.0
                            order_result = {'order_id': order_id, 'action': action,
                                            'price': cur_price, 'quantity': quantity,
                                            'value': round(cur_price * quantity, 2)}
                            state['orders'].append({**order_result, 'timestamp': datetime.utcnow().isoformat()})
                            action_taken = action
                            logger.info(f"Bot order placed: {action} {sym} qty={quantity} @ ₹{cur_price:.2f} (₹{cur_price*quantity:.0f})")
                        except Exception as e:
                            order_result = {'error': str(e)}
                            logger.error(f"Bot order error for {sym}: {e}")
                    else:
                        # Kite not authenticated — simulate signal only
                        action_taken = action
                        if action == 'BUY':
                            state['buy_price']  = cur_price
                            state['peak_price'] = cur_price
                        elif action == 'SELL':
                            state['buy_price']  = 0.0
                            state['peak_price'] = 0.0
                        order_result = {'simulated': True, 'action': action, 'price': cur_price,
                                        'quantity': quantity,
                                        'value': round(cur_price * quantity, 2),
                                        'note': 'Kite not authenticated — signal only, no real order'}

        return {
            'symbol': sym, 'signal': signal,
            'action_taken': action_taken, 'order': order_result,
            'quantity': quantity, 'cur_price': cur_price,
            'bot_state': {k: v for k, v in state.items() if k not in ('orders',)},
        }

    result = await _run(_cycle)
    return JSONResponse(result)
