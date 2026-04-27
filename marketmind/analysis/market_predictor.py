"""
MarketMind AI - Market Predictor & Analysis Engine
Automatic sector/index prediction, fundamental analysis,
news-stock correlation, and investment timing recommendations.
"""

import threading
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# FUNDAMENTAL DATA
# ============================================================

class FundamentalAnalyzer:
    """
    Fetches and analyzes fundamental data for stocks.
    Uses NSE India API + Screener.in for fundamentals.
    """

    # Key metrics thresholds
    PE_CHEAP  = 15
    PE_FAIR   = 25
    PE_RICH   = 40
    PB_CHEAP  = 1.5
    ROE_GOOD  = 15   # percent
    DEBT_OK   = 1.0  # D/E ratio

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = 3600 * 4  # 4 hours

    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data for a stock symbol (NSE) via Screener.in"""
        now = datetime.now()
        if (symbol in self._cache
                and self._cache_expiry.get(symbol, now) > now):
            return self._cache[symbol]

        data = self._fetch_from_screener(symbol)
        if data:
            self._cache[symbol] = data
            self._cache_expiry[symbol] = now + timedelta(seconds=self._cache_ttl)
        return data

    def _fetch_from_screener(self, symbol: str) -> Dict:
        """Fetch fundamentals from Screener.in via price_fetcher."""
        try:
            from ..core.price_fetcher import get_price_fetcher
            pf = get_price_fetcher()
            # get_stock_price already enriches with Screener fundamentals
            price_data = pf.get_stock_price(symbol)
            if not price_data:
                return {}

            pe = price_data.get('pe_ratio')
            pb = price_data.get('pb_ratio')
            roe = price_data.get('roe', 0)
            market_cap = price_data.get('market_cap', 0)

            return {
                'symbol': symbol,
                'name': price_data.get('name', symbol),
                'sector': 'Unknown',
                'industry': 'Unknown',
                'pe_ratio': pe,
                'pb_ratio': pb,
                'roe': roe,
                'debt_equity': price_data.get('debt_equity', 0),
                'revenue_growth': price_data.get('revenue_growth', 0),
                'earnings_growth': price_data.get('profit_growth', 0),
                'profit_margin': 0,
                'dividend_yield': price_data.get('dividend_yield', 0),
                'beta': 1.0,
                'market_cap': market_cap,
                'market_cap_cr': round(market_cap / 1e7, 0) if market_cap else 0,
                'free_cashflow': 0,
                'current_ratio': 0,
                'quick_ratio': 0,
                'eps': price_data.get('eps', 0),
                'book_value': 0,
                'enterprise_value': 0,
                'ev_ebitda': 0,
            }
        except Exception as e:
            logger.error(f"Screener fundamental fetch error for {symbol}: {e}")
            return {}

    def score_stock(self, fundamentals: Dict) -> Dict:
        """
        Score a stock on fundamentals. Returns a score 0-100 and
        a recommendation with reasons.
        """
        score = 50  # neutral baseline
        reasons = []
        warnings = []

        pe = fundamentals.get('pe_ratio')
        if pe:
            if pe < self.PE_CHEAP:
                score += 15; reasons.append(f"P/E {pe:.1f} — undervalued")
            elif pe < self.PE_FAIR:
                score += 5;  reasons.append(f"P/E {pe:.1f} — fairly valued")
            elif pe < self.PE_RICH:
                score -= 5;  warnings.append(f"P/E {pe:.1f} — expensive")
            else:
                score -= 15; warnings.append(f"P/E {pe:.1f} — very expensive")

        roe = fundamentals.get('roe', 0)
        if roe:
            if roe > self.ROE_GOOD:
                score += 10; reasons.append(f"ROE {roe:.1f}% — strong profitability")
            elif roe > 8:
                score += 3
            else:
                score -= 5; warnings.append(f"ROE {roe:.1f}% — weak profitability")

        de = fundamentals.get('debt_equity', 0)
        if de:
            if de < 0.5:
                score += 8; reasons.append("Low debt")
            elif de < self.DEBT_OK:
                score += 3
            else:
                score -= 8; warnings.append(f"D/E {de:.2f} — high debt")

        eg = fundamentals.get('earnings_growth', 0)
        if eg:
            if eg > 20:
                score += 12; reasons.append(f"Earnings growth {eg:.1f}%")
            elif eg > 10:
                score += 6;  reasons.append(f"Earnings growth {eg:.1f}%")
            elif eg < 0:
                score -= 10; warnings.append(f"Earnings declining {eg:.1f}%")

        pm = fundamentals.get('profit_margin', 0)
        if pm:
            if pm > 20:
                score += 8;  reasons.append(f"Strong margins {pm:.1f}%")
            elif pm < 5:
                score -= 5;  warnings.append(f"Thin margins {pm:.1f}%")

        score = max(0, min(100, score))

        if score >= 70:
            verdict = "STRONG BUY"
        elif score >= 60:
            verdict = "BUY"
        elif score >= 45:
            verdict = "HOLD"
        elif score >= 35:
            verdict = "SELL"
        else:
            verdict = "STRONG SELL"

        return {
            'score': score,
            'verdict': verdict,
            'reasons': reasons,
            'warnings': warnings,
        }


# ============================================================
# SECTOR PREDICTOR
# ============================================================

class SectorPredictor:
    """
    Predicts sector/index direction and magnitude using:
    - News sentiment momentum
    - Price momentum (technical)
    - Correlation with global indices
    - Earnings season signals
    """

    SECTORS = {
        'IT':      ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
        'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
        'Auto':    ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
        'Pharma':  ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA'],
        'FMCG':    ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR', 'BRITANNIA'],
        'Metal':   ['TATASTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC'],
        'Energy':  ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL'],
        'Realty':  ['DLF', 'GODREJPROP', 'BRIGADE', 'PRESTIGE', 'OBEROIRLTY'],
        'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'ICICIPRULI', 'HDFC'],
    }

    def __init__(self, price_fetcher, sentiment_analyzer, sector_classifier):
        self.price_fetcher = price_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.sector_classifier = sector_classifier
        self._predictions: Dict[str, Dict] = {}

    def predict_all_sectors(self, news: List[Dict]) -> Dict[str, Dict]:
        """
        Predict direction/magnitude for all sectors.
        Returns dict: sector -> prediction
        """
        predictions = {}
        for sector, stocks in self.SECTORS.items():
            try:
                pred = self._predict_sector(sector, stocks, news)
                predictions[sector] = pred
            except Exception as e:
                logger.error(f"Prediction error for {sector}: {e}")
                predictions[sector] = self._default_prediction(sector)

        self._predictions = predictions
        return predictions

    def _predict_sector(self, sector: str, stocks: List[str], news: List[Dict]) -> Dict:
        """Predict a single sector"""
        # 1. News sentiment score
        sector_news = self.sector_classifier.get_sector_news(news, sector)
        sentiment = self.sentiment_analyzer.aggregate_sentiment(sector_news)
        sent_score = sentiment['score']  # -1 to +1

        # 2. Technical momentum for representative stocks
        momentum_scores = []
        rsi_scores = []
        for sym in stocks[:3]:  # use top 3 stocks
            try:
                indicators = self.price_fetcher.calculate_technical_indicators(sym)
                if indicators:
                    m5  = indicators.get('momentum_5', 0)
                    m20 = indicators.get('momentum_20', 0)
                    rsi = indicators.get('rsi', 50)
                    momentum_scores.append((m5 + m20) / 2)
                    rsi_scores.append(rsi)
            except Exception:
                pass

        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0
        avg_rsi = np.mean(rsi_scores) if rsi_scores else 50

        # 3. Composite score
        # Sentiment weight: 40%, Momentum: 40%, RSI mean-reversion: 20%
        rsi_signal = (50 - avg_rsi) / 50 * 0.3  # oversold RSI → positive signal
        composite = (sent_score * 0.40) + (avg_momentum * 0.40) + (rsi_signal * 0.20)

        # 4. Convert to direction and estimated % move
        direction = "UP" if composite > 0.02 else ("DOWN" if composite < -0.02 else "NEUTRAL")
        magnitude = abs(composite) * 100  # rough % estimate

        # 5. Confidence level
        confidence = min(95, max(20, int(abs(composite) * 200)))

        # 6. Time horizon suggestion
        if abs(composite) > 0.1:
            horizon = "1-3 days"
        elif abs(composite) > 0.05:
            horizon = "3-7 days"
        else:
            horizon = "1-2 weeks"

        # 7. Reasons
        reasons = []
        if sent_score > 0.1:
            reasons.append(f"Positive news sentiment ({sent_score:+.2f})")
        elif sent_score < -0.1:
            reasons.append(f"Negative news sentiment ({sent_score:+.2f})")
        if avg_momentum > 0.02:
            reasons.append(f"Positive price momentum ({avg_momentum * 100:+.1f}%)")
        elif avg_momentum < -0.02:
            reasons.append(f"Negative price momentum ({avg_momentum * 100:+.1f}%)")
        if avg_rsi < 35:
            reasons.append(f"Oversold RSI ({avg_rsi:.0f}) — potential bounce")
        elif avg_rsi > 70:
            reasons.append(f"Overbought RSI ({avg_rsi:.0f}) — potential pullback")
        if len(sector_news) > 3:
            reasons.append(f"{len(sector_news)} news articles found")

        return {
            'sector': sector,
            'direction': direction,
            'magnitude_pct': round(magnitude, 2),
            'confidence': confidence,
            'composite_score': composite,
            'sentiment_score': sent_score,
            'momentum': avg_momentum,
            'rsi': avg_rsi,
            'horizon': horizon,
            'reasons': reasons,
            'news_count': len(sector_news),
            'stocks': stocks[:5],
        }

    def _default_prediction(self, sector: str) -> Dict:
        return {
            'sector': sector,
            'direction': 'NEUTRAL',
            'magnitude_pct': 0,
            'confidence': 0,
            'composite_score': 0,
            'sentiment_score': 0,
            'momentum': 0,
            'rsi': 50,
            'horizon': 'Unknown',
            'reasons': ['Insufficient data'],
            'news_count': 0,
            'stocks': self.SECTORS.get(sector, []),
        }


# ============================================================
# INVESTMENT ADVISOR
# ============================================================

class InvestmentAdvisor:
    """
    Generates concrete investment recommendations:
    - Entry price zones
    - Exit (target) levels
    - Stop-loss levels
    - Position sizing
    - Risk/reward ratio
    - Timing windows
    """

    def __init__(self, price_fetcher, fundamental_analyzer: FundamentalAnalyzer):
        self.price_fetcher = price_fetcher
        self.fundamentals = fundamental_analyzer

    def analyze_stock(self, symbol: str, sentiment_score: float = 0,
                      sector_direction: str = 'NEUTRAL') -> Dict:
        """
        Complete investment analysis for a stock.
        Returns entry zone, targets, stop-loss, and recommendation.
        """
        try:
            indicators = self.price_fetcher.calculate_technical_indicators(symbol)
            if not indicators:
                return {}

            fundamentals = self.fundamentals.get_fundamentals(symbol)
            fund_score = self.fundamentals.score_stock(fundamentals) if fundamentals else {}

            price = indicators.get('current_price', 0)
            if not price:
                return {}

            atr = indicators.get('atr', price * 0.02)
            rsi = indicators.get('rsi', 50)
            ma_20 = indicators.get('ma_20', price)
            ma_50 = indicators.get('ma_50', price)
            bb_upper = indicators.get('bb_upper', price * 1.02)
            bb_lower = indicators.get('bb_lower', price * 0.98)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            above_ma20 = indicators.get('above_ma_20', False)
            above_ma50 = indicators.get('above_ma_50', False)

            # ── Entry zone ──────────────────────────────────────
            if rsi < 35 and above_ma50:
                # Oversold bounce setup — buy dip near support
                entry_low  = max(bb_lower, ma_50 * 0.99)
                entry_high = price * 1.005
                entry_reason = "Oversold RSI near MA50 support"
            elif above_ma20 and above_ma50 and macd > macd_signal:
                # Momentum setup — buy breakout
                entry_low  = ma_20 * 0.99
                entry_high = price * 1.01
                entry_reason = "Bullish momentum above MA20/MA50, MACD positive"
            else:
                # Neutral entry
                entry_low  = price * 0.995
                entry_high = price * 1.005
                entry_reason = "At current market price"

            # ── Stop-loss ───────────────────────────────────────
            sl_1 = price - 1.5 * atr      # ATR-based
            sl_2 = bb_lower * 0.99         # below lower band
            sl_3 = ma_50 * 0.98            # below MA50
            stop_loss = max(sl_1, sl_2, sl_3)  # tightest SL
            sl_pct = (price - stop_loss) / price * 100

            # ── Target levels ───────────────────────────────────
            t1 = price + 1.5 * atr      # conservative (1.5x ATR)
            t2 = price + 3.0 * atr      # moderate (3x ATR)
            t3 = bb_upper * 1.005        # aggressive (above upper band)

            rr_t1 = (t1 - price) / (price - stop_loss)

            # ── Timing signal ───────────────────────────────────
            timing_signals = []
            if rsi < 35:
                timing_signals.append("Oversold — potential reversal soon")
            if macd > macd_signal and macd > 0:
                timing_signals.append("MACD bullish crossover confirmed")
            if above_ma20 and above_ma50:
                timing_signals.append("Price above key moving averages — trend is up")
            if sector_direction == 'UP':
                timing_signals.append("Sector predicted to move UP — tailwind")

            # ── Avoid signal ────────────────────────────────────
            avoid_signals = []
            if rsi > 75:
                avoid_signals.append("Overbought RSI — wait for pullback")
            if not above_ma50:
                avoid_signals.append("Below MA50 — trend is down")
            if macd < macd_signal:
                avoid_signals.append("MACD bearish — momentum fading")
            if sector_direction == 'DOWN':
                avoid_signals.append("Sector predicted to move DOWN — headwind")
            if sentiment_score < -0.2:
                avoid_signals.append("Negative news sentiment")

            # ── Final recommendation ────────────────────────────
            buy_score = 0
            buy_score += 20 if rsi < 45 else (-10 if rsi > 70 else 0)
            buy_score += 20 if (macd > macd_signal) else -10
            buy_score += 15 if above_ma20 else -10
            buy_score += 10 if above_ma50 else -10
            buy_score += 15 if sentiment_score > 0.1 else (-10 if sentiment_score < -0.1 else 0)
            buy_score += 10 if sector_direction == 'UP' else (-10 if sector_direction == 'DOWN' else 0)
            buy_score += fund_score.get('score', 50) - 50  # fundamental contribution

            buy_score = max(0, min(100, buy_score + 50))

            if buy_score >= 70:
                action = "BUY NOW"
                action_color = 'green'
            elif buy_score >= 60:
                action = "BUY ON DIP"
                action_color = 'green'
            elif buy_score >= 45:
                action = "HOLD / WATCH"
                action_color = 'gray'
            elif buy_score >= 35:
                action = "AVOID / SELL"
                action_color = 'red'
            else:
                action = "STRONG SELL"
                action_color = 'red'

            return {
                'symbol': symbol,
                'current_price': price,
                'action': action,
                'action_score': buy_score,
                'action_color': action_color,
                'entry_low': round(entry_low, 2),
                'entry_high': round(entry_high, 2),
                'entry_reason': entry_reason,
                'stop_loss': round(stop_loss, 2),
                'stop_loss_pct': round(sl_pct, 2),
                'target_1': round(t1, 2),
                'target_2': round(t2, 2),
                'target_3': round(t3, 2),
                'rr_ratio': round(rr_t1, 2),
                'timing_signals': timing_signals,
                'avoid_signals': avoid_signals,
                'indicators': {
                    'rsi': round(rsi, 1),
                    'macd': round(macd, 2),
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2),
                    'atr': round(atr, 2),
                    'above_ma20': above_ma20,
                    'above_ma50': above_ma50,
                },
                'fundamentals': fundamentals,
                'fundamental_score': fund_score,
            }

        except Exception as e:
            logger.error(f"Investment analysis error for {symbol}: {e}")
            return {}

    def analyze_watchlist(self, symbols: List[str], news: List[Dict] = None,
                          sector_predictions: Dict = None) -> List[Dict]:
        """Analyze entire watchlist and rank by opportunity"""
        results = []
        for sym in symbols:
            sentiment = 0
            if news:
                for n in news:
                    text = n.get('title', '') + n.get('content', '')
                    if sym.lower() in text.lower():
                        sentiment += n.get('sentiment_score', 0)

            sector_dir = 'NEUTRAL'
            if sector_predictions:
                # find which sector this stock belongs to
                for sect, pred in sector_predictions.items():
                    if sym in pred.get('stocks', []):
                        sector_dir = pred.get('direction', 'NEUTRAL')
                        break

            analysis = self.analyze_stock(sym, sentiment_score=sentiment,
                                          sector_direction=sector_dir)
            if analysis:
                results.append(analysis)

        # Sort by action score (highest first)
        results.sort(key=lambda x: x.get('action_score', 50), reverse=True)
        return results


# ============================================================
# NEWS-STOCK CORRELATOR
# ============================================================

class NewsStockCorrelator:
    """
    Correlates news sentiment with actual stock price movements.
    Identifies which stocks are mentioned in news and how that
    correlates with their price moves.
    """

    # Company name / ticker mappings for Indian market
    COMPANY_ALIASES = {
        'reliance': 'RELIANCE', 'ril': 'RELIANCE',
        'tcs': 'TCS', 'tata consultancy': 'TCS',
        'infosys': 'INFY', 'infy': 'INFY',
        'hdfc bank': 'HDFCBANK', 'hdfc': 'HDFCBANK',
        'icici bank': 'ICICIBANK', 'icici': 'ICICIBANK',
        'sbi': 'SBIN', 'state bank': 'SBIN',
        'wipro': 'WIPRO',
        'hcl': 'HCLTECH', 'hcl tech': 'HCLTECH',
        'maruti': 'MARUTI', 'maruti suzuki': 'MARUTI',
        'bajaj finance': 'BAJFINANCE',
        'tata motors': 'TATAMOTORS',
        'sun pharma': 'SUNPHARMA',
        'kotak': 'KOTAKBANK', 'kotak bank': 'KOTAKBANK',
        'axis bank': 'AXISBANK',
        'itc': 'ITC',
        'ongc': 'ONGC',
        'ntpc': 'NTPC',
        'lt': 'LT', 'larsen': 'LT',
        'bharti airtel': 'BHARTIARTL', 'airtel': 'BHARTIARTL',
        'adani': 'ADANIPORTS', 'adani ports': 'ADANIPORTS',
        'dr reddy': 'DRREDDY', "dr. reddy": 'DRREDDY',
        'cipla': 'CIPLA',
        'lupin': 'LUPIN',
        'tatasteel': 'TATASTEEL', 'tata steel': 'TATASTEEL',
        'hindalco': 'HINDALCO',
        'dlf': 'DLF',
        'asian paints': 'ASIANPAINT',
        'nestle': 'NESTLEIND', 'nestle india': 'NESTLEIND',
        'hindustan unilever': 'HINDUNILVR', 'hul': 'HINDUNILVR',
        'nifty': 'NIFTY500', 'nifty 500': 'NIFTY500', 'nifty500': 'NIFTY500',
        'nifty 50': 'NIFTY500', 'nifty50': 'NIFTY500',
        'sensex': 'SENSEX',
    }

    def correlate(self, news: List[Dict], watchlist: List[str]) -> List[Dict]:
        """
        For each news article, identify mentioned stocks and
        return a correlated list with impact scores.
        """
        correlated = []
        for article in news:
            text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
            mentioned = self._find_mentioned_stocks(text, watchlist)
            if mentioned:
                sentiment = article.get('sentiment_score', 0)
                impact = 'POSITIVE' if sentiment > 0.05 else ('NEGATIVE' if sentiment < -0.05 else 'NEUTRAL')
                correlated.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'sentiment': sentiment,
                    'impact': impact,
                    'mentioned_stocks': mentioned,
                    'published_at': article.get('published_at', ''),
                })

        return correlated

    def _find_mentioned_stocks(self, text: str, watchlist: List[str]) -> List[str]:
        """Find which stocks from watchlist are mentioned in text"""
        found = set()

        # Direct symbol match
        for sym in watchlist:
            if sym.lower() in text:
                found.add(sym)

        # Alias match
        for alias, sym in self.COMPANY_ALIASES.items():
            if alias in text and (sym in watchlist or sym in ['NIFTY500', 'SENSEX']):
                found.add(sym)

        return list(found)

    def aggregate_stock_sentiment(self, news: List[Dict], symbol: str) -> Dict:
        """Aggregate all news sentiment for a specific stock"""
        correlated = self.correlate(news, [symbol])
        if not correlated:
            return {'symbol': symbol, 'score': 0, 'articles': 0, 'impact': 'NEUTRAL'}

        scores = [c['sentiment'] for c in correlated if symbol in c['mentioned_stocks']]
        if not scores:
            return {'symbol': symbol, 'score': 0, 'articles': 0, 'impact': 'NEUTRAL'}

        avg_score = np.mean(scores)
        impact = 'POSITIVE' if avg_score > 0.05 else ('NEGATIVE' if avg_score < -0.05 else 'NEUTRAL')

        return {
            'symbol': symbol,
            'score': avg_score,
            'articles': len(scores),
            'impact': impact,
            'articles_list': [c for c in correlated if symbol in c['mentioned_stocks']],
        }
