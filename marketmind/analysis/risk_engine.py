"""
MarketMind AI - Risk Engine
Institutional-grade risk metrics: VaR, CVaR, stress testing, portfolio Greeks.
"""
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Computes portfolio and single-stock risk metrics.
    - Historical VaR (1-day, 95%/99%)
    - Parametric VaR (normal distribution)
    - CVaR / Expected Shortfall
    - Stress testing (predefined India-relevant scenarios)
    - Beta, volatility, Sharpe, Sortino
    - Concentration risk
    """

    STRESS_SCENARIOS = {
        'Nifty -10%': {'nifty': -0.10, 'usdinr': +0.03, 'crude': -0.08},
        'Nifty -20% (Bear)': {'nifty': -0.20, 'usdinr': +0.07, 'crude': -0.15},
        'Nifty -30% (Crash)': {'nifty': -0.30, 'usdinr': +0.12, 'crude': -0.20},
        'Rate Hike +100bps': {'nifty': -0.06, 'usdinr': -0.01, 'crude': +0.02},
        'INR -5% Depreciation': {'nifty': -0.04, 'usdinr': +0.05, 'crude': +0.04},
        'Oil +30% Spike': {'nifty': -0.05, 'usdinr': +0.03, 'crude': +0.30},
        'FII Sell-off (2008)': {'nifty': -0.22, 'usdinr': +0.09, 'crude': -0.25},
        'COVID Crash (2020)': {'nifty': -0.38, 'usdinr': +0.08, 'crude': -0.55},
        'Budget Day Rally': {'nifty': +0.08, 'usdinr': -0.02, 'crude': 0.0},
        'Nifty +20% (Bull)': {'nifty': +0.20, 'usdinr': -0.05, 'crude': +0.10},
    }

    # Sector betas relative to Nifty (approx)
    SECTOR_BETAS = {
        'IT': 0.85, 'Banking': 1.35, 'Finance': 1.25, 'Auto': 1.10,
        'Pharma': 0.65, 'FMCG': 0.55, 'Energy': 1.05, 'Metal': 1.40,
        'Realty': 1.50, 'Infra': 1.20,
    }

    def __init__(self, price_fetcher=None):
        self._pf = price_fetcher

    def _get_returns(self, symbol: str, days: int = 252) -> Optional[pd.Series]:
        if self._pf is None:
            from marketmind.core.price_fetcher import get_price_fetcher
            self._pf = get_price_fetcher()
        hist = self._pf.get_historical_data(symbol, days=days)
        if hist.empty or len(hist) < 20:
            return None
        return hist['close'].astype(float).pct_change().dropna()

    # ── Single-stock risk ──────────────────────────────────────────────

    def stock_var(self, symbol: str, confidence: float = 0.95,
                  holding_value: float = 100000) -> Dict:
        """Compute VaR for a single stock position."""
        rets = self._get_returns(symbol, days=252)
        if rets is None or len(rets) < 20:
            return {'error': f'Insufficient data for {symbol}'}

        # Historical VaR
        hist_var_95 = float(np.percentile(rets, 5)) * holding_value
        hist_var_99 = float(np.percentile(rets, 1)) * holding_value

        # Parametric VaR (normal)
        mu = float(rets.mean())
        sigma = float(rets.std())
        from scipy.stats import norm
        para_var_95 = (mu - 1.645 * sigma) * holding_value
        para_var_99 = (mu - 2.326 * sigma) * holding_value

        # CVaR (Expected Shortfall) = mean of losses beyond VaR
        tail_95 = rets[rets <= np.percentile(rets, 5)]
        cvar_95 = float(tail_95.mean()) * holding_value if len(tail_95) > 0 else hist_var_95

        # Annualised volatility
        ann_vol = sigma * math.sqrt(252)
        ann_ret = ((1 + mu) ** 252 - 1)
        sharpe = (ann_ret - 0.065) / ann_vol if ann_vol > 0 else 0  # 6.5% RFR

        # Max drawdown
        cumret = (1 + rets).cumprod()
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        return {
            'symbol': symbol,
            'holding_value': holding_value,
            'hist_var_95': round(hist_var_95, 0),
            'hist_var_99': round(hist_var_99, 0),
            'para_var_95': round(para_var_95, 0),
            'para_var_99': round(para_var_99, 0),
            'cvar_95': round(cvar_95, 0),
            'daily_vol_pct': round(sigma * 100, 3),
            'annual_vol_pct': round(ann_vol * 100, 2),
            'annual_return_pct': round(ann_ret * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'interpretation': (
                f"1-day 95% VaR = ₹{abs(hist_var_95):.0f} "
                f"(worst expected loss in 1 of 20 trading days). "
                f"Annual vol = {ann_vol*100:.1f}%."
            ),
        }

    # ── Portfolio risk ─────────────────────────────────────────────────

    def portfolio_var(self, holdings: List[Dict]) -> Dict:
        """
        Compute portfolio-level VaR using variance-covariance method.
        holdings: [{'symbol': str, 'value': float, 'sector': str}, ...]
        """
        if not holdings:
            return {'error': 'No holdings provided'}

        symbols = [h['symbol'] for h in holdings]
        values = [h.get('value', 100000) for h in holdings]
        total = sum(values) or 1
        weights = [v / total for v in values]

        returns_data = {}
        for sym in symbols:
            r = self._get_returns(sym, days=252)
            if r is not None and len(r) > 20:
                returns_data[sym] = r

        if len(returns_data) < 2:
            return {'error': 'Need at least 2 symbols with data'}

        # Align all return series
        df = pd.DataFrame(returns_data).dropna()
        valid_syms = [s for s in symbols if s in df.columns]
        valid_weights = [w for s, w in zip(symbols, weights) if s in df.columns]
        w_sum = sum(valid_weights) or 1
        valid_weights = [w / w_sum for w in valid_weights]

        df = df[valid_syms]
        w = np.array(valid_weights)

        # Portfolio daily returns
        port_rets = df.dot(w)
        port_var_95 = float(np.percentile(port_rets, 5)) * total
        port_var_99 = float(np.percentile(port_rets, 1)) * total

        tail_95 = port_rets[port_rets <= np.percentile(port_rets, 5)]
        cvar_95 = float(tail_95.mean()) * total

        # Diversification benefit
        individual_vars = [float(np.percentile(df[s], 5)) * total * v_w
                           for s, v_w in zip(valid_syms, valid_weights)]
        undiversified_var = sum(abs(v) for v in individual_vars)
        diversification_ratio = abs(port_var_95) / undiversified_var if undiversified_var else 1

        # Concentration risk (Herfindahl index)
        hhi = sum(w_i ** 2 for w_i in valid_weights)

        # Sector concentration
        sector_map: Dict[str, float] = {}
        for h in holdings:
            sec = h.get('sector', 'Other')
            v = h.get('value', 0)
            sector_map[sec] = sector_map.get(sec, 0) + v / total
        max_sector = max(sector_map.values()) if sector_map else 0

        ann_vol = float(port_rets.std()) * math.sqrt(252)
        ann_ret = float((1 + port_rets.mean()) ** 252 - 1)
        sharpe = (ann_ret - 0.065) / ann_vol if ann_vol > 0 else 0

        return {
            'total_value': total,
            'portfolio_var_95': round(port_var_95, 0),
            'portfolio_var_99': round(port_var_99, 0),
            'portfolio_cvar_95': round(cvar_95, 0),
            'diversification_ratio': round(diversification_ratio, 3),
            'concentration_hhi': round(hhi, 4),
            'max_sector_weight': round(max_sector * 100, 1),
            'sector_weights': {k: round(v * 100, 1) for k, v in sector_map.items()},
            'annual_vol_pct': round(ann_vol * 100, 2),
            'annual_return_pct': round(ann_ret * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'symbols_analyzed': valid_syms,
            'interpretation': (
                f"Portfolio daily 95% VaR = ₹{abs(port_var_95):.0f}. "
                f"Diversification reduces risk by {((1-diversification_ratio)*100):.1f}%. "
                f"{'⚠️ High concentration: ' + max(sector_map,key=sector_map.get) if max_sector > 0.4 else 'Sector concentration OK'}."
            ),
        }

    # ── Stress testing ─────────────────────────────────────────────────

    def stress_test(self, holdings: List[Dict]) -> Dict:
        """
        Apply predefined stress scenarios to a portfolio.
        Uses sector betas to estimate impact.
        """
        if not holdings:
            return {'scenarios': [], 'worst_case': 0}

        total_value = sum(h.get('value', 0) for h in holdings)

        # Build sector exposure
        sector_exposure: Dict[str, float] = {}
        for h in holdings:
            sec = h.get('sector', 'Banking')
            val = h.get('value', 0)
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val

        results = []
        for scenario_name, shocks in self.STRESS_SCENARIOS.items():
            nifty_shock = shocks.get('nifty', 0)
            portfolio_pnl = 0

            for sec, exposure in sector_exposure.items():
                beta = self.SECTOR_BETAS.get(sec, 1.0)
                stock_impact = beta * nifty_shock
                portfolio_pnl += exposure * stock_impact

            pnl_pct = portfolio_pnl / total_value * 100 if total_value else 0
            results.append({
                'scenario': scenario_name,
                'nifty_shock_pct': round(nifty_shock * 100, 1),
                'portfolio_pnl': round(portfolio_pnl, 0),
                'pnl_pct': round(pnl_pct, 2),
            })

        results.sort(key=lambda x: x['portfolio_pnl'])
        worst = results[0] if results else {}
        best = results[-1] if results else {}

        return {
            'scenarios': results,
            'total_value': total_value,
            'worst_case': worst.get('portfolio_pnl', 0),
            'worst_scenario': worst.get('scenario', ''),
            'best_case': best.get('portfolio_pnl', 0),
            'best_scenario': best.get('scenario', ''),
        }

    def concentration_analysis(self, holdings: List[Dict]) -> Dict:
        """Analyse portfolio concentration risk."""
        total = sum(h.get('value', 0) for h in holdings) or 1
        stocks = sorted(holdings, key=lambda h: h.get('value', 0), reverse=True)
        top5_pct = sum(h.get('value', 0) for h in stocks[:5]) / total * 100
        top10_pct = sum(h.get('value', 0) for h in stocks[:10]) / total * 100
        hhi = sum((h.get('value', 0) / total) ** 2 for h in holdings)

        return {
            'top5_concentration': round(top5_pct, 1),
            'top10_concentration': round(top10_pct, 1),
            'hhi': round(hhi, 4),
            'effective_positions': round(1 / hhi) if hhi > 0 else 0,
            'risk_level': (
                'High' if top5_pct > 60 or hhi > 0.2 else
                'Medium' if top5_pct > 40 else 'Low'
            ),
        }


_engine: Optional[RiskEngine] = None

def get_risk_engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
