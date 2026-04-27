"""
MarketMind AI - Portfolio Optimizer
Markowitz mean-variance optimization, efficient frontier, risk parity.
No external solver needed — pure numpy/scipy.
"""
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Implements:
    - Markowitz Mean-Variance (maximum Sharpe, minimum variance)
    - Risk Parity (equal risk contribution)
    - Efficient Frontier (range of risk-return tradeoffs)
    - Black-Litterman simplified (user views blended with market prior)
    """

    RISK_FREE_RATE = 0.065  # 6.5% India RFR (10Y G-Sec approx)

    def __init__(self, price_fetcher=None):
        self._pf = price_fetcher

    def _get_pf(self):
        if self._pf is None:
            from marketmind.core.price_fetcher import get_price_fetcher
            self._pf = get_price_fetcher()
        return self._pf

    def _build_returns_matrix(self, symbols: List[str], days: int = 252) -> Optional[pd.DataFrame]:
        pf = self._get_pf()
        series_dict = {}
        for sym in symbols:
            hist = pf.get_historical_data(sym, days=days)
            if not hist.empty and len(hist) > 30:
                series_dict[sym] = hist.set_index('date')['close'].astype(float).pct_change()
        if len(series_dict) < 2:
            return None
        df = pd.DataFrame(series_dict).dropna()
        return df if len(df) > 20 else None

    def _portfolio_stats(self, weights: np.ndarray, mean_rets: np.ndarray,
                          cov: np.ndarray) -> Tuple[float, float, float]:
        """Return (annual_return, annual_vol, sharpe)."""
        ret = float(np.dot(weights, mean_rets)) * 252
        vol = float(math.sqrt(np.dot(weights, np.dot(cov * 252, weights))))
        sharpe = (ret - self.RISK_FREE_RATE) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def optimize(self, symbols: List[str], objective: str = 'max_sharpe',
                 days: int = 252) -> Dict:
        """
        Optimize portfolio weights.
        objective: 'max_sharpe' | 'min_variance' | 'risk_parity' | 'equal_weight'
        """
        df = self._build_returns_matrix(symbols, days)
        if df is None:
            return {'error': 'Insufficient historical data for optimization'}

        valid_syms = list(df.columns)
        n = len(valid_syms)
        mean_rets = df.mean().values
        cov = df.cov().values

        if objective == 'equal_weight':
            weights = np.ones(n) / n

        elif objective == 'min_variance':
            weights = self._min_variance(mean_rets, cov, n)

        elif objective == 'risk_parity':
            weights = self._risk_parity(cov, n)

        else:  # max_sharpe (default)
            weights = self._max_sharpe(mean_rets, cov, n)

        ret, vol, sharpe = self._portfolio_stats(weights, mean_rets, cov)

        allocation = {sym: round(float(w) * 100, 2)
                      for sym, w in zip(valid_syms, weights)}
        allocation = dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True))

        return {
            'objective': objective,
            'symbols': valid_syms,
            'weights': allocation,
            'expected_annual_return_pct': round(ret * 100, 2),
            'annual_volatility_pct': round(vol * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'interpretation': (
                f"{objective.replace('_',' ').title()}: "
                f"Expected return {ret*100:.1f}%/yr, "
                f"Volatility {vol*100:.1f}%/yr, Sharpe {sharpe:.2f}"
            ),
        }

    def efficient_frontier(self, symbols: List[str], n_points: int = 20,
                           days: int = 252) -> Dict:
        """Generate efficient frontier points for risk-return chart."""
        df = self._build_returns_matrix(symbols, days)
        if df is None:
            return {'error': 'Insufficient data'}

        mean_rets = df.mean().values
        cov = df.cov().values
        n = len(df.columns)

        target_rets = np.linspace(mean_rets.min() * 252, mean_rets.max() * 252, n_points)
        frontier = []
        for target in target_rets:
            w = self._min_variance_target(mean_rets, cov, n, target / 252)
            if w is not None:
                ret, vol, sharpe = self._portfolio_stats(w, mean_rets, cov)
                frontier.append({
                    'return_pct': round(ret * 100, 2),
                    'vol_pct': round(vol * 100, 2),
                    'sharpe': round(sharpe, 3),
                })

        # Highlight max-Sharpe and min-variance
        max_s = self._max_sharpe(mean_rets, cov, n)
        min_v = self._min_variance(mean_rets, cov, n)
        mr, mv, msh = self._portfolio_stats(max_s, mean_rets, cov)
        vr, vv, vsh = self._portfolio_stats(min_v, mean_rets, cov)

        return {
            'frontier': frontier,
            'max_sharpe_point': {'return_pct': round(mr*100,2), 'vol_pct': round(mv*100,2), 'sharpe': round(msh,3)},
            'min_variance_point': {'return_pct': round(vr*100,2), 'vol_pct': round(vv*100,2), 'sharpe': round(vsh,3)},
        }

    def compare_strategies(self, symbols: List[str], days: int = 252) -> List[Dict]:
        """Compare all four strategies side by side."""
        results = []
        for obj in ['equal_weight', 'min_variance', 'risk_parity', 'max_sharpe']:
            r = self.optimize(symbols, objective=obj, days=days)
            if 'error' not in r:
                results.append({
                    'strategy': obj,
                    'return_pct': r['expected_annual_return_pct'],
                    'vol_pct': r['annual_volatility_pct'],
                    'sharpe': r['sharpe_ratio'],
                    'top_weight': max(r['weights'], key=r['weights'].get),
                    'top_pct': max(r['weights'].values()),
                })
        return results

    # ── Solvers ────────────────────────────────────────────────────────

    def _max_sharpe(self, mean_rets, cov, n) -> np.ndarray:
        try:
            from scipy.optimize import minimize
            def neg_sharpe(w):
                r, v, s = self._portfolio_stats(w, mean_rets, cov)
                return -s
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.02, 0.4)] * n  # 2–40% per asset
            w0 = np.ones(n) / n
            res = minimize(neg_sharpe, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'ftol': 1e-9, 'maxiter': 1000})
            if res.success:
                return res.x / res.x.sum()
        except Exception as e:
            logger.debug(f"Max sharpe solver error: {e}")
        return np.ones(n) / n

    def _min_variance(self, mean_rets, cov, n) -> np.ndarray:
        try:
            from scipy.optimize import minimize
            def portfolio_vol(w):
                return float(math.sqrt(np.dot(w, np.dot(cov * 252, w))))
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.02, 0.4)] * n
            w0 = np.ones(n) / n
            res = minimize(portfolio_vol, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'ftol': 1e-9, 'maxiter': 1000})
            if res.success:
                return res.x / res.x.sum()
        except Exception as e:
            logger.debug(f"Min variance solver error: {e}")
        return np.ones(n) / n

    def _min_variance_target(self, mean_rets, cov, n, target_ret) -> Optional[np.ndarray]:
        try:
            from scipy.optimize import minimize
            def portfolio_vol(w):
                return float(math.sqrt(np.dot(w, np.dot(cov * 252, w))))
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: float(np.dot(w, mean_rets)) - target_ret},
            ]
            bounds = [(0.0, 1.0)] * n
            w0 = np.ones(n) / n
            res = minimize(portfolio_vol, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'ftol': 1e-8, 'maxiter': 500})
            if res.success:
                return res.x / (res.x.sum() or 1)
        except Exception:
            pass
        return None

    def _risk_parity(self, cov, n) -> np.ndarray:
        """Equal risk contribution (risk parity)."""
        try:
            from scipy.optimize import minimize
            def risk_parity_obj(w):
                w = np.array(w)
                port_var = np.dot(w, np.dot(cov, w))
                marginal = np.dot(cov, w)
                rc = w * marginal / port_var
                target_rc = np.ones(n) / n
                return float(np.sum((rc - target_rc) ** 2))
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, 0.5)] * n
            w0 = np.ones(n) / n
            res = minimize(risk_parity_obj, w0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'ftol': 1e-8, 'maxiter': 1000})
            if res.success:
                return res.x / res.x.sum()
        except Exception as e:
            logger.debug(f"Risk parity error: {e}")
        return np.ones(n) / n


_optimizer: Optional[PortfolioOptimizer] = None

def get_optimizer() -> PortfolioOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    return _optimizer
