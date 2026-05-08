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
    - Hierarchical Risk Parity (cluster-based, no matrix inversion)
    - Black-Litterman (user views blended with market prior)
    - Markowitz Mean-Variance (maximum Sharpe, minimum variance)
    - Risk Parity (equal risk contribution)
    - Efficient Frontier (range of risk-return tradeoffs)
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
                 days: int = 252,
                 views: Optional[List[Dict]] = None,
                 market_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Optimize portfolio weights.
        objective: 'hrp' | 'black_litterman' | 'max_sharpe' | 'min_variance' | 'risk_parity' | 'equal_weight'
        views: list of dicts for Black-Litterman, e.g.
            [{'assets':['TCS'],'type':'absolute','magnitude':0.10,'confidence':0.6},
             {'assets':['TCS','RELIANCE'],'type':'relative','magnitude':0.03,'confidence':0.5}]
        market_weights: dict of symbol->weight for market-cap prior (default equal-weight)
        """
        df = self._build_returns_matrix(symbols, days)
        if df is None:
            return {'error': 'Insufficient historical data for optimization'}

        valid_syms = list(df.columns)
        n = len(valid_syms)
        mean_rets = df.mean().values
        cov = df.cov().values

        stats_mean_rets = mean_rets
        if objective == 'equal_weight':
            weights = np.ones(n) / n

        elif objective == 'min_variance':
            weights = self._min_variance(mean_rets, cov, n)

        elif objective == 'risk_parity':
            weights = self._risk_parity(cov, n)

        elif objective == 'hrp':
            weights = self._hrp(df)

        elif objective == 'black_litterman':
            bl_result = self._black_litterman(mean_rets, cov, n, valid_syms, views, market_weights)
            if isinstance(bl_result, tuple) and len(bl_result) == 2:
                weights, bl_mean_rets = bl_result
            else:
                weights = bl_result
                bl_mean_rets = None
            stats_mean_rets = bl_mean_rets if bl_mean_rets is not None else mean_rets

        else:  # max_sharpe (default)
            weights = self._max_sharpe(mean_rets, cov, n)

        ret, vol, sharpe = self._portfolio_stats(weights, stats_mean_rets, cov)

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

    # ── Hierarchical Risk Parity ───────────────────────────────────────

    def _hrp(self, df: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity via recursive bisection (López de Prado)."""
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
            cov = df.cov()
            corr = df.corr()
            dist = np.sqrt(0.5 * (1 - corr))
            dist_arr = dist.values
            np.fill_diagonal(dist_arr, 0)
            condensed = squareform(dist_arr, checks=False)
            link = linkage(condensed, method='single')
            sort_ix = self._get_quasi_diag(link)
            w = self._get_recursive_bisection(cov, sort_ix)
            return w / w.sum()
        except Exception as e:
            logger.debug(f"HRP error: {e}")
            return np.ones(len(df.columns)) / len(df.columns)

    def _get_quasi_diag(self, link):
        """Sort items by hierarchical clustering distance."""
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _get_cluster_var(self, cov, cluster_items):
        """Variance of inverse-variance portfolio within cluster."""
        cov_slice = cov.iloc[cluster_items, cluster_items]
        w_ = 1.0 / np.diag(cov_slice)
        w_ /= w_.sum()
        return float(np.dot(w_, np.dot(cov_slice, w_)))

    def _get_recursive_bisection(self, cov, sorted_idx):
        """Allocate weights recursively by inverse cluster variance."""
        w = pd.Series(1.0, index=sorted_idx)
        clusters = [sorted_idx]
        while len(clusters) > 0:
            new_clusters = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    c1 = c[:mid]
                    c2 = c[mid:]
                    var1 = self._get_cluster_var(cov, c1)
                    var2 = self._get_cluster_var(cov, c2)
                    denom = var1 + var2
                    alpha = var2 / denom if denom > 0 else 0.5
                    w[c1] *= alpha
                    w[c2] *= 1.0 - alpha
                    new_clusters.extend([c1, c2])
            clusters = new_clusters
        return w.values

    # ── Black-Litterman ────────────────────────────────────────────────

    def _black_litterman(self, mean_rets, cov, n, valid_syms,
                         views=None, market_weights=None):
        """Simplified Black-Litterman: blend user views with market prior.

        Returns (weights, posterior_daily_total_returns) or (weights, None) on fallback.
        """
        try:
            # Market prior (equal-weight fallback)
            if market_weights is not None:
                w_eq = np.array([market_weights.get(s, 1.0 / n) for s in valid_syms])
            else:
                w_eq = np.ones(n) / n
            w_eq = w_eq / w_eq.sum()

            # Implied risk aversion from market portfolio (daily units)
            market_ret = float(np.dot(w_eq, mean_rets))
            market_var = float(np.dot(w_eq, np.dot(cov, w_eq)))
            delta = (market_ret - self.RISK_FREE_RATE / 252.0) / market_var if market_var > 0 else 2.5

            # Prior excess returns (daily)
            Pi = delta * np.dot(cov, w_eq)
            Pi_total = Pi + self.RISK_FREE_RATE / 252.0

            if views is None or len(views) == 0:
                # No views: return market prior weights + prior returns
                if market_weights is not None:
                    w = np.array([market_weights.get(s, 1.0 / n) for s in valid_syms])
                    return w / w.sum(), Pi_total
                return np.ones(n) / n, Pi_total

            tau = 0.05
            M = tau * cov

            P = []
            Q = []
            omega_diag = []

            for view in views:
                assets = view.get('assets', [])
                if len(assets) == 0:
                    continue

                row = np.zeros(n)
                if view.get('type') == 'relative' and len(assets) >= 2:
                    if assets[0] in valid_syms and assets[1] in valid_syms:
                        row[valid_syms.index(assets[0])] = 1.0
                        row[valid_syms.index(assets[1])] = -1.0
                    else:
                        continue
                    Q.append(view.get('magnitude', 0.0) / 252.0)  # annual → daily
                else:
                    # absolute
                    valid_assets = [a for a in assets if a in valid_syms]
                    if not valid_assets:
                        continue
                    for a in valid_assets:
                        row[valid_syms.index(a)] = 1.0 / len(valid_assets)
                    Q.append(view.get('magnitude', 0.0) / 252.0)

                P.append(row)
                conf = view.get('confidence', 0.5)
                conf = max(0.01, min(0.99, conf))
                prior_var_view = float(np.dot(row, np.dot(M, row)))
                omega_diag.append(((1.0 - conf) / conf) * prior_var_view)

            if len(P) == 0:
                return w_eq, Pi_total

            P = np.array(P)
            Q = np.array(Q)
            Omega = np.diag(omega_diag)

            # Posterior expected returns: Er = Pi + M @ P.T @ inv(P @ M @ P.T + Omega) @ (Q - P @ Pi)
            pmp = np.dot(P, np.dot(M, P.T)) + Omega
            # Add tiny jitter for numerical stability
            pmp += np.eye(pmp.shape[0]) * 1e-12
            inv_pmp = np.linalg.inv(pmp)
            rhs = Q - np.dot(P, Pi)
            Er = Pi + np.dot(M, np.dot(P.T, np.dot(inv_pmp, rhs)))
            Er_total = Er + self.RISK_FREE_RATE / 252.0

            # Optimize on posterior total returns using existing max-sharpe solver
            weights = self._max_sharpe(Er_total, cov, n)
            return weights, Er_total
        except Exception as e:
            logger.debug(f"Black-Litterman error: {e}")
            return np.ones(n) / n, None


_optimizer: Optional[PortfolioOptimizer] = None

def get_optimizer() -> PortfolioOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PortfolioOptimizer()
    return _optimizer
