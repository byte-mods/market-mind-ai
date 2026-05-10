"""Factor Model Exposure — cross-sectional Z-score engine for Indian equities.

Implements a 6-factor model inspired by Fama-French 5 + Quality:
  Value, Size, Profitability, Investment, Momentum, Quality.

Because Screener.in provides only *current* fundamentals (no quarterly history),
we use cross-sectional Z-scores rather than time-series regression betas.
A Z-score of +1.5 on Value means "cheaper than 93% of the universe" —
interpretable without a multi-year fundamental panel.

Portfolio attribution is the weighted average of constituent exposures.
Factor momentum measures which factors have been working recently via
cross-sectional correlation with recent returns.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default universe — top NSE liquid names that Screener.in covers well.
# Callers can override with any symbol list.
DEFAULT_UNIVERSE: List[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
    "BHARTIARTL", "ITC", "LT", "HINDUNILVR", "AXISBANK", "KOTAKBANK",
    "MARUTI", "TATAMOTORS", "M&M", "TATASTEEL", "SUNPHARMA", "ULTRACEMCO",
    "NESTLEIND", "ONGC", "POWERGRID", "NTPC", "COALINDIA", "BAJFINANCE",
    "BAJAJFINSV", "HCLTECH", "WIPRO", "TECHM", "ADANIPORTS", "ADANIENT",
    "GRASIM", "JSWSTEEL", "BPCL", "BRITANNIA", "CIPLA", "APOLLOHOSP",
    "DRREDDY", "EICHERMOT", "DIVISLAB", "HEROMOTOCO", "BAJAJ-AUTO",
    "HAL", "PNB", "ZOMATO", "IRCTC", "BEL", "GAIL", "SHRIRAMFIN",
    "INDUSINDBK", "HDFCLIFE", "LUPIN", "AUROPHARMA", "DLF", "ABB",
    "PIDILITIND", "DABUR", "TRENT", "ASIANPAINT", "HAVELLS", "BERGEPAINT",
    "COLPAL", "MARICO", "TITAN", "DMART", "NYKAA", "PAYTM",
]

# Factors and their direction (higher raw score = higher exposure)
_FACTOR_DIRECTION = {
    "value": 1,          # low P/B → high value exposure
    "size": 1,           # large market cap → high size exposure
    "profitability": 1,  # high ROE → high profitability exposure
    "investment": 1,     # high revenue growth → high investment exposure
    "momentum": 1,       # high 12-1m return → high momentum exposure
    "quality": 1,        # high ROCE, low D/E → high quality exposure
}


def _z_score(series: np.ndarray) -> np.ndarray:
    """Robust Z-score: (x - median) / MAD × 1.4826.

    Uses median absolute deviation instead of standard deviation to be
    resistant to outliers (extreme P/B or market-cap values).
    NaN entries are preserved (not converted to 0.0).
    """
    # Work on a copy so we don't modify the original
    out = np.full_like(series, np.nan, dtype=float)
    valid = ~np.isnan(series)
    if valid.sum() == 0:
        return out
    median = float(np.median(series[valid]))
    mad = float(np.median(np.abs(series[valid] - median)))
    std_est = mad * 1.4826  # consistent estimator for normal std
    if std_est < 1e-12:
        # Constant series: all valid entries get 0.0, NaNs stay NaN
        out[valid] = 0.0
        return out
    out[valid] = (series[valid] - median) / std_est
    return out


def _safe_float(val) -> Optional[float]:
    """Coerce to float, returning None on failure or NaN/Inf."""
    try:
        f = float(val)
        return f if not math.isnan(f) and math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class FactorEngine:
    """Compute factor exposures and portfolio attribution."""

    def __init__(self, price_fetcher=None):
        self._pf = price_fetcher

    def _get_pf(self):
        if self._pf is None:
            from marketmind.core.price_fetcher import get_price_fetcher
            self._pf = get_price_fetcher()
        return self._pf

    # ── Universe snapshot ───────────────────────────────────────────────

    def build_universe_snapshot(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Fetch fundamentals + momentum for each symbol in the universe.

        ``symbols`` may be an empty list to force an empty universe (useful for
        testing edge cases). ``None`` falls back to ``DEFAULT_UNIVERSE``.
        """
        universe = DEFAULT_UNIVERSE if symbols is None else list(symbols)
        pf = self._get_pf()
        snapshot: Dict[str, Dict] = {}

        for sym in universe:
            try:
                fund = pf._get_screener_fundamentals(sym) or {}
                hist = pf.get_historical_data(sym, days=300)
            except Exception as e:
                logger.debug(f"Factor snapshot skip {sym}: {e}")
                continue

            row: Dict[str, float] = {}

            # Valuation
            pb = _safe_float(fund.get("pb_ratio"))
            pe = _safe_float(fund.get("pe_ratio"))
            if pb is not None and pb > 0:
                row["pb"] = pb
            if pe is not None and pe > 0:
                row["pe"] = pe

            # Profitability
            roe = _safe_float(fund.get("roe"))
            roce = _safe_float(fund.get("roce"))
            if roe is not None:
                row["roe"] = roe
            if roce is not None:
                row["roce"] = roce

            # Size
            mcap = _safe_float(fund.get("market_cap"))
            if mcap is not None and mcap > 0:
                row["market_cap"] = mcap

            # Growth (proxy for investment)
            rev_g = _safe_float(fund.get("revenue_growth"))
            prf_g = _safe_float(fund.get("profit_growth"))
            if rev_g is not None:
                row["revenue_growth"] = rev_g
            if prf_g is not None:
                row["profit_growth"] = prf_g

            # Leverage
            de = _safe_float(fund.get("debt_equity"))
            if de is not None:
                row["debt_equity"] = de

            # Momentum: 12-1 month return (skip most recent month to avoid reversal)
            if hist is not None and not hist.empty and len(hist) >= 230:
                try:
                    close = hist["close"].astype(float)
                    # Classic 12-1 momentum: price 1 month ago / price 12 months ago - 1
                    mom = float(close.iloc[-22] / close.iloc[-230] - 1)
                    row["momentum_12m"] = mom
                except Exception:
                    pass
            elif hist is not None and not hist.empty and len(hist) >= 22:
                try:
                    close = hist["close"].astype(float)
                    # Fallback: use available history as momentum proxy
                    mom = float(close.iloc[-1] / close.iloc[-22] - 1)
                    row["momentum_12m"] = mom
                except Exception:
                    pass

            if row:
                snapshot[sym] = row

        return snapshot

    # ── Factor exposures (Z-scores) ─────────────────────────────────────

    def compute_exposures(
        self,
        snapshot: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Convert raw fundamentals into Z-scores per factor.

        Returns {symbol: {factor: z_score, ...}}.
        Missing factors are omitted (not zero) so callers know data was unavailable.
        """
        if not snapshot:
            return {}

        symbols = list(snapshot.keys())
        n = len(symbols)

        # Extract raw arrays
        def _arr(key):
            return np.array([snapshot[s].get(key) for s in symbols], dtype=float)

        pb_arr = _arr("pb")
        pe_arr = _arr("pe")
        roe_arr = _arr("roe")
        roce_arr = _arr("roce")
        mcap_arr = _arr("market_cap")
        rev_g_arr = _arr("revenue_growth")
        de_arr = _arr("debt_equity")
        mom_arr = _arr("momentum_12m")

        # Compute Z-scores per factor
        # Value: lower P/B = higher value → invert
        value_z = -_z_score(pb_arr) if np.any(~np.isnan(pb_arr)) else None

        # Size: larger market cap = higher size
        size_z = _z_score(np.log1p(mcap_arr)) if np.any(~np.isnan(mcap_arr)) else None

        # Profitability: higher ROE = higher profitability
        prof_z = _z_score(roe_arr) if np.any(~np.isnan(roe_arr)) else None

        # Investment: higher revenue growth = higher investment
        inv_z = _z_score(rev_g_arr) if np.any(~np.isnan(rev_g_arr)) else None

        # Momentum: higher 12-1m return = higher momentum
        mom_z = _z_score(mom_arr) if np.any(~np.isnan(mom_arr)) else None

        # Quality: high ROCE, low D/E
        quality_z = None
        if np.any(~np.isnan(roce_arr)):
            q = _z_score(roce_arr)
            if np.any(~np.isnan(de_arr)):
                # Penalise high leverage
                q = q - 0.5 * _z_score(de_arr)
            quality_z = q

        results: Dict[str, Dict] = {}
        for i, sym in enumerate(symbols):
            exp: Dict[str, float] = {}
            if value_z is not None:
                v = float(value_z[i])
                if not math.isnan(v):
                    exp["value"] = round(v, 3)
            if size_z is not None:
                v = float(size_z[i])
                if not math.isnan(v):
                    exp["size"] = round(v, 3)
            if prof_z is not None:
                v = float(prof_z[i])
                if not math.isnan(v):
                    exp["profitability"] = round(v, 3)
            if inv_z is not None:
                v = float(inv_z[i])
                if not math.isnan(v):
                    exp["investment"] = round(v, 3)
            if mom_z is not None:
                v = float(mom_z[i])
                if not math.isnan(v):
                    exp["momentum"] = round(v, 3)
            if quality_z is not None:
                v = float(quality_z[i])
                if not math.isnan(v):
                    exp["quality"] = round(v, 3)
            if exp:
                results[sym] = exp

        return results

    def get_stock_exposure(
        self,
        symbol: str,
        universe: Optional[List[str]] = None,
    ) -> Dict:
        """Factor exposure for a single stock relative to the universe."""
        syms = DEFAULT_UNIVERSE if universe is None else list(universe)
        if symbol.upper() not in [s.upper() for s in syms]:
            syms = list(syms) + [symbol.upper()]

        snapshot = self.build_universe_snapshot(syms)
        exposures = self.compute_exposures(snapshot)
        sym_upper = symbol.upper()
        exp = exposures.get(sym_upper, {})

        # Percentile interpretation
        percentiles: Dict[str, float] = {}
        for factor in exp:
            vals = [v[factor] for v in exposures.values() if factor in v]
            if vals:
                pct = sum(1 for v in vals if v < exp[factor]) / len(vals) * 100.0
                percentiles[factor] = round(pct, 1)

        return {
            "symbol": sym_upper,
            "exposures": exp,
            "percentiles": percentiles,
            "universe_size": len(exposures),
            "status": "ready" if exp else "insufficient_data",
        }

    # ── Portfolio factor attribution ────────────────────────────────────

    def portfolio_attribution(
        self,
        holdings: List[Dict],
        universe: Optional[List[str]] = None,
    ) -> Dict:
        """Weighted factor exposure of a portfolio.

        ``holdings`` = [{symbol, weight? quantity? current_value?}].
        Weights are normalised from ``current_value`` if present, else
        ``quantity * price``, else equal-weight.
        """
        if not holdings:
            return {"status": "no_holdings", "factors": {}}

        syms = [h.get("symbol", h.get("tradingsymbol", "")).upper() for h in holdings]
        syms = [s for s in syms if s]
        if not syms:
            return {"status": "no_symbols", "factors": {}}

        # Expand universe to include holdings
        all_syms = list(set((DEFAULT_UNIVERSE if universe is None else list(universe)) + syms))
        snapshot = self.build_universe_snapshot(all_syms)
        exposures = self.compute_exposures(snapshot)

        # Compute holding weights
        total = 0.0
        weights: Dict[str, float] = {}
        for h in holdings:
            sym = h.get("symbol", h.get("tradingsymbol", "")).upper()
            if not sym:
                continue
            val = h.get("current_value")
            if val is None:
                qty = h.get("quantity", 0)
                price = h.get("price", h.get("last_price", 0))
                val = qty * price
            val = float(val) if val else 0.0
            weights[sym] = weights.get(sym, 0.0) + val
            total += val

        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        else:
            # Equal weight fallback
            n = len(weights)
            weights = {s: 1.0 / n for s in weights}

        # Re-normalise weights to holdings that have factor data
        covered_syms = [s for s in weights if s in exposures]
        covered_total = sum(weights[s] for s in covered_syms)
        if covered_total > 0:
            norm_weights = {s: weights[s] / covered_total for s in covered_syms}
        else:
            norm_weights = {s: 1.0 / len(covered_syms) for s in covered_syms} if covered_syms else {}

        # Weighted factor exposure
        factor_sums: Dict[str, float] = {}
        for sym, w in norm_weights.items():
            exp = exposures.get(sym, {})
            for f, z in exp.items():
                factor_sums[f] = factor_sums.get(f, 0.0) + w * z

        # Drift: how far each factor is from benchmark (equal-weight universe)
        benchmark: Dict[str, float] = {}
        for f in factor_sums:
            vals = [v[f] for v in exposures.values() if f in v]
            if vals:
                benchmark[f] = float(np.mean(vals))

        drift = {f: round(factor_sums[f] - benchmark.get(f, 0), 3)
                 for f in factor_sums}

        return {
            "status": "ready",
            "portfolio_factors": {f: round(v, 3) for f, v in factor_sums.items()},
            "benchmark_factors": {f: round(v, 3) for f, v in benchmark.items()},
            "factor_drift": drift,
            "holdings_covered": len([s for s in weights if s in exposures]),
            "holdings_total": len(weights),
            "universe_size": len(exposures),
        }

    # ── Factor momentum ─────────────────────────────────────────────────

    def factor_momentum(
        self,
        lookback_days: int = 63,  # ~3 months
        universe: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Measure which factors have been working recently.

        For each factor, compute the Spearman correlation between the factor
        score and the recent return over ``lookback_days``. Positive correlation
        = stocks with high factor exposure outperformed.

        Also returns the top / bottom quintile spread (factor return proxy).
        """
        syms = DEFAULT_UNIVERSE if universe is None else list(universe)
        snapshot = self.build_universe_snapshot(syms)
        exposures = self.compute_exposures(snapshot)
        if not exposures:
            return {}

        pf = self._get_pf()
        recent_rets: Dict[str, float] = {}
        for sym in list(exposures.keys()):
            try:
                hist = pf.get_historical_data(sym, days=lookback_days + 5)
                if hist is not None and not hist.empty and len(hist) >= 2:
                    close = hist["close"].astype(float)
                    ret = float(close.iloc[-1] / close.iloc[0] - 1)
                    recent_rets[sym] = ret
            except Exception:
                continue

        common = [s for s in exposures if s in recent_rets]
        if len(common) < 10:
            return {}

        results: Dict[str, Dict] = {}
        for factor in ["value", "size", "profitability", "investment", "momentum", "quality"]:
            scores = np.array([exposures[s].get(factor, np.nan) for s in common])
            rets = np.array([recent_rets[s] for s in common])
            valid = ~np.isnan(scores)
            if valid.sum() < 10:
                continue

            s_valid = scores[valid]
            r_valid = rets[valid]

            # Spearman correlation (rank-based, robust)
            try:
                from scipy.stats import spearmanr
                # Guard against constant inputs (e.g., all same factor score)
                if np.unique(s_valid).size < 2 or np.unique(r_valid).size < 2:
                    continue
                corr, pval = spearmanr(s_valid, r_valid)
            except Exception:
                continue

            # Quintile spread: return of top 20% minus bottom 20% by factor score
            sorted_idx = np.argsort(s_valid)
            n = len(sorted_idx)
            q20 = max(1, n // 5)
            bottom_ret = float(np.mean(r_valid[sorted_idx[:q20]]))
            top_ret = float(np.mean(r_valid[sorted_idx[-q20:]]))
            spread = top_ret - bottom_ret

            results[factor] = {
                "correlation": round(float(corr), 3) if not math.isnan(corr) else None,
                "p_value": round(float(pval), 4) if not math.isnan(pval) else None,
                "top_quintile_return": round(top_ret * 100, 2),
                "bottom_quintile_return": round(bottom_ret * 100, 2),
                "spread_pct": round(spread * 100, 2),
                "regime": "positive" if corr > 0.15 else ("negative" if corr < -0.15 else "neutral"),
            }

        return results

    # ── Factor summary (convenience) ────────────────────────────────────

    def factor_summary(
        self,
        universe: Optional[List[str]] = None,
    ) -> Dict:
        """One-call summary: exposures for full universe + factor momentum."""
        syms = DEFAULT_UNIVERSE if universe is None else list(universe)
        snapshot = self.build_universe_snapshot(syms)
        exposures = self.compute_exposures(snapshot)
        momentum = self.factor_momentum(lookback_days=63, universe=syms)

        # Aggregate stats
        stats: Dict[str, Dict] = {}
        for factor in ["value", "size", "profitability", "investment", "momentum", "quality"]:
            vals = [v[factor] for v in exposures.values() if factor in v]
            if vals:
                stats[factor] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                    "min": round(float(np.min(vals)), 3),
                    "max": round(float(np.max(vals)), 3),
                }

        return {
            "universe_size": len(exposures),
            "exposures": exposures,
            "momentum": momentum,
            "stats": stats,
        }
