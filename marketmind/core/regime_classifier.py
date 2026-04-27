"""
MarketMind AI - Market Regime Classifier (W1.2)

Classifies the current Indian market into one of five regimes from the tuple:
  (nifty returns, india vix, market breadth, sector dispersion).

Regimes: Trending Bull, Range, Volatile, Crash, Recovery.

Approach: deterministic rules over windowed statistics (per plan.md note that
small N for Indian regimes makes pure HMM unreliable — supplement with rules).
We walk the recent 120 days of Nifty 500 closes, classify each day with the
same ruleset, then derive `days_in_state` and an empirical transition matrix.

The classifier is read-only over existing fetchers — no new external calls.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATES = ("Trending Bull", "Range", "Volatile", "Crash", "Recovery")

# Hooks: which strategy families are active in each regime.
STRATEGY_HOOKS: Dict[str, Dict[str, str]] = {
    "Trending Bull":  {"trend_following": "active", "mean_reversion": "muted",  "vol_selling": "active"},
    "Range":          {"trend_following": "muted",  "mean_reversion": "active", "vol_selling": "active"},
    "Volatile":       {"trend_following": "muted",  "mean_reversion": "muted",  "vol_selling": "muted"},
    "Crash":          {"trend_following": "short_only", "mean_reversion": "muted", "vol_selling": "blocked"},
    "Recovery":       {"trend_following": "active", "mean_reversion": "active", "vol_selling": "muted"},
}


def _annualised_vol(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(252) * 100)


def _classify_window(
    ret_20d: float,
    ret_60d: float,
    vol_20d: float,
    vol_60d: float,
    vix: float,
    breadth_adr: float,
    sector_dispersion: float,
) -> Tuple[str, float]:
    """
    Classify a single window. Returns (state, confidence in [0,1]).

    Rule precedence (most extreme first):
      1. Crash:        sharp drawdown + high vol/VIX
      2. Recovery:     prior 60d red, last 20d positive turn
      3. Volatile:     realized vol elevated vs long-term, breadth mixed
      4. Trending Bull: positive return + strong breadth + benign VIX
      5. Range:        flat returns + benign vol  (default fallback)
    """
    # 1) Crash
    if ret_20d < -8 and (vix > 22 or vol_20d > 25):
        conf = min(1.0, abs(ret_20d) / 15 * 0.6 + (vix / 35) * 0.4)
        return "Crash", round(conf, 2)

    # 2) Recovery: ugly trailing 60d but recent 20d up
    if ret_60d < -5 and ret_20d > 2:
        conf = min(1.0, (ret_20d / 8) * 0.6 + (abs(ret_60d) / 15) * 0.4)
        return "Recovery", round(max(conf, 0.4), 2)

    # 3) Volatile (vol elevated but no clean trend)
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0
    if vol_ratio > 1.4 and abs(ret_20d) < 5 and 0.6 < breadth_adr < 1.5:
        conf = min(1.0, (vol_ratio - 1.0) * 0.8)
        return "Volatile", round(max(conf, 0.45), 2)

    # 4) Trending Bull — positive trend with confirming breadth, moderate VIX
    if ret_20d > 3 and breadth_adr > 1.2 and vix < 22:
        conf = min(1.0, (ret_20d / 8) * 0.5 + (breadth_adr / 2.5) * 0.3 + ((22 - vix) / 22) * 0.2)
        return "Trending Bull", round(max(conf, 0.5), 2)

    # 5) Range — default
    flatness = max(0.0, 1.0 - abs(ret_20d) / 5)
    breadth_neutrality = max(0.0, 1.0 - abs(breadth_adr - 1.0))
    conf = min(1.0, 0.4 + 0.3 * flatness + 0.3 * breadth_neutrality)
    return "Range", round(conf, 2)


def _compute_sector_dispersion() -> float:
    """
    Cross-section std (in %) of NSE sector index daily changes.
    Reads NSE allIndices via the macro fetcher's session — no new HTTP plumbing.
    Falls back to a neutral value if data is unavailable.
    """
    try:
        from marketmind.core.macro_fetcher import get_macro_fetcher
        data = get_macro_fetcher()._nse_get("allIndices")
        if not data:
            return 1.5
        sector_keywords = ("NIFTY BANK", "NIFTY IT", "NIFTY AUTO", "NIFTY FMCG",
                           "NIFTY PHARMA", "NIFTY METAL", "NIFTY ENERGY",
                           "NIFTY REALTY", "NIFTY MEDIA", "NIFTY PSU BANK")
        changes: List[float] = []
        for idx in data.get("data", []):
            name = (idx.get("index") or idx.get("indexSymbol") or "").upper()
            if any(k in name for k in sector_keywords):
                pct = idx.get("percentChange")
                try:
                    changes.append(float(pct))
                except (TypeError, ValueError):
                    continue
        if len(changes) < 3:
            return 1.5
        return round(float(np.std(changes, ddof=1)), 3)
    except Exception as e:
        logger.debug(f"sector_dispersion fallback: {e}")
        return 1.5


class RegimeClassifier:
    NIFTY_BENCHMARK = "NIFTY500"
    LOOKBACK_DAYS = 180   # enough history for 60-day windows + a 120-day walk
    WALK_DAYS = 120       # how many recent days we re-classify for state-history

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_ts: float = 0.0
        self._cache_ttl = 600  # 10 min

    def _get_nifty_history(self) -> Optional[pd.DataFrame]:
        try:
            from marketmind.core.price_fetcher import get_price_fetcher
            df = get_price_fetcher().get_historical_data(self.NIFTY_BENCHMARK, days=self.LOOKBACK_DAYS)
            if df is None or df.empty or "close" not in df.columns:
                return None
            df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning(f"regime: nifty history fetch failed: {e}")
            return None

    def _macro_now(self) -> Tuple[float, float]:
        """Return (vix_value, breadth_adr) using cached macro fetcher."""
        try:
            from marketmind.core.macro_fetcher import get_macro_fetcher
            mf = get_macro_fetcher()
            vix = float(mf.get_india_vix().get("value", 14.5))
            breadth_adr = float(mf.get_market_breadth().get("adr", 1.0))
            return vix, breadth_adr
        except Exception as e:
            logger.debug(f"regime: macro fallback: {e}")
            return 14.5, 1.0

    def _walk_history(
        self,
        df: pd.DataFrame,
        vix_today: float,
        breadth_today: float,
        sector_disp_today: float,
    ) -> List[str]:
        """
        Re-classify each of the last WALK_DAYS using the rule-set.
        For historical days we don't have intraday VIX/breadth, so we proxy:
          - VIX proxy: rolling 20-day realized vol mapped to a VIX-like scale.
          - Breadth proxy: sign of 5-day return scaled into adr-like value.
          - Sector dispersion: today's value (we don't store history).
        Today's classification uses real VIX/breadth instead.
        """
        closes = df["close"].astype(float).values
        if len(closes) < 65:
            return []
        rets = np.diff(np.log(closes))
        states: List[str] = []
        n = len(closes)
        start = max(60, n - self.WALK_DAYS)
        for i in range(start, n):
            window_20 = rets[i - 20:i]
            window_60 = rets[i - 60:i]
            ret_20d = float((closes[i] / closes[i - 20] - 1) * 100)
            ret_60d = float((closes[i] / closes[i - 60] - 1) * 100)
            vol_20d = _annualised_vol(window_20)
            vol_60d = _annualised_vol(window_60)
            is_today = (i == n - 1)
            if is_today:
                vix = vix_today
                breadth = breadth_today
            else:
                # Realized-vol → VIX-like proxy (Indian VIX historically ~1.05× realized)
                vix = max(8.0, min(45.0, vol_20d * 1.05))
                # Breadth proxy: 5-day return centred on 1.0
                ret_5d = float((closes[i] / closes[i - 5] - 1) * 100)
                breadth = max(0.2, min(3.0, 1.0 + ret_5d / 4))
            state, _ = _classify_window(
                ret_20d, ret_60d, vol_20d, vol_60d, vix, breadth, sector_disp_today,
            )
            states.append(state)
        return states

    def _empirical_transitions(self, states: List[str]) -> Dict[str, Dict[str, float]]:
        if len(states) < 2:
            return {s: {t: 0.0 for t in STATES} for s in STATES}
        counts: Dict[str, Counter] = {s: Counter() for s in STATES}
        for a, b in zip(states[:-1], states[1:]):
            counts[a][b] += 1
        out: Dict[str, Dict[str, float]] = {}
        for s in STATES:
            row = counts[s]
            total = sum(row.values())
            if total == 0:
                out[s] = {t: 0.0 for t in STATES}
            else:
                out[s] = {t: round(row[t] / total, 3) for t in STATES}
        return out

    def classify(self, force_refresh: bool = False) -> Dict:
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        df = self._get_nifty_history()
        vix_today, breadth_today = self._macro_now()
        sector_disp = _compute_sector_dispersion()

        if df is None or len(df) < 65:
            # Lean fallback: classify off macro alone with a heuristic 20d/60d set to 0.
            state, conf = _classify_window(
                0.0, 0.0, vix_today * 0.95, vix_today * 0.95, vix_today,
                breadth_today, sector_disp,
            )
            result = {
                "state": state,
                "confidence": conf,
                "days_in_state": 0,
                "signals": {
                    "nifty_return_20d": None, "nifty_return_60d": None,
                    "realized_vol_20d": None, "realized_vol_60d": None,
                    "india_vix": vix_today, "breadth_adr": breadth_today,
                    "sector_dispersion": sector_disp,
                },
                "transition_probs": {s: {t: 0.0 for t in STATES} for s in STATES},
                "history": [], "states": list(STATES),
                "strategy_hooks": STRATEGY_HOOKS[state],
                "data_source": "macro-only (no historical Nifty)",
                "timestamp": time.time(),
            }
            self._cache, self._cache_ts = result, now
            return result

        closes = df["close"].astype(float).values
        n = len(closes)
        rets = np.diff(np.log(closes))
        ret_20d = float((closes[-1] / closes[-21] - 1) * 100)
        ret_60d = float((closes[-1] / closes[-61] - 1) * 100)
        vol_20d = _annualised_vol(rets[-20:])
        vol_60d = _annualised_vol(rets[-60:])

        state, confidence = _classify_window(
            ret_20d, ret_60d, vol_20d, vol_60d, vix_today, breadth_today, sector_disp,
        )

        history_states = self._walk_history(df, vix_today, breadth_today, sector_disp)
        # Force the last entry to match `state` (rule-set agreement guaranteed,
        # but defensive against proxy-mismatch on the final bar).
        if history_states:
            history_states[-1] = state

        # days_in_state — count back from today while we stay in `state`
        days_in_state = 0
        for s in reversed(history_states):
            if s == state:
                days_in_state += 1
            else:
                break

        transitions = self._empirical_transitions(history_states)

        # Pair history states with their dates for UI sparkline
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()[-len(history_states):]
        else:
            dates = [str(i) for i in range(len(history_states))]
        history = [{"date": d, "state": s} for d, s in zip(dates, history_states)]

        result = {
            "state": state,
            "confidence": confidence,
            "days_in_state": int(days_in_state),
            "signals": {
                "nifty_return_20d": round(ret_20d, 2),
                "nifty_return_60d": round(ret_60d, 2),
                "realized_vol_20d": round(vol_20d, 2),
                "realized_vol_60d": round(vol_60d, 2),
                "india_vix": round(vix_today, 2),
                "breadth_adr": round(breadth_today, 2),
                "sector_dispersion": sector_disp,
            },
            "transition_probs": transitions,
            "history": history,
            "states": list(STATES),
            "strategy_hooks": STRATEGY_HOOKS[state],
            "data_source": f"nifty500 ({n} bars)",
            "timestamp": time.time(),
        }
        self._cache, self._cache_ts = result, now
        return result


_classifier: Optional[RegimeClassifier] = None


def get_regime_classifier() -> RegimeClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RegimeClassifier()
    return _classifier
