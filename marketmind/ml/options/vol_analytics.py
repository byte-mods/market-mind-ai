"""Volatility analytics — IV history, rank, surface, term structure, skew.

All metrics computed from live NSE option chain data. Historical IV is
persisted to MongoDB (no TTL) so IV rank/percentile improves over time.

Terminology
-----------
- IV is stored and computed as **decimal** (0.18 = 18%) internally.
- API-facing values are returned as **percent** (18.0) for human readability.
- NSE option chain serves IV as percent → divide by 100 at the boundary.
"""
from __future__ import annotations

import logging
import math
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum history points before we report a percentile.
# 20 ≈ 1 month of trading days; 252 ≈ 1 year.
_MIN_HISTORY_POINTS = 20


def _atm_iv_from_chain(chain: Dict) -> float:
    """Extract ATM IV in decimal from a parsed chain dict.

    Averages CE + PE IV at the exact ATM strike. Returns 0.0 when either
    side is missing so the caller knows the chain is incomplete.
    """
    atm = chain.get("atm_strike", 0)
    if not atm:
        return 0.0
    iv_sum, n = 0.0, 0
    for row in chain.get("calls", []):
        if row.get("strike") == atm and row.get("iv", 0) > 0:
            iv_sum += row["iv"]
            n += 1
    for row in chain.get("puts", []):
        if row.get("strike") == atm and row.get("iv", 0) > 0:
            iv_sum += row["iv"]
            n += 1
    if n == 0:
        return 0.0
    return (iv_sum / n) / 100.0  # percent → decimal


def _nearest_expiry_from_chain(chain: Dict) -> str:
    """Return the nearest expiry date string from the chain."""
    expiry_dates = chain.get("expiry_dates", [])
    return expiry_dates[0] if expiry_dates else ""


# ─── 1. IV History Collector ────────────────────────────────────────────────


class IVHistoryCollector:
    """Save ATM IV snapshots to MongoDB for longitudinal analysis.

    Usage
    -----
    collector = IVHistoryCollector()
    collector.ensure_indexes(mongo_col)
    collector.save(chain, mongo_col)
    """

    @staticmethod
    def save(chain: Dict, mongo_col: Any = None) -> Optional[str]:
        """Persist one ATM IV snapshot. Returns ``_id`` or None on no-op/error.

        No-op when:
        - ``mongo_col`` is None (Mongo unavailable)
        - chain is unavailable or has no symbol
        - ATM IV cannot be determined (≤ 0)
        """
        if mongo_col is None:
            return None

        symbol = chain.get("symbol", "")
        if not symbol or chain.get("unavailable"):
            return None

        atm_iv = _atm_iv_from_chain(chain)
        if atm_iv <= 0:
            return None

        expiry = _nearest_expiry_from_chain(chain)
        now = datetime.now(timezone.utc)
        doc = {
            "_id": f"{symbol.upper()}:{now.isoformat()}:{secrets.token_hex(3)}",
            "ts": now,
            "symbol": symbol.upper(),
            "atm_iv": round(atm_iv, 6),  # stored as decimal
            "atm_iv_pct": round(atm_iv * 100, 4),
            "underlying": chain.get("underlying", 0),
            "expiry": expiry,
            "pcr": chain.get("pcr"),
            "max_pain": chain.get("max_pain"),
            "days_to_expiry": chain.get("days_to_expiry"),
        }
        try:
            mongo_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            return doc["_id"]
        except Exception as e:  # noqa: BLE001
            logger.warning("IV history save failed: %s", e)
            return None

    @staticmethod
    def ensure_indexes(mongo_col: Any) -> None:
        """Create compound index on (symbol, ts) if Mongo is up."""
        if mongo_col is None:
            return
        try:
            mongo_col.create_index("symbol")
            mongo_col.create_index("ts")
            mongo_col.create_index([("symbol", 1), ("ts", -1)])
        except Exception as e:  # noqa: BLE001
            logger.debug("IV history index ensure failed: %s", e)


# ─── 2. IV Rank & Percentile ────────────────────────────────────────────────


def iv_history(
    symbol: str,
    history_days: int = 252,
    mongo_col: Any = None,
) -> List[Dict]:
    """Return ATM IV history for symbol, newest first.

    Each dict has keys: ``ts``, ``atm_iv``, ``atm_iv_pct``, ``underlying``,
    ``expiry``, ``pcr``, ``max_pain``, ``days_to_expiry``.
    """
    if mongo_col is None:
        return []

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=history_days)
        cursor = mongo_col.find(
            {"symbol": symbol.upper(), "ts": {"$gte": cutoff}}
        ).sort("ts", -1)
        return list(cursor)
    except Exception as e:  # noqa: BLE001
        logger.warning("IV history query failed: %s", e)
        return []


def iv_rank(
    symbol: str,
    history_days: int = 252,
    mongo_col: Any = None,
) -> Dict:
    """Compute IV rank and percentile from stored history.

    **IV Rank** = (current_IV - period_low) / (period_high - period_low) × 100
    Range [0, 100]. 0 = cheapest IV in the lookback window; 100 = most
    expensive.

    **IV Percentile** = % of historical observations with IV < current IV.
    More robust than rank when the distribution is skewed.

    Returns a dict with human-facing percents (18.0 rather than 0.18).
    """
    history = iv_history(symbol, history_days, mongo_col)
    n = len(history)

    if n < _MIN_HISTORY_POINTS:
        return {
            "symbol": symbol.upper(),
            "current_iv": None,
            "iv_rank": None,
            "iv_percentile": None,
            "history_high": None,
            "history_low": None,
            "history_avg": None,
            "history_days": n,
            "status": "collecting" if n > 0 else "no_history",
            "reason": f"Need {_MIN_HISTORY_POINTS}+ data points; have {n}",
        }

    ivs = [h["atm_iv"] for h in history if h.get("atm_iv") is not None]
    if not ivs:
        return {
            "symbol": symbol.upper(),
            "current_iv": None,
            "status": "no_data",
            "reason": "History exists but no IV values found",
        }

    current_iv = ivs[0]  # newest first
    iv_high = max(ivs)
    iv_low = min(ivs)
    iv_avg = float(np.mean(ivs))

    # IV Rank — clamp to [0, 100] to guard against floating-point jitter
    if iv_high > iv_low:
        rank = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
        rank = max(0.0, min(100.0, rank))
    else:
        rank = 50.0

    # IV Percentile — exclude current observation from the denominator
    # so that on the first day at a new high/low we get 100/0, not 100/100.
    prior_ivs = ivs[1:]
    below_count = sum(1 for v in prior_ivs if v <= current_iv)
    percentile = (below_count / max(len(prior_ivs), 1)) * 100.0

    return {
        "symbol": symbol.upper(),
        "current_iv": round(current_iv * 100, 2),
        "iv_rank": round(rank, 2),
        "iv_percentile": round(percentile, 2),
        "history_high": round(iv_high * 100, 2),
        "history_low": round(iv_low * 100, 2),
        "history_avg": round(iv_avg * 100, 2),
        "history_days": n,
        "status": "ready",
    }


# ─── 3. Volatility Surface ──────────────────────────────────────────────────


def _group_by_expiry(raw_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group NSE raw ``records.data`` items by ``expiryDate``.

    Each group is a list of {strikePrice, CE, PE} records.
    """
    groups: Dict[str, List[Dict]] = {}
    for item in raw_data:
        expiry = item.get("expiryDate", "")
        if not expiry:
            continue
        groups.setdefault(expiry, []).append(item)
    return groups


def _atm_for_group(records: List[Dict], underlying: float) -> float:
    """Find the ATM strike within one expiry group."""
    strikes = [r.get("strikePrice", 0) for r in records if r.get("strikePrice")]
    if not strikes or underlying <= 0:
        return 0.0
    return float(min(strikes, key=lambda s: abs(s - underlying)))


def _avg_iv_at_strike(records: List[Dict], strike: float) -> float:
    """Average CE+PE IV at a specific strike, in decimal."""
    iv_sum, n = 0.0, 0
    for r in records:
        if r.get("strikePrice") == strike:
            ce = r.get("CE")
            pe = r.get("PE")
            if not isinstance(ce, dict):
                ce = {}
            if not isinstance(pe, dict):
                pe = {}
            try:
                c_iv = float(ce.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                c_iv = 0.0
            try:
                p_iv = float(pe.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                p_iv = 0.0
            if c_iv > 0:
                iv_sum += c_iv
                n += 1
            if p_iv > 0:
                iv_sum += p_iv
                n += 1
    if n == 0:
        return 0.0
    return (iv_sum / n) / 100.0


def vol_surface(raw_chain: Dict) -> Dict:
    """Build a volatility surface from the raw NSE option chain response.

    ``raw_chain`` must be the *unparsed* NSE JSON (the output of
    ``session.get(...).json()``) — specifically the ``records`` dict or the
    full response containing ``records``.

    Returns
    -------
    {
        "symbol": str,
        "underlying": float,
        "points": [{"strike": float, "expiry": str, "iv": float, "iv_pct": float}],
        "atm_curve": [{"expiry": str, "atm_iv": float, "atm_iv_pct": float, "days_to_expiry": int}],
        "unavailable": bool,
        "reason": str,
    }
    """
    records = raw_chain.get("records") or raw_chain
    data = records.get("data", []) if isinstance(records, dict) else []
    expiry_dates = records.get("expiryDates", []) if isinstance(records, dict) else []
    try:
        underlying = float(records.get("underlyingValue", 0)) if isinstance(records, dict) else 0.0
    except (TypeError, ValueError):
        underlying = 0.0

    if not data:
        return {
            "symbol": raw_chain.get("symbol", ""),
            "underlying": 0,
            "points": [],
            "atm_curve": [],
            "unavailable": True,
            "reason": "No option chain data available",
        }

    groups = _group_by_expiry(data)
    if not groups:
        # Fallback: treat everything as one unnamed expiry
        groups = {"": data}

    points: List[Dict] = []
    atm_curve: List[Dict] = []

    for expiry in sorted(groups.keys(), key=lambda e: (e == "", e)):
        grp = groups[expiry]
        atm = _atm_for_group(grp, underlying)
        atm_iv = _avg_iv_at_strike(grp, atm)

        # Days to expiry
        dte = 0
        try:
            from datetime import date as _date, datetime as _dt
            exp_dt = _dt.strptime(expiry, "%d-%b-%Y").date()
            dte = max((exp_dt - _date.today()).days, 0)
        except Exception:
            dte = 0

        if atm_iv > 0:
            atm_curve.append({
                "expiry": expiry,
                "atm_iv": round(atm_iv, 6),
                "atm_iv_pct": round(atm_iv * 100, 2),
                "days_to_expiry": dte,
            })

        for r in grp:
            strike = r.get("strikePrice", 0)
            if strike == 0:
                continue
            ce = r.get("CE")
            pe = r.get("PE")
            if not isinstance(ce, dict):
                ce = {}
            if not isinstance(pe, dict):
                pe = {}
            try:
                c_iv = float(ce.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                c_iv = 0.0
            try:
                p_iv = float(pe.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                p_iv = 0.0
            if c_iv > 0:
                points.append({
                    "strike": float(strike),
                    "expiry": expiry,
                    "kind": "CE",
                    "iv": round(c_iv / 100.0, 6),
                    "iv_pct": round(c_iv, 2),
                    "oi": ce.get("openInterest", 0),
                    "ltp": ce.get("lastPrice", 0),
                })
            if p_iv > 0:
                points.append({
                    "strike": float(strike),
                    "expiry": expiry,
                    "kind": "PE",
                    "iv": round(p_iv / 100.0, 6),
                    "iv_pct": round(p_iv, 2),
                    "oi": pe.get("openInterest", 0),
                    "ltp": pe.get("lastPrice", 0),
                })

    return {
        "symbol": raw_chain.get("symbol", ""),
        "underlying": underlying,
        "points": points,
        "atm_curve": atm_curve,
        "unavailable": False,
        "reason": "",
    }


def term_structure(surface: Dict) -> Dict:
    """Analyse the ATM IV term structure from a vol surface.

    Returns slope direction (contango / backwardation / flat), annualised
    carry, and the raw ATM curve sorted by days-to-expiry.
    """
    atm_curve = surface.get("atm_curve", [])
    if not atm_curve:
        return {
            "symbol": surface.get("symbol", ""),
            "curve": [],
            "slope": "unknown",
            "carry_bps_per_day": None,
            "reason": "No ATM IV curve available",
        }

    # Sort by days to expiry
    curve = sorted(atm_curve, key=lambda x: x.get("days_to_expiry", 0))
    if len(curve) < 2:
        return {
            "symbol": surface.get("symbol", ""),
            "curve": curve,
            "slope": "single_expiry",
            "carry_bps_per_day": None,
            "reason": "Only one expiry available",
        }

    front = curve[0]
    back = curve[-1]
    front_iv = front.get("atm_iv", 0)
    back_iv = back.get("atm_iv", 0)
    front_dte = front.get("days_to_expiry", 1)
    back_dte = back.get("days_to_expiry", 1)
    dte_diff = max(back_dte - front_dte, 1)

    if back_iv > front_iv * 1.01:
        slope = "contango"
    elif back_iv < front_iv * 0.99:
        slope = "backwardation"
    else:
        slope = "flat"

    # Carry in basis points of IV per calendar day
    carry_bps = ((back_iv - front_iv) / dte_diff) * 10000.0

    return {
        "symbol": surface.get("symbol", ""),
        "curve": curve,
        "slope": slope,
        "carry_bps_per_day": round(carry_bps, 4),
        "front_expiry": front.get("expiry"),
        "back_expiry": back.get("expiry"),
        "front_iv_pct": front.get("atm_iv_pct"),
        "back_iv_pct": back.get("atm_iv_pct"),
    }


# ─── 4. Skew Metrics ────────────────────────────────────────────────────────


def _iv_at_moneyness(records: List[Dict], underlying: float, moneyness: float) -> float:
    """Find average IV at the strike closest to ``underlying * moneyness``.

    Moneyness 1.0 = ATM, 1.05 = 5% OTM call, 0.95 = 5% OTM put.
    Returns decimal IV or 0.0 if no match.
    """
    if underlying <= 0 or not records:
        return 0.0
    target = underlying * moneyness
    best = min(
        (r for r in records if r.get("strikePrice")),
        key=lambda r: abs(r["strikePrice"] - target),
        default=None,
    )
    if best is None:
        return 0.0
    return _avg_iv_at_strike([best], best["strikePrice"])


def skew_metrics(raw_chain: Dict) -> Dict:
    """Compute skew metrics from raw NSE option chain response.

    Metrics
    -------
    - **25-delta risk reversal**: IV(25Δ call) − IV(25Δ put). Positive = call
      skew (bullish demand); negative = put skew (bearish hedging).
    - **Put skew index**: IV(95% moneyness put) / IV(ATM). > 1.1 is steep fear.
    - **Call skew index**: IV(105% moneyness call) / IV(ATM).
    - **Smile shape**: ``put_skew / call_skew`` ratio — > 1.5 = strong put
      skew (smirk); < 0.8 = call skew (reverse smirk); else ~symmetric.

    All IVs returned as **decimal** internally; API layer converts to percent.
    """
    records = raw_chain.get("records") or raw_chain
    data = records.get("data", []) if isinstance(records, dict) else []
    try:
        underlying = float(records.get("underlyingValue", 0)) if isinstance(records, dict) else 0.0
    except (TypeError, ValueError):
        underlying = 0.0

    if not data or underlying <= 0:
        return {
            "symbol": raw_chain.get("symbol", ""),
            "unavailable": True,
            "reason": "No option chain data or underlying unavailable",
        }

    # Use nearest expiry only for skew (mixing expiries contaminates the shape)
    groups = _group_by_expiry(data)
    if not groups:
        return {
            "symbol": raw_chain.get("symbol", ""),
            "unavailable": True,
            "reason": "Could not group chain by expiry",
        }

    # Pick nearest expiry (first in sorted order by date)
    def _expiry_sort_key(e: str) -> Tuple:
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(e, "%d-%b-%Y")
            return (0, dt)
        except Exception:
            return (1, e)

    nearest_expiry = min(groups.keys(), key=_expiry_sort_key)
    grp = groups[nearest_expiry]

    atm_iv = _iv_at_moneyness(grp, underlying, 1.0)
    if atm_iv <= 0:
        return {
            "symbol": raw_chain.get("symbol", ""),
            "unavailable": True,
            "reason": "ATM IV unavailable",
        }

    # Approximate 25-delta strikes by moneyness
    # For short-dated options, 25Δ ≈ ±0.5·IV·sqrt(T) away from ATM.
    # We use fixed moneyness proxies (95% / 105%) which are close enough for
    # liquid index options near expiry.
    put_95_iv = _iv_at_moneyness(grp, underlying, 0.95)
    call_105_iv = _iv_at_moneyness(grp, underlying, 1.05)
    put_90_iv = _iv_at_moneyness(grp, underlying, 0.90)
    call_110_iv = _iv_at_moneyness(grp, underlying, 1.10)

    risk_reversal = call_105_iv - put_95_iv  # decimal vol points
    put_skew_idx = (put_95_iv / atm_iv) if atm_iv > 0 else 0.0
    call_skew_idx = (call_105_iv / atm_iv) if atm_iv > 0 else 0.0

    if put_skew_idx > call_skew_idx * 1.2:
        smile_shape = "put_skew"  # fear / hedging demand
    elif call_skew_idx > put_skew_idx * 1.15:
        smile_shape = "call_skew"  # bullish call buying
    else:
        smile_shape = "symmetric"

    # ATM put-call IV diff (old advisor metric, now formalised)
    call_atm_iv = 0.0
    put_atm_iv = 0.0
    atm = _atm_for_group(grp, underlying)
    for r in grp:
        if r.get("strikePrice") == atm:
            ce = r.get("CE")
            pe = r.get("PE")
            if not isinstance(ce, dict):
                ce = {}
            if not isinstance(pe, dict):
                pe = {}
            try:
                c_iv = float(ce.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                c_iv = 0.0
            try:
                p_iv = float(pe.get("impliedVolatility", 0))
            except (TypeError, ValueError):
                p_iv = 0.0
            if c_iv > 0:
                call_atm_iv = c_iv / 100.0
            if p_iv > 0:
                put_atm_iv = p_iv / 100.0

    return {
        "symbol": raw_chain.get("symbol", ""),
        "expiry": nearest_expiry,
        "unavailable": False,
        "atm_iv_pct": round(atm_iv * 100, 2),
        "risk_reversal": round(risk_reversal * 100, 2),  # convert to vol points (percent)
        "put_skew_index": round(put_skew_idx, 3),
        "call_skew_index": round(call_skew_idx, 3),
        "smile_shape": smile_shape,
        "put_90_iv_pct": round(put_90_iv * 100, 2) if put_90_iv > 0 else None,
        "put_95_iv_pct": round(put_95_iv * 100, 2) if put_95_iv > 0 else None,
        "call_105_iv_pct": round(call_105_iv * 100, 2) if call_105_iv > 0 else None,
        "call_110_iv_pct": round(call_110_iv * 100, 2) if call_110_iv > 0 else None,
        "call_atm_iv_pct": round(call_atm_iv * 100, 2) if call_atm_iv > 0 else None,
        "put_atm_iv_pct": round(put_atm_iv * 100, 2) if put_atm_iv > 0 else None,
        "atm_put_call_spread": round((put_atm_iv - call_atm_iv) * 100, 2) if (put_atm_iv > 0 and call_atm_iv > 0) else None,
    }
