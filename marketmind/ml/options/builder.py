"""Strategy analytics — legs → payoff curve, max P/L, break-evens, net Greeks.

Pure-numpy vectorised payoff. All payoffs computed at expiry (intrinsic), with
Greeks evaluated at the per-leg IV / time-to-expiry.

For multi-expiry strategies (calendar_spread), payoff at the *near* expiry
under-states the back-month's residual extrinsic value — flagged in the result.
"""
from __future__ import annotations

from time import time
from typing import Dict, List, Optional

import numpy as np

from .pricing import DEFAULT_RISK_FREE, bs_greeks, bs_price, days_to_T

PAYOFF_POINTS = 200
PAYOFF_RANGE_PCT = 0.30  # ±30% around underlying


def _validate_legs(legs: List[Dict]) -> None:
    if not legs:
        raise ValueError("at least one leg required")
    for i, leg in enumerate(legs):
        if leg.get("action") not in ("BUY", "SELL"):
            raise ValueError(f"leg[{i}].action must be BUY or SELL")
        if leg.get("kind") not in ("CE", "PE"):
            raise ValueError(f"leg[{i}].kind must be CE or PE")
        if float(leg.get("strike", 0)) <= 0:
            raise ValueError(f"leg[{i}].strike must be > 0")
        if int(leg.get("qty", 0)) <= 0:
            raise ValueError(f"leg[{i}].qty must be > 0")
        if float(leg.get("expiry_days", 0)) < 0:
            raise ValueError(f"leg[{i}].expiry_days must be >= 0")


def _intrinsic(underlying: np.ndarray, strike: float, kind: str) -> np.ndarray:
    if kind == "CE":
        return np.maximum(underlying - strike, 0.0)
    return np.maximum(strike - underlying, 0.0)


def _leg_payoff(leg: Dict, underlying: np.ndarray) -> np.ndarray:
    intrinsic = _intrinsic(underlying, float(leg["strike"]), leg["kind"])
    sign = 1.0 if leg["action"] == "BUY" else -1.0
    qty = int(leg["qty"])
    premium = float(leg.get("premium", 0.0))
    # Long: pnl = (intrinsic - premium) * qty
    # Short: pnl = (premium - intrinsic) * qty
    return sign * (intrinsic - premium) * qty


def _find_break_evens(underlying: np.ndarray, pnl: np.ndarray) -> List[float]:
    """Linear-interpolated zero crossings of the payoff curve."""
    bes: List[float] = []
    for i in range(len(pnl) - 1):
        a, b = pnl[i], pnl[i + 1]
        if a == 0:
            bes.append(float(underlying[i]))
        elif (a < 0 < b) or (a > 0 > b):
            # Linear interpolation between (u_i, a) and (u_{i+1}, b).
            t = a / (a - b)
            bes.append(float(underlying[i] + t * (underlying[i + 1] - underlying[i])))
    # Dedup with tolerance — tiny numerical jitters at sampling boundaries.
    out: List[float] = []
    for be in bes:
        if not out or abs(be - out[-1]) > 1e-3:
            out.append(round(be, 4))
    return out


def _net_greeks(legs: List[Dict], underlying: float, r: float) -> Dict[str, float]:
    """Sum signed Greeks across legs at the underlying's current price."""
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        T = days_to_T(leg["expiry_days"])
        sigma = float(leg.get("iv", 0.0))
        if sigma <= 0 or T <= 0:
            continue
        g = bs_greeks(underlying, float(leg["strike"]), T, sigma, leg["kind"], r=r)
        sign = 1.0 if leg["action"] == "BUY" else -1.0
        qty = int(leg["qty"])
        for k in totals:
            totals[k] += sign * qty * g[k]
    return {k: round(v, 6) for k, v in totals.items()}


def _theoretical_value(legs: List[Dict], underlying: float, r: float) -> float:
    """Net BS value of the strategy now (signed sum across legs)."""
    val = 0.0
    for leg in legs:
        T = days_to_T(leg["expiry_days"])
        sigma = float(leg.get("iv", 0.0))
        sign = 1.0 if leg["action"] == "BUY" else -1.0
        qty = int(leg["qty"])
        if sigma <= 0 or T <= 0:
            # Use intrinsic at the expiry boundary.
            intrinsic = max(underlying - leg["strike"], 0.0) if leg["kind"] == "CE" else max(leg["strike"] - underlying, 0.0)
            val += sign * qty * intrinsic
        else:
            val += sign * qty * bs_price(underlying, float(leg["strike"]), T, sigma, leg["kind"], r=r)
    return val


def _net_premium(legs: List[Dict]) -> float:
    """Net premium paid (positive) or received (negative) at entry."""
    net = 0.0
    for leg in legs:
        sign = 1.0 if leg["action"] == "BUY" else -1.0
        net += sign * float(leg.get("premium", 0.0)) * int(leg["qty"])
    return net


def analyse(
    legs: List[Dict],
    underlying: float,
    strategy_name: Optional[str] = None,
    risk_free: float = DEFAULT_RISK_FREE,
    multi_expiry: bool = False,
) -> Dict:
    """Compute payoff analytics for a leg list.

    Returns: {strategy, legs, payoff: [{underlying, pnl}], max_profit, max_loss,
              break_evens, net_greeks, net_premium, theoretical_value,
              margin_proxy, pricing_model, generated_at, notes}
    """
    _validate_legs(legs)
    if underlying <= 0:
        raise ValueError("underlying must be > 0")

    lo = underlying * (1.0 - PAYOFF_RANGE_PCT)
    hi = underlying * (1.0 + PAYOFF_RANGE_PCT)
    grid = np.linspace(lo, hi, PAYOFF_POINTS)

    pnl = np.zeros_like(grid)
    for leg in legs:
        pnl += _leg_payoff(leg, grid)

    max_profit = float(np.max(pnl))
    max_loss = float(np.min(pnl))
    break_evens = _find_break_evens(grid, pnl)
    net_greeks = _net_greeks(legs, underlying, risk_free)
    theoretical = _theoretical_value(legs, underlying, risk_free)
    net_premium = _net_premium(legs)

    # Conservative margin proxy: |max_loss| * 1.2. Real SPAN/exposure
    # margin requires regulator-published parameters; flagged in result.
    margin_proxy = round(abs(max_loss) * 1.2, 2) if max_loss < 0 else 0.0

    notes: List[str] = []
    if multi_expiry:
        notes.append("payoff shown at near expiry; back-month extrinsic ignored")
    if not break_evens:
        notes.append("no break-even within ±30% of underlying")

    return {
        "strategy": strategy_name,
        "legs": legs,
        "payoff": [{"underlying": float(u), "pnl": float(p)} for u, p in zip(grid, pnl)],
        "max_profit": round(max_profit, 4),
        "max_loss": round(max_loss, 4),
        "break_evens": break_evens,
        "net_greeks": net_greeks,
        "net_premium": round(net_premium, 4),
        "theoretical_value": round(theoretical, 4),
        "margin_proxy": margin_proxy,
        "margin_is_proxy": True,
        "pricing_model": "black_scholes_european",
        "generated_at": time(),
        "notes": notes,
    }
