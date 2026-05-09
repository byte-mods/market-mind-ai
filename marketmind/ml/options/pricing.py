"""Black-Scholes pricing + Greeks for European-style options.

Notes
-----
- Index options on NSE (NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY) are European-style;
  Black-Scholes is exact for these. Equity options are American — BS is a close
  approximation away from dividends. We surface this via `pricing_model` in
  the analytics result.
- All inputs are decimals (rates, vol). NSE returns IV in percent; convert at
  the boundary via `iv_to_decimal`.
- Risk-free rate default: 6.5 % (RBI repo, FY26). Override per call if needed.
"""
from __future__ import annotations

import math
from typing import Dict, Literal

import numpy as np
from scipy.stats import norm

OptionKind = Literal["CE", "PE"]

DEFAULT_RISK_FREE = 0.065  # 6.5% — RBI repo, FY26
DAYS_PER_YEAR = 365.0


def iv_to_decimal(iv_percent: float) -> float:
    """NSE serves IV as percentage; convert to decimal for BS."""
    if iv_percent is None or iv_percent <= 0:
        return 0.0
    return float(iv_percent) / 100.0


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        # Degenerate inputs — caller handles via intrinsic-only path.
        return float("nan"), float("nan")
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vt
    d2 = d1 - vt
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    kind: OptionKind,
    r: float = DEFAULT_RISK_FREE,
) -> float:
    """Black-Scholes price for a European call ('CE') or put ('PE')."""
    if T <= 0 or sigma <= 0:
        # Expired or zero-vol: intrinsic value only.
        return max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0)
    disc = math.exp(-r * T)
    if kind == "CE":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    kind: OptionKind,
    r: float = DEFAULT_RISK_FREE,
) -> Dict[str, float]:
    """Return delta, gamma, theta (per day), vega (per 1% vol), rho (per 1% rate)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    pdf_d1 = norm.pdf(d1)
    disc = math.exp(-r * T)
    sqrtT = math.sqrt(T)

    # Delta
    if kind == "CE":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    # Gamma — same for call and put
    gamma = pdf_d1 / (S * sigma * sqrtT)

    # Vega — per 1% absolute change in vol (i.e. divide by 100)
    vega = S * pdf_d1 * sqrtT / 100.0

    # Theta — annualised then divided by DAYS_PER_YEAR for "per day"
    theta_annual_common = -(S * pdf_d1 * sigma) / (2.0 * sqrtT)
    if kind == "CE":
        theta_annual = theta_annual_common - r * K * disc * norm.cdf(d2)
    else:
        theta_annual = theta_annual_common + r * K * disc * norm.cdf(-d2)
    theta = theta_annual / DAYS_PER_YEAR

    # Rho — per 1% absolute rate change
    if kind == "CE":
        rho = K * T * disc * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * disc * norm.cdf(-d2) / 100.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def days_to_T(days: float) -> float:
    """Convert calendar days-to-expiry into year fraction."""
    return max(float(days), 0.0) / DAYS_PER_YEAR


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    kind: OptionKind,
    r: float = DEFAULT_RISK_FREE,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Back-solve Black-Scholes implied volatility from a market price.

    Uses bisection in [1e-4, 5.0] (i.e. 0.01% – 500% annualised vol).
    Returns 0.0 when the price violates no-arbitrage bounds (e.g. quote is below
    intrinsic, expired option, or zero LTP) so the caller can flag the strike
    as IV-unavailable instead of poisoning downstream Greeks with a bad number.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    intrinsic = max(S - K, 0.0) if kind == "CE" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return 0.0  # arb-violating quote — refuse rather than mislead.
    lo, hi = 1e-4, 5.0
    p_lo = bs_price(S, K, T, lo, kind, r)
    p_hi = bs_price(S, K, T, hi, kind, r)
    # Price is monotonic in vol — if the target lies outside [p_lo, p_hi] the
    # quote is unreachable under BS at any vol. Clamp to nearest endpoint.
    if market_price <= p_lo:
        return lo
    if market_price >= p_hi:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(S, K, T, mid, kind, r)
        if abs(p_mid - market_price) < tol:
            return mid
        if p_mid < market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
