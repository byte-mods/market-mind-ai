"""Named option strategies → leg lists, optionally seeded from an option chain.

A `Leg` dict has the shape:
    {action: "BUY"|"SELL", kind: "CE"|"PE", strike: float, premium: float,
     iv: float (decimal), qty: int, expiry_days: int}

`build_default_legs(name, chain, expiry_days, lots, lot_size)` selects strikes
from the chain (ATM ± k) and pulls premium + IV from the matched row.

Same-expiry strategies only in v1; calendar_spread requires `back_expiry_days`
in the caller and is rejected at the API layer if missing.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from .pricing import iv_to_decimal

StrategyName = Literal[
    "covered_call",
    "cash_secured_put",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "calendar_spread",
    "ratio_spread",
]

ALL_STRATEGIES: tuple[StrategyName, ...] = (
    "covered_call",
    "cash_secured_put",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "calendar_spread",
    "ratio_spread",
)


def list_strategies() -> List[str]:
    return list(ALL_STRATEGIES)


def _row_by_strike(rows: List[Dict], strike: float) -> Optional[Dict]:
    if not rows:
        return None
    # Closest available strike — chain may not contain the exact ask.
    return min(rows, key=lambda r: abs(float(r.get("strike", 0)) - float(strike)))


def _ladder(rows: List[Dict], atm: float) -> List[float]:
    """Return sorted unique strikes from chain rows."""
    return sorted({float(r.get("strike", 0)) for r in rows if r.get("strike")})


def _step(strikes: List[float], atm: float) -> float:
    """Estimate strike step (difference between adjacent strikes near ATM)."""
    if len(strikes) < 2:
        return max(atm * 0.01, 1.0)
    diffs = sorted({round(strikes[i + 1] - strikes[i], 4) for i in range(len(strikes) - 1)})
    # Most common nonzero diff is the step.
    nz = [d for d in diffs if d > 0]
    return nz[len(nz) // 2] if nz else max(atm * 0.01, 1.0)


def _leg(
    action: str,
    kind: str,
    strike: float,
    rows: List[Dict],
    qty: int,
    expiry_days: int,
) -> Dict:
    row = _row_by_strike(rows, strike) or {}
    return {
        "action": action,
        "kind": kind,
        "strike": float(row.get("strike", strike)),
        "premium": float(row.get("ltp", 0.0) or 0.0),
        "iv": iv_to_decimal(float(row.get("iv", 0.0) or 0.0)),
        "qty": int(qty),
        "expiry_days": int(expiry_days),
    }


def build_default_legs(
    name: StrategyName,
    chain: Dict,
    expiry_days: int,
    lots: int = 1,
    lot_size: int = 1,
) -> List[Dict]:
    """Construct a default leg list for `name` from `chain`.

    `chain` is the dict returned by OptionsFetcher.get_option_chain — needs
    `calls`, `puts`, `atm_strike`, `underlying`. Quantity per leg = lots * lot_size.

    Raises ValueError if `name` is unknown or the chain has no strikes.
    """
    if name not in ALL_STRATEGIES:
        raise ValueError(f"unknown strategy: {name}")

    calls = chain.get("calls") or []
    puts = chain.get("puts") or []
    atm = float(chain.get("atm_strike") or chain.get("underlying") or 0)
    if atm <= 0:
        raise ValueError("chain has no ATM strike — cannot seed default legs")

    strikes = _ladder(calls or puts, atm)
    step = _step(strikes, atm)
    qty = max(int(lots) * int(lot_size), 1)

    if name == "covered_call":
        # Long stock implicit; sell 1 OTM call.
        otm_call = atm + step
        return [_leg("SELL", "CE", otm_call, calls, qty, expiry_days)]

    if name == "cash_secured_put":
        otm_put = atm - step
        return [_leg("SELL", "PE", otm_put, puts, qty, expiry_days)]

    if name == "bull_call_spread":
        return [
            _leg("BUY", "CE", atm, calls, qty, expiry_days),
            _leg("SELL", "CE", atm + 2 * step, calls, qty, expiry_days),
        ]

    if name == "bear_put_spread":
        return [
            _leg("BUY", "PE", atm, puts, qty, expiry_days),
            _leg("SELL", "PE", atm - 2 * step, puts, qty, expiry_days),
        ]

    if name == "straddle":
        return [
            _leg("BUY", "CE", atm, calls, qty, expiry_days),
            _leg("BUY", "PE", atm, puts, qty, expiry_days),
        ]

    if name == "strangle":
        return [
            _leg("BUY", "CE", atm + step, calls, qty, expiry_days),
            _leg("BUY", "PE", atm - step, puts, qty, expiry_days),
        ]

    if name == "iron_condor":
        return [
            _leg("SELL", "PE", atm - step, puts, qty, expiry_days),
            _leg("BUY", "PE", atm - 2 * step, puts, qty, expiry_days),
            _leg("SELL", "CE", atm + step, calls, qty, expiry_days),
            _leg("BUY", "CE", atm + 2 * step, calls, qty, expiry_days),
        ]

    if name == "calendar_spread":
        # Same-strike, two expiries. v1 returns two legs at same expiry — caller
        # must supply back_expiry_days externally and replace the second leg's
        # expiry. API layer enforces presence.
        return [
            _leg("SELL", "CE", atm, calls, qty, expiry_days),
            _leg("BUY", "CE", atm, calls, qty, expiry_days),
        ]

    if name == "ratio_spread":
        # 1x2 call ratio: long 1 ATM call, short 2 OTM calls.
        return [
            _leg("BUY", "CE", atm, calls, qty, expiry_days),
            _leg("SELL", "CE", atm + 2 * step, calls, 2 * qty, expiry_days),
        ]

    # Unreachable — guarded by ALL_STRATEGIES check above.
    raise ValueError(f"unhandled strategy: {name}")
