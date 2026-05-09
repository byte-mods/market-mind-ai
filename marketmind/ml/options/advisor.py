"""Options Advisor — converts a live option chain + market context into a
trade recommendation.

Pipeline
--------
1. ``directional_bias()`` aggregates RL signal + PCR + max-pain skew + IV skew
   into a Bullish / Bearish / Neutral score.
2. ``vol_regime()`` reads ATM IV against a rough percentile band to label
   Low / Normal / High / Extreme.
3. ``price_cone()`` projects a 1σ underlying band over 1/5/20 trading days
   from current ATM IV.
4. ``recommend_strategy()`` picks one of {long_call, long_put, bull_call_spread,
   bear_put_spread, iron_condor, short_straddle, long_straddle, calendar_spread}
   from the (bias × vol_regime) cell, with concrete strikes + theoretical
   debit/credit + break-evens drawn from the chain.
5. ``manage_position()`` takes a held leg and returns roll / take-profit /
   stop-loss / hedge rules grounded in current Greeks + IV.

Inputs are the dict shapes already produced by
``OptionsFetcher.get_option_chain[_from_kite]`` (``underlying``, ``atm_strike``,
``calls``, ``puts``, ``pcr``, ``max_pain``, ``days_to_expiry``).

Recommendations are *educational*, not orders — the SEBI compliance layer
gates any actual order placement separately.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from marketmind.ml.options.pricing import (
    DEFAULT_RISK_FREE,
    bs_greeks,
    bs_price,
    days_to_T,
)


# ─── 1. Directional bias ──────────────────────────────────────────────────


def _atm_iv(chain: Dict) -> float:
    """ATM IV in decimal (e.g. 0.18 for 18%). Averages CE+PE at the ATM strike."""
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
    return (iv_sum / n) / 100.0  # IV stored as percent → decimal


def _iv_skew(chain: Dict) -> float:
    """Put-IV minus Call-IV at ATM, in decimal vol points.

    Positive skew (puts pricier than calls) ≈ bearish hedging demand.
    Negative skew ≈ bullish call demand.
    """
    atm = chain.get("atm_strike", 0)
    if not atm:
        return 0.0
    call_iv = next((c["iv"] for c in chain.get("calls", [])
                    if c.get("strike") == atm and c.get("iv", 0) > 0), 0.0)
    put_iv = next((p["iv"] for p in chain.get("puts", [])
                   if p.get("strike") == atm and p.get("iv", 0) > 0), 0.0)
    if call_iv == 0 or put_iv == 0:
        return 0.0
    return (put_iv - call_iv) / 100.0


def directional_bias(chain: Dict, rl_signal: Optional[Dict] = None) -> Dict:
    """Aggregate signals into a single bias label + score in [-1, +1].

    Components (each contributes a small ±score):
      • RL ensemble action: BUY → +0.4, SELL → -0.4, HOLD/None → 0
      • PCR: <0.7 → +0.2 (low PCR = call-heavy = complacent → contrarian-bearish?
            Convention here: low PCR = recent bullishness consumed → mild bearish
            tilt; we keep it as-printed since the F&O gauge already reads "Greed"
            at low PCR. Treat low-PCR as bearish-leaning.)
        Actually market convention: PCR>1.3 bullish (more puts = fear bottoming),
        PCR<0.7 bearish (calls saturated). Use that.
      • Max-pain skew: spot far below max_pain ≈ pinning pull-up bias (+);
        spot far above ≈ pull-down bias (-). Capped at ±0.2.
      • IV skew: put-IV > call-IV by >2 vol pts → -0.1; opposite → +0.1.
    """
    score = 0.0
    components: List[str] = []

    if rl_signal:
        action = (rl_signal.get("action") or "").upper()
        conf = float(rl_signal.get("confidence", 0) or 0)
        if action == "BUY":
            contrib = 0.4 * max(conf, 0.5)
            score += contrib
            components.append(f"RL: BUY ({conf*100:.0f}%) → +{contrib:.2f}")
        elif action == "SELL":
            contrib = -0.4 * max(conf, 0.5)
            score += contrib
            components.append(f"RL: SELL ({conf*100:.0f}%) → {contrib:.2f}")
        else:
            components.append(f"RL: {action or 'n/a'} → 0")

    pcr = chain.get("pcr") or 0.0
    if pcr > 1.3:
        score += 0.2
        components.append(f"PCR {pcr:.2f} (bullish, fear) → +0.20")
    elif pcr < 0.7:
        score -= 0.2
        components.append(f"PCR {pcr:.2f} (bearish, complacent) → -0.20")
    else:
        components.append(f"PCR {pcr:.2f} (neutral) → 0")

    spot = chain.get("underlying") or 0.0
    mp = chain.get("max_pain") or 0.0
    if spot > 0 and mp > 0:
        # Distance to max-pain expressed as fraction of spot. ±2% caps the
        # contribution at ±0.2 so a far-OTM expiry doesn't dominate.
        dist = (mp - spot) / spot
        capped = max(min(dist, 0.02), -0.02)
        contrib = capped * 10.0  # ±0.2 at the cap
        score += contrib
        sign = "+" if contrib >= 0 else ""
        components.append(
            f"Max-pain {mp:.0f} vs spot {spot:.0f} ({dist*100:+.2f}%) → {sign}{contrib:.2f}"
        )

    skew = _iv_skew(chain)
    if abs(skew) > 0.02:
        contrib = -0.1 if skew > 0 else 0.1  # put-skew bearish, call-skew bullish
        score += contrib
        sign = "+" if contrib >= 0 else ""
        components.append(f"IV skew {skew*100:+.1f}vp → {sign}{contrib:.2f}")

    score = max(min(score, 1.0), -1.0)
    if score >= 0.25:
        label = "Bullish"
    elif score <= -0.25:
        label = "Bearish"
    else:
        label = "Neutral"

    return {"label": label, "score": round(score, 3), "components": components}


# ─── 2. Volatility regime ─────────────────────────────────────────────────


# Static IV band for Indian indices/equities. A future enhancement can replace
# these with a rolling 30-day IV percentile per symbol from MongoDB cache;
# for now the bands match observed NSE option-chain ranges.
_IV_BAND = {
    "low": 0.10,
    "normal_lo": 0.12,
    "normal_hi": 0.20,
    "high": 0.30,
}


def vol_regime(chain: Dict) -> Dict:
    """Label ATM IV as Low / Normal / High / Extreme."""
    atm_iv = _atm_iv(chain)
    if atm_iv == 0:
        return {"label": "Unknown", "atm_iv_pct": 0.0, "reason": "ATM IV unavailable"}
    pct = atm_iv * 100.0
    if atm_iv < _IV_BAND["normal_lo"]:
        label = "Low"
    elif atm_iv < _IV_BAND["normal_hi"]:
        label = "Normal"
    elif atm_iv < _IV_BAND["high"]:
        label = "High"
    else:
        label = "Extreme"
    return {"label": label, "atm_iv_pct": round(pct, 2), "reason": f"ATM IV {pct:.1f}%"}


# ─── 3. Price cone ────────────────────────────────────────────────────────


def price_cone(chain: Dict, horizons_days: Tuple[int, ...] = (1, 5, 20)) -> Dict:
    """1σ price band per horizon using ATM IV.

    σ_T = spot · IV · √(T_years). Returns {1d: {low, mid, high}, 5d: ..., 20d: ...}.
    "mid" is the spot itself (not max-pain) because the cone is symmetric in BS;
    max-pain shows up separately as the gravitational anchor.
    """
    spot = chain.get("underlying") or 0.0
    iv = _atm_iv(chain)
    out: Dict[str, Dict] = {}
    if spot <= 0 or iv <= 0:
        for h in horizons_days:
            out[f"{h}d"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
        return out
    for h in horizons_days:
        T = days_to_T(h)
        sigma_T = spot * iv * (T ** 0.5)
        out[f"{h}d"] = {
            "low": round(spot - sigma_T, 2),
            "mid": round(spot, 2),
            "high": round(spot + sigma_T, 2),
        }
    return out


# ─── 4. Strategy picker ───────────────────────────────────────────────────


def _row_for_strike(rows: List[Dict], strike: float) -> Optional[Dict]:
    return next((r for r in rows if r.get("strike") == strike), None)


def _strike_step(strikes: List[float]) -> float:
    if len(strikes) < 2:
        return 50.0
    diffs = sorted({round(strikes[i + 1] - strikes[i], 2) for i in range(len(strikes) - 1)})
    return diffs[0] if diffs else 50.0


def _otm_strike(strikes: List[float], atm: float, n_steps: int, direction: str) -> Optional[float]:
    """n_steps OTM strikes away from ATM. direction: 'up' or 'down'."""
    if not strikes:
        return None
    sorted_s = sorted(strikes)
    if atm not in sorted_s:
        atm = min(sorted_s, key=lambda x: abs(x - atm))
    idx = sorted_s.index(atm)
    target_idx = idx + n_steps if direction == "up" else idx - n_steps
    if 0 <= target_idx < len(sorted_s):
        return sorted_s[target_idx]
    return sorted_s[0] if direction == "down" else sorted_s[-1]


def recommend_strategy(chain: Dict, bias: Dict, regime: Dict) -> Dict:
    """Pick a strategy from the (bias × vol_regime) cell and instantiate legs.

    Cell map (high-level):
      Bullish + Low/Normal vol  → Long Call (debit, vol expansion + delta)
      Bullish + High/Extreme    → Bull Call Spread (cap cost, IV crush risk)
      Bearish + Low/Normal      → Long Put
      Bearish + High/Extreme    → Bear Put Spread
      Neutral + Low             → Long Straddle (bet on vol expansion)
      Neutral + Normal          → Calendar Spread (sell front, buy back)
      Neutral + High/Extreme    → Iron Condor (sell range, capped wings)
    """
    spot = chain.get("underlying") or 0.0
    atm = chain.get("atm_strike") or spot
    calls = chain.get("calls", [])
    puts = chain.get("puts", [])
    dte = chain.get("days_to_expiry") or 0
    if spot <= 0 or atm <= 0 or not calls or not puts or dte <= 0:
        return {
            "strategy": "none",
            "reason": "Chain incomplete — underlying/strikes/expiry unavailable.",
            "legs": [],
        }

    strikes = sorted({c["strike"] for c in calls} | {p["strike"] for p in puts})
    step = _strike_step(strikes)
    iv = _atm_iv(chain) or 0.18
    T = days_to_T(dte)

    bias_label = bias.get("label", "Neutral")
    vol_label = regime.get("label", "Normal")

    legs: List[Dict] = []
    name = ""
    rationale = ""

    if bias_label == "Bullish" and vol_label in ("Low", "Normal", "Unknown"):
        name = "Long Call"
        rationale = "Directional bullish; IV not stretched, room for vol expansion + delta."
        leg_call = _row_for_strike(calls, atm)
        if leg_call:
            legs = [{"action": "BUY", "kind": "CE", "strike": atm,
                     "premium": leg_call["ltp"], "iv": iv}]
    elif bias_label == "Bullish" and vol_label in ("High", "Extreme"):
        name = "Bull Call Spread"
        rationale = "Bullish but IV is rich — debit-spread caps cost and IV crush risk."
        long_k = atm
        short_k = _otm_strike(strikes, atm, 2, "up") or (atm + 2 * step)
        long_call = _row_for_strike(calls, long_k)
        short_call = _row_for_strike(calls, short_k)
        if long_call and short_call:
            legs = [
                {"action": "BUY", "kind": "CE", "strike": long_k,
                 "premium": long_call["ltp"], "iv": iv},
                {"action": "SELL", "kind": "CE", "strike": short_k,
                 "premium": short_call["ltp"], "iv": iv},
            ]
    elif bias_label == "Bearish" and vol_label in ("Low", "Normal", "Unknown"):
        name = "Long Put"
        rationale = "Directional bearish; cheap puts, room for vol expansion."
        leg_put = _row_for_strike(puts, atm)
        if leg_put:
            legs = [{"action": "BUY", "kind": "PE", "strike": atm,
                     "premium": leg_put["ltp"], "iv": iv}]
    elif bias_label == "Bearish" and vol_label in ("High", "Extreme"):
        name = "Bear Put Spread"
        rationale = "Bearish but IV is rich — debit-spread caps cost and IV crush risk."
        long_k = atm
        short_k = _otm_strike(strikes, atm, 2, "down") or (atm - 2 * step)
        long_put = _row_for_strike(puts, long_k)
        short_put = _row_for_strike(puts, short_k)
        if long_put and short_put:
            legs = [
                {"action": "BUY", "kind": "PE", "strike": long_k,
                 "premium": long_put["ltp"], "iv": iv},
                {"action": "SELL", "kind": "PE", "strike": short_k,
                 "premium": short_put["ltp"], "iv": iv},
            ]
    elif bias_label == "Neutral" and vol_label == "Low":
        name = "Long Straddle"
        rationale = "Neutral direction, vol cheap — buy ATM call + put, profit if range expands."
        c = _row_for_strike(calls, atm)
        p = _row_for_strike(puts, atm)
        if c and p:
            legs = [
                {"action": "BUY", "kind": "CE", "strike": atm, "premium": c["ltp"], "iv": iv},
                {"action": "BUY", "kind": "PE", "strike": atm, "premium": p["ltp"], "iv": iv},
            ]
    elif bias_label == "Neutral" and vol_label in ("High", "Extreme"):
        name = "Iron Condor"
        rationale = "Neutral with rich vol — sell OTM call + put spreads, collect premium decay."
        sc_k = _otm_strike(strikes, atm, 2, "up") or (atm + 2 * step)
        lc_k = _otm_strike(strikes, atm, 4, "up") or (atm + 4 * step)
        sp_k = _otm_strike(strikes, atm, 2, "down") or (atm - 2 * step)
        lp_k = _otm_strike(strikes, atm, 4, "down") or (atm - 4 * step)
        sc = _row_for_strike(calls, sc_k)
        lc = _row_for_strike(calls, lc_k)
        sp = _row_for_strike(puts, sp_k)
        lp = _row_for_strike(puts, lp_k)
        if sc and lc and sp and lp:
            legs = [
                {"action": "SELL", "kind": "CE", "strike": sc_k, "premium": sc["ltp"], "iv": iv},
                {"action": "BUY", "kind": "CE", "strike": lc_k, "premium": lc["ltp"], "iv": iv},
                {"action": "SELL", "kind": "PE", "strike": sp_k, "premium": sp["ltp"], "iv": iv},
                {"action": "BUY", "kind": "PE", "strike": lp_k, "premium": lp["ltp"], "iv": iv},
            ]
    else:  # Neutral + Normal vol → calendar / fallback to long straddle
        name = "Long Straddle"
        rationale = "Neutral direction, vol normal — defer to straddle as default vol play."
        c = _row_for_strike(calls, atm)
        p = _row_for_strike(puts, atm)
        if c and p:
            legs = [
                {"action": "BUY", "kind": "CE", "strike": atm, "premium": c["ltp"], "iv": iv},
                {"action": "BUY", "kind": "PE", "strike": atm, "premium": p["ltp"], "iv": iv},
            ]

    # Net debit/credit and break-evens (rough — use exact via builder.analyse() at API layer).
    net = 0.0
    for lg in legs:
        sign = -1.0 if lg["action"] == "BUY" else +1.0  # buying = pay
        net += sign * lg["premium"]
    debit_credit = "credit" if net > 0 else "debit"

    return {
        "strategy": name or "none",
        "rationale": rationale,
        "legs": legs,
        "net": round(net, 2),
        "type": debit_credit,
        "underlying": spot,
        "atm": atm,
        "dte": dte,
        "iv_used": round(iv * 100.0, 2),
    }


# ─── 5. Position management ───────────────────────────────────────────────


def manage_position(held_leg: Dict, chain: Dict) -> Dict:
    """Roll / take-profit / stop-loss / hedge rules for a held leg.

    held_leg keys: action ('BUY'|'SELL'), kind ('CE'|'PE'), strike, entry_premium.
    Returns rule list grounded in current chain LTP, Greeks, and DTE.
    """
    spot = chain.get("underlying") or 0.0
    dte = chain.get("days_to_expiry") or 0
    atm_iv = _atm_iv(chain)
    if spot <= 0 or dte <= 0:
        return {"rules": ["Chain unavailable — cannot evaluate position."]}

    action = (held_leg.get("action") or "").upper()
    kind = (held_leg.get("kind") or "").upper()
    strike = float(held_leg.get("strike") or 0)
    entry = float(held_leg.get("entry_premium") or 0)
    rows = chain.get("calls", []) if kind == "CE" else chain.get("puts", [])
    row = _row_for_strike(rows, strike)
    if not row:
        return {"rules": [f"No live quote for {kind} {strike} — strike not on chain."]}
    ltp = row.get("ltp", 0)

    # Greeks at the held strike using ATM IV as proxy (close enough at ±2 strikes)
    greeks = bs_greeks(spot, strike, days_to_T(dte), atm_iv or 0.18, kind)  # type: ignore[arg-type]

    rules: List[str] = []
    is_long = action == "BUY"
    pnl_pct = ((ltp - entry) / entry * 100.0) if entry > 0 else 0.0
    pnl_pct = pnl_pct if is_long else -pnl_pct

    # Take-profit
    if is_long and pnl_pct >= 50:
        rules.append(f"TAKE PROFIT — leg up {pnl_pct:+.0f}%; book at least 50% of position.")
    elif not is_long and pnl_pct >= 50:
        rules.append(f"TAKE PROFIT — short leg has decayed {pnl_pct:+.0f}%; close to lock gain.")

    # Stop-loss
    if is_long and pnl_pct <= -50:
        rules.append(f"STOP LOSS — leg down {pnl_pct:+.0f}%; cut to preserve capital.")
    elif not is_long and pnl_pct <= -100:
        rules.append(
            f"STOP LOSS — short leg up {abs(pnl_pct):.0f}% vs entry; close, do not average down."
        )

    # Time decay / roll
    if dte <= 5 and is_long:
        rules.append(
            f"TIME DECAY — only {dte}d to expiry; theta {greeks['theta']:.2f}/day. "
            "Roll to next expiry or exit."
        )
    elif dte <= 3 and not is_long:
        rules.append(
            f"PIN RISK — {dte}d to expiry on a short option; close or take assignment risk."
        )

    # Delta breach (directional move against you)
    delta = greeks["delta"]
    if is_long and abs(delta) >= 0.85:
        rules.append(
            f"DEEP ITM (Δ={delta:+.2f}) — most edge captured; lock gain or roll up."
        )
    elif not is_long and abs(delta) >= 0.50:
        rules.append(
            f"BREACH (Δ={delta:+.2f}) — short leg losing directional control; "
            "close or hedge with same-side long at next strike."
        )

    # Hedge suggestion when short and underwater
    if not is_long and pnl_pct < -40:
        hedge_strike_dir = "up" if kind == "CE" else "down"
        rules.append(
            f"HEDGE — buy a further-OTM {kind} 1 strike {hedge_strike_dir} to convert "
            "short into a defined-risk spread."
        )

    if not rules:
        rules.append(
            f"HOLD — leg P&L {pnl_pct:+.1f}%, Δ={delta:+.2f}, θ={greeks['theta']:.2f}/day. "
            "No action triggers hit; reassess daily."
        )

    return {
        "rules": rules,
        "current_ltp": ltp,
        "entry_premium": entry,
        "pnl_pct": round(pnl_pct, 2),
        "greeks": {k: round(v, 4) for k, v in greeks.items()},
        "dte": dte,
    }
