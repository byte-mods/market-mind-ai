"""Strategy template tests — leg counts, sign convention, strike ordering."""
from __future__ import annotations

import pytest

from marketmind.ml.options.strategies import (
    ALL_STRATEGIES,
    build_default_legs,
    list_strategies,
)


# ─── synthetic chain helper ───────────────────────────────────────────────


def _chain(atm: float = 100.0, step: float = 5.0, vol: float = 20.0) -> dict:
    """Five-strike chain: atm-2*step .. atm+2*step."""
    strikes = [atm + step * k for k in (-2, -1, 0, 1, 2)]
    calls = [
        {"strike": s, "ltp": max(atm - s, 0) + 2.0, "iv": vol, "oi": 1000}
        for s in strikes
    ]
    puts = [
        {"strike": s, "ltp": max(s - atm, 0) + 2.0, "iv": vol, "oi": 1000}
        for s in strikes
    ]
    return {
        "symbol": "TEST",
        "underlying": atm,
        "atm_strike": atm,
        "calls": calls,
        "puts": puts,
        "expiry_dates": ["30-Apr-2026"],
    }


# ─── registry ─────────────────────────────────────────────────────────────


def test_strategies_list_strategies_returns_nine_names():
    names = list_strategies()
    assert len(names) == 9
    assert set(names) == set(ALL_STRATEGIES)


def test_strategies_unknown_name_raises_valueerror():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_default_legs("not_a_strategy", _chain(), expiry_days=30)  # type: ignore[arg-type]


def test_strategies_chain_without_atm_raises_valueerror():
    with pytest.raises(ValueError, match="ATM strike"):
        build_default_legs(
            "straddle",
            {"calls": [], "puts": [], "atm_strike": 0, "underlying": 0},
            expiry_days=30,
        )


# ─── per-strategy leg shape ───────────────────────────────────────────────


def test_strategies_covered_call_sells_one_otm_call():
    legs = build_default_legs("covered_call", _chain(), expiry_days=30)
    assert len(legs) == 1
    assert legs[0]["action"] == "SELL"
    assert legs[0]["kind"] == "CE"
    assert legs[0]["strike"] > 100.0


def test_strategies_cash_secured_put_sells_one_otm_put():
    legs = build_default_legs("cash_secured_put", _chain(), expiry_days=30)
    assert len(legs) == 1
    assert legs[0]["action"] == "SELL"
    assert legs[0]["kind"] == "PE"
    assert legs[0]["strike"] < 100.0


def test_strategies_bull_call_spread_buys_atm_sells_otm():
    legs = build_default_legs("bull_call_spread", _chain(), expiry_days=30)
    assert len(legs) == 2
    buy, sell = legs[0], legs[1]
    assert buy["action"] == "BUY" and buy["kind"] == "CE"
    assert sell["action"] == "SELL" and sell["kind"] == "CE"
    assert sell["strike"] > buy["strike"]


def test_strategies_bear_put_spread_buys_atm_sells_otm_put():
    legs = build_default_legs("bear_put_spread", _chain(), expiry_days=30)
    assert len(legs) == 2
    buy, sell = legs[0], legs[1]
    assert buy["action"] == "BUY" and buy["kind"] == "PE"
    assert sell["action"] == "SELL" and sell["kind"] == "PE"
    assert sell["strike"] < buy["strike"]


def test_strategies_straddle_buys_atm_call_and_atm_put():
    legs = build_default_legs("straddle", _chain(), expiry_days=30)
    assert len(legs) == 2
    kinds = sorted((l["kind"] for l in legs))
    assert kinds == ["CE", "PE"]
    assert all(l["action"] == "BUY" for l in legs)
    assert legs[0]["strike"] == legs[1]["strike"] == 100.0


def test_strategies_strangle_uses_split_strikes():
    legs = build_default_legs("strangle", _chain(), expiry_days=30)
    assert len(legs) == 2
    assert all(l["action"] == "BUY" for l in legs)
    call = next(l for l in legs if l["kind"] == "CE")
    put = next(l for l in legs if l["kind"] == "PE")
    assert call["strike"] > put["strike"]


def test_strategies_iron_condor_has_four_legs_with_wings():
    legs = build_default_legs("iron_condor", _chain(), expiry_days=30)
    assert len(legs) == 4
    actions = [l["action"] for l in legs]
    assert actions.count("SELL") == 2 and actions.count("BUY") == 2
    # Long wings sit further from ATM than short body legs.
    short_put = next(l for l in legs if l["kind"] == "PE" and l["action"] == "SELL")
    long_put = next(l for l in legs if l["kind"] == "PE" and l["action"] == "BUY")
    assert long_put["strike"] < short_put["strike"]
    short_call = next(l for l in legs if l["kind"] == "CE" and l["action"] == "SELL")
    long_call = next(l for l in legs if l["kind"] == "CE" and l["action"] == "BUY")
    assert long_call["strike"] > short_call["strike"]


def test_strategies_calendar_spread_uses_same_strike_two_legs():
    legs = build_default_legs("calendar_spread", _chain(), expiry_days=30)
    assert len(legs) == 2
    assert legs[0]["strike"] == legs[1]["strike"]


def test_strategies_ratio_spread_is_long_one_short_two():
    legs = build_default_legs("ratio_spread", _chain(), expiry_days=30, lots=1, lot_size=1)
    assert len(legs) == 2
    long_leg = next(l for l in legs if l["action"] == "BUY")
    short_leg = next(l for l in legs if l["action"] == "SELL")
    assert short_leg["qty"] == 2 * long_leg["qty"]
    assert short_leg["strike"] > long_leg["strike"]


# ─── lot-size + IV conversion ─────────────────────────────────────────────


def test_strategies_lot_size_multiplies_quantities():
    legs = build_default_legs("straddle", _chain(), expiry_days=30, lots=2, lot_size=50)
    assert all(l["qty"] == 100 for l in legs)


def test_strategies_iv_is_converted_to_decimal():
    legs = build_default_legs("straddle", _chain(vol=22.5), expiry_days=30)
    assert all(l["iv"] == pytest.approx(0.225) for l in legs)


def test_strategies_premium_pulled_from_chain_ltp():
    legs = build_default_legs("bull_call_spread", _chain(), expiry_days=30)
    # ATM call premium is intrinsic 0 + buffer 2.0 in the synthetic chain.
    assert legs[0]["premium"] == pytest.approx(2.0)


def test_strategies_expiry_days_propagates_to_every_leg():
    legs = build_default_legs("iron_condor", _chain(), expiry_days=14)
    assert all(l["expiry_days"] == 14 for l in legs)


def test_strategies_strikes_snap_to_nearest_chain_strike():
    # Chain step is 5; if requested strike falls between rows, use closest.
    chain = _chain(atm=100.0, step=5.0)
    legs = build_default_legs("bull_call_spread", chain, expiry_days=30)
    # buy ATM (100), sell ATM + 2*5 = 110
    assert legs[0]["strike"] == 100.0
    assert legs[1]["strike"] == 110.0


def test_strategies_step_estimation_handles_irregular_chain():
    chain = {
        "symbol": "TEST",
        "underlying": 24500,
        "atm_strike": 24500,
        "calls": [{"strike": s, "ltp": 100, "iv": 18.0, "oi": 1} for s in (24300, 24400, 24500, 24600, 24700)],
        "puts": [{"strike": s, "ltp": 100, "iv": 18.0, "oi": 1} for s in (24300, 24400, 24500, 24600, 24700)],
    }
    legs = build_default_legs("iron_condor", chain, expiry_days=7)
    strikes = sorted(l["strike"] for l in legs)
    # Expect 24300, 24400, 24600, 24700 — wings 100 either side of body.
    assert strikes == [24300.0, 24400.0, 24600.0, 24700.0]
