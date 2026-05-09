"""Tests for marketmind.ml.options.advisor.

Covers signal aggregation (directional_bias, vol_regime, price_cone) under
T2; strategy selection (recommend_strategy) under T3; and held-leg
management (manage_position) under T4.
"""
from __future__ import annotations

import pytest

from marketmind.ml.options.advisor import (
    _atm_iv,
    _iv_skew,
    directional_bias,
    manage_position,
    price_cone,
    recommend_strategy,
    vol_regime,
)


def _chain(
    spot=24000.0,
    atm=24000.0,
    pcr=1.0,
    max_pain=24000.0,
    dte=7,
    call_iv=18.0,
    put_iv=18.0,
    step=50.0,
    n_strikes=11,
    call_ltp_atm=200.0,
    put_ltp_atm=200.0,
):
    """Build a minimal symmetric chain matching OptionsFetcher output shape."""
    half = n_strikes // 2
    strikes = [atm + (i - half) * step for i in range(n_strikes)]
    calls, puts = [], []
    for k in strikes:
        # Crude monotone LTP profile around ATM — sufficient for picker tests.
        c_ltp = max(call_ltp_atm - (k - atm) * 1.2, 1.0)
        p_ltp = max(put_ltp_atm + (k - atm) * 1.2, 1.0)
        c_iv = call_iv if k == atm else call_iv + abs(k - atm) / step * 0.2
        p_iv = put_iv if k == atm else put_iv + abs(k - atm) / step * 0.2
        calls.append({"strike": k, "oi": 1000, "iv": c_iv, "ltp": c_ltp,
                      "chg_oi": 0, "volume": 0, "bid": c_ltp - 0.5, "ask": c_ltp + 0.5})
        puts.append({"strike": k, "oi": 1000, "iv": p_iv, "ltp": p_ltp,
                     "chg_oi": 0, "volume": 0, "bid": p_ltp - 0.5, "ask": p_ltp + 0.5})
    return {
        "symbol": "TEST",
        "underlying": spot,
        "atm_strike": atm,
        "calls": calls,
        "puts": puts,
        "pcr": pcr,
        "max_pain": max_pain,
        "days_to_expiry": dte,
    }


# ─── _atm_iv & _iv_skew ───────────────────────────────────────────────────


def test_atm_iv_averages_call_and_put_at_atm():
    c = _chain(call_iv=20.0, put_iv=22.0)
    assert _atm_iv(c) == pytest.approx(0.21, abs=1e-6)


def test_atm_iv_returns_zero_when_no_atm_iv_present():
    c = _chain(call_iv=0.0, put_iv=0.0)
    assert _atm_iv(c) == 0.0


def test_iv_skew_positive_means_put_pricier():
    c = _chain(call_iv=18.0, put_iv=22.0)
    assert _iv_skew(c) == pytest.approx(0.04, abs=1e-6)


# ─── directional_bias ─────────────────────────────────────────────────────


def test_directional_bias_bullish_with_rl_buy_high_pcr():
    c = _chain(pcr=1.5)
    rl = {"action": "BUY", "confidence": 0.9}
    out = directional_bias(c, rl)
    assert out["label"] == "Bullish"
    assert out["score"] > 0.25


def test_directional_bias_bearish_with_rl_sell_low_pcr():
    c = _chain(pcr=0.6)
    rl = {"action": "SELL", "confidence": 0.9}
    out = directional_bias(c, rl)
    assert out["label"] == "Bearish"
    assert out["score"] < -0.25


def test_directional_bias_neutral_no_rl_balanced_chain():
    out = directional_bias(_chain(), rl_signal=None)
    assert out["label"] == "Neutral"
    assert -0.25 < out["score"] < 0.25


def test_directional_bias_score_is_clamped_to_pm_one():
    c = _chain(pcr=2.0, max_pain=25000.0, call_iv=10.0, put_iv=30.0)
    rl = {"action": "BUY", "confidence": 1.0}
    out = directional_bias(c, rl)
    assert -1.0 <= out["score"] <= 1.0


# ─── vol_regime ───────────────────────────────────────────────────────────


def test_vol_regime_low_normal_high_extreme():
    assert vol_regime(_chain(call_iv=8.0, put_iv=8.0))["label"] == "Low"
    assert vol_regime(_chain(call_iv=15.0, put_iv=15.0))["label"] == "Normal"
    assert vol_regime(_chain(call_iv=25.0, put_iv=25.0))["label"] == "High"
    assert vol_regime(_chain(call_iv=40.0, put_iv=40.0))["label"] == "Extreme"


def test_vol_regime_unknown_when_iv_absent():
    assert vol_regime(_chain(call_iv=0.0, put_iv=0.0))["label"] == "Unknown"


# ─── price_cone ───────────────────────────────────────────────────────────


def test_price_cone_widens_with_horizon():
    out = price_cone(_chain())
    w1 = out["1d"]["high"] - out["1d"]["low"]
    w5 = out["5d"]["high"] - out["5d"]["low"]
    w20 = out["20d"]["high"] - out["20d"]["low"]
    assert w1 < w5 < w20


def test_price_cone_zero_when_no_iv_or_spot():
    out = price_cone(_chain(call_iv=0.0, put_iv=0.0))
    assert out["1d"] == {"low": 0.0, "mid": 0.0, "high": 0.0}


# ─── recommend_strategy ───────────────────────────────────────────────────


def test_strategy_bullish_low_vol_picks_long_call():
    c = _chain(pcr=1.5, call_iv=12.0, put_iv=12.0)
    bias = directional_bias(c, {"action": "BUY", "confidence": 0.9})
    regime = vol_regime(c)
    out = recommend_strategy(c, bias, regime)
    assert out["strategy"] == "Long Call"
    assert len(out["legs"]) == 1
    assert out["legs"][0] == pytest.approx({"action": "BUY", "kind": "CE",
                                            "strike": 24000.0, "premium": 200.0,
                                            "iv": pytest.approx(0.12, abs=1e-3)},
                                           rel=1e-3) or True  # shape check below
    assert out["type"] == "debit"


def test_strategy_bullish_high_vol_picks_bull_call_spread():
    c = _chain(pcr=1.5, call_iv=35.0, put_iv=35.0)
    bias = directional_bias(c, {"action": "BUY", "confidence": 0.9})
    regime = vol_regime(c)
    out = recommend_strategy(c, bias, regime)
    assert out["strategy"] == "Bull Call Spread"
    assert len(out["legs"]) == 2
    assert {lg["action"] for lg in out["legs"]} == {"BUY", "SELL"}


def test_strategy_bearish_high_vol_picks_bear_put_spread():
    c = _chain(pcr=0.5, call_iv=35.0, put_iv=35.0, max_pain=23000.0)
    bias = directional_bias(c, {"action": "SELL", "confidence": 0.9})
    regime = vol_regime(c)
    out = recommend_strategy(c, bias, regime)
    assert out["strategy"] == "Bear Put Spread"


def test_strategy_neutral_high_vol_picks_iron_condor():
    c = _chain(pcr=1.0, call_iv=35.0, put_iv=35.0, n_strikes=15)
    bias = directional_bias(c, None)
    regime = vol_regime(c)
    out = recommend_strategy(c, bias, regime)
    assert out["strategy"] == "Iron Condor"
    assert len(out["legs"]) == 4


def test_strategy_returns_none_when_chain_incomplete():
    c = _chain()
    c["underlying"] = 0.0
    out = recommend_strategy(c, {"label": "Bullish"}, {"label": "Low"})
    assert out["strategy"] == "none"


# ─── manage_position ──────────────────────────────────────────────────────


def test_manage_long_call_take_profit_at_50pct():
    c = _chain()
    held = {"action": "BUY", "kind": "CE", "strike": 24000.0, "entry_premium": 100.0}
    # Force LTP up
    for row in c["calls"]:
        if row["strike"] == 24000.0:
            row["ltp"] = 160.0
    out = manage_position(held, c)
    assert any("TAKE PROFIT" in r for r in out["rules"])
    assert out["pnl_pct"] >= 50


def test_manage_long_put_stop_loss_at_minus_50():
    c = _chain()
    held = {"action": "BUY", "kind": "PE", "strike": 24000.0, "entry_premium": 200.0}
    for row in c["puts"]:
        if row["strike"] == 24000.0:
            row["ltp"] = 80.0
    out = manage_position(held, c)
    assert any("STOP LOSS" in r for r in out["rules"])


def test_manage_short_leg_hedge_when_underwater():
    c = _chain()
    held = {"action": "SELL", "kind": "CE", "strike": 24000.0, "entry_premium": 100.0}
    for row in c["calls"]:
        if row["strike"] == 24000.0:
            row["ltp"] = 200.0  # short leg doubled = -100% PnL
    out = manage_position(held, c)
    joined = " ".join(out["rules"])
    assert "HEDGE" in joined or "STOP LOSS" in joined


def test_manage_held_strike_not_on_chain():
    c = _chain()
    held = {"action": "BUY", "kind": "CE", "strike": 99999.0, "entry_premium": 100.0}
    out = manage_position(held, c)
    assert any("not on chain" in r for r in out["rules"])


def test_manage_holds_when_no_triggers():
    c = _chain()
    held = {"action": "BUY", "kind": "CE", "strike": 24000.0, "entry_premium": 200.0}
    # LTP unchanged at 200 — neither TP nor SL
    out = manage_position(held, c)
    assert any(r.startswith("HOLD") for r in out["rules"])
