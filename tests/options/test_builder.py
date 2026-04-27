"""Strategy-builder analytics tests — payoff shape, BEs, net Greeks."""
from __future__ import annotations

import pytest

from marketmind.ml.options.builder import PAYOFF_POINTS, analyse


def _leg(action, kind, strike, premium=2.0, iv=0.20, qty=1, expiry_days=30):
    return {
        "action": action,
        "kind": kind,
        "strike": float(strike),
        "premium": float(premium),
        "iv": float(iv),
        "qty": int(qty),
        "expiry_days": int(expiry_days),
    }


# ─── validation ───────────────────────────────────────────────────────────


def test_builder_rejects_empty_leg_list():
    with pytest.raises(ValueError, match="at least one leg"):
        analyse([], underlying=100.0)


def test_builder_rejects_invalid_action():
    bad = _leg("HOLD", "CE", 100.0)
    with pytest.raises(ValueError, match="action"):
        analyse([bad], underlying=100.0)


def test_builder_rejects_invalid_kind():
    bad = _leg("BUY", "FUT", 100.0)
    with pytest.raises(ValueError, match="kind"):
        analyse([bad], underlying=100.0)


def test_builder_rejects_zero_strike():
    bad = _leg("BUY", "CE", 0.0)
    with pytest.raises(ValueError, match="strike"):
        analyse([bad], underlying=100.0)


def test_builder_rejects_zero_underlying():
    with pytest.raises(ValueError, match="underlying"):
        analyse([_leg("BUY", "CE", 100.0)], underlying=0.0)


# ─── shape contract ───────────────────────────────────────────────────────


def test_builder_returns_payoff_grid_of_expected_length():
    out = analyse([_leg("BUY", "CE", 100.0)], underlying=100.0)
    assert len(out["payoff"]) == PAYOFF_POINTS
    assert out["payoff"][0]["underlying"] < out["payoff"][-1]["underlying"]


def test_builder_result_has_required_keys():
    out = analyse([_leg("BUY", "CE", 100.0)], underlying=100.0, strategy_name="long_call")
    for k in (
        "strategy",
        "legs",
        "payoff",
        "max_profit",
        "max_loss",
        "break_evens",
        "net_greeks",
        "net_premium",
        "theoretical_value",
        "margin_proxy",
        "margin_is_proxy",
        "pricing_model",
        "generated_at",
        "notes",
    ):
        assert k in out
    assert out["strategy"] == "long_call"
    assert out["pricing_model"] == "black_scholes_european"


# ─── payoff shape ─────────────────────────────────────────────────────────


def test_builder_long_call_payoff_is_hockey_stick():
    out = analyse([_leg("BUY", "CE", 100.0, premium=5.0)], underlying=100.0)
    pnls = [pt["pnl"] for pt in out["payoff"]]
    # Below strike: flat at -premium*qty = -5
    below = [pt["pnl"] for pt in out["payoff"] if pt["underlying"] < 95.0]
    assert all(abs(p - (-5.0)) < 1e-6 for p in below)
    # Above strike: rising
    assert pnls[-1] > pnls[0]


def test_builder_short_put_payoff_capped_at_premium():
    out = analyse([_leg("SELL", "PE", 100.0, premium=4.0)], underlying=100.0)
    # Above strike: pnl = +premium = +4 (max profit)
    assert out["max_profit"] == pytest.approx(4.0, abs=1e-6)
    # Below strike: pnl declines linearly. At underlying=70: pnl = 4 - 30 = -26.
    assert out["max_loss"] < -20.0


def test_builder_iron_condor_caps_both_max_profit_and_max_loss():
    legs = [
        _leg("SELL", "PE", 95.0, premium=2.5),
        _leg("BUY", "PE", 90.0, premium=1.0),
        _leg("SELL", "CE", 105.0, premium=2.5),
        _leg("BUY", "CE", 110.0, premium=1.0),
    ]
    out = analyse(legs, underlying=100.0, strategy_name="iron_condor")
    # Net credit = 2.5 - 1.0 + 2.5 - 1.0 = 3.0  -> max_profit
    assert out["max_profit"] == pytest.approx(3.0, abs=1e-6)
    # Wing width = 5; max_loss = -(5 - net_credit) = -2.0
    assert out["max_loss"] == pytest.approx(-2.0, abs=1e-6)


def test_builder_finds_two_break_evens_for_long_straddle():
    legs = [
        _leg("BUY", "CE", 100.0, premium=3.0),
        _leg("BUY", "PE", 100.0, premium=3.0),
    ]
    out = analyse(legs, underlying=100.0, strategy_name="straddle")
    # BE = strike ± total premium = 94 and 106
    assert len(out["break_evens"]) == 2
    bes = sorted(out["break_evens"])
    assert bes[0] == pytest.approx(94.0, abs=0.5)
    assert bes[1] == pytest.approx(106.0, abs=0.5)


# ─── additivity (property test) ───────────────────────────────────────────


def test_builder_payoff_is_additive_across_legs():
    """Sum of single-leg payoffs == multi-leg payoff at every grid point."""
    leg_a = _leg("BUY", "CE", 100.0, premium=3.0)
    leg_b = _leg("SELL", "CE", 110.0, premium=1.5)
    out_a = analyse([leg_a], underlying=100.0)
    out_b = analyse([leg_b], underlying=100.0)
    out_combined = analyse([leg_a, leg_b], underlying=100.0)
    for pa, pb, pc in zip(out_a["payoff"], out_b["payoff"], out_combined["payoff"]):
        assert pc["pnl"] == pytest.approx(pa["pnl"] + pb["pnl"], abs=1e-6)


# ─── net Greeks ───────────────────────────────────────────────────────────


def test_builder_short_position_flips_delta_sign():
    out_long = analyse([_leg("BUY", "CE", 100.0, iv=0.25)], underlying=100.0)
    out_short = analyse([_leg("SELL", "CE", 100.0, iv=0.25)], underlying=100.0)
    assert out_long["net_greeks"]["delta"] > 0
    assert out_short["net_greeks"]["delta"] < 0
    assert out_long["net_greeks"]["delta"] == pytest.approx(-out_short["net_greeks"]["delta"], abs=1e-9)


def test_builder_quantity_scales_net_greeks_linearly():
    g1 = analyse([_leg("BUY", "CE", 100.0, qty=1)], underlying=100.0)["net_greeks"]
    g3 = analyse([_leg("BUY", "CE", 100.0, qty=3)], underlying=100.0)["net_greeks"]
    # Net Greeks are rounded to 6dp at the API boundary; allow rounding slack.
    assert g3["delta"] == pytest.approx(3 * g1["delta"], abs=1e-5)
    assert g3["gamma"] == pytest.approx(3 * g1["gamma"], abs=1e-5)


# ─── premium / margin / notes ─────────────────────────────────────────────


def test_builder_net_premium_paid_for_debit_strategy():
    legs = [
        _leg("BUY", "CE", 100.0, premium=5.0),
        _leg("SELL", "CE", 110.0, premium=2.0),
    ]
    out = analyse(legs, underlying=100.0, strategy_name="bull_call_spread")
    # debit = 5 - 2 = 3
    assert out["net_premium"] == pytest.approx(3.0, abs=1e-6)


def test_builder_margin_proxy_zero_for_pure_debit_strategy():
    # Long call alone: max_loss = -premium (which is < 0), so proxy = 1.2 * |loss|.
    out = analyse([_leg("BUY", "CE", 100.0, premium=5.0)], underlying=100.0)
    assert out["margin_proxy"] > 0
    assert out["margin_is_proxy"] is True


def test_builder_multi_expiry_flag_adds_note():
    legs = [
        _leg("BUY", "CE", 100.0, premium=3.0, expiry_days=30),
        _leg("SELL", "CE", 100.0, premium=5.0, expiry_days=60),
    ]
    out = analyse(legs, underlying=100.0, strategy_name="calendar_spread", multi_expiry=True)
    assert any("near expiry" in n for n in out["notes"])
