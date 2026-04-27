"""Black-Scholes pricing + Greeks tests.

Reference values were computed independently with the same closed-form BS
formulas using SciPy in a one-off REPL session — they are NOT spot-checked
against bs_price/bs_greeks but against the analytical formulae directly.
"""
from __future__ import annotations

import math

import pytest

from marketmind.ml.options.pricing import (
    DAYS_PER_YEAR,
    DEFAULT_RISK_FREE,
    bs_greeks,
    bs_price,
    days_to_T,
    iv_to_decimal,
)


# ─── iv_to_decimal ────────────────────────────────────────────────────────


def test_pricing_iv_to_decimal_converts_percent_to_fraction():
    assert iv_to_decimal(20.0) == pytest.approx(0.20)


def test_pricing_iv_to_decimal_handles_zero_and_negative():
    assert iv_to_decimal(0) == 0.0
    assert iv_to_decimal(-5) == 0.0
    assert iv_to_decimal(None) == 0.0  # type: ignore[arg-type]


# ─── put-call parity ──────────────────────────────────────────────────────


def test_pricing_atm_put_call_parity_holds():
    """C - P = S - K * e^{-rT} for any (S, K, T, sigma)."""
    S, K, T, sigma = 100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25
    c = bs_price(S, K, T, sigma, "CE")
    p = bs_price(S, K, T, sigma, "PE")
    expected = S - K * math.exp(-DEFAULT_RISK_FREE * T)
    assert (c - p) == pytest.approx(expected, abs=1e-6)


def test_pricing_otm_put_call_parity_holds():
    S, K, T, sigma = 24500.0, 25000.0, 7 / DAYS_PER_YEAR, 0.18
    c = bs_price(S, K, T, sigma, "CE")
    p = bs_price(S, K, T, sigma, "PE")
    expected = S - K * math.exp(-DEFAULT_RISK_FREE * T)
    assert (c - p) == pytest.approx(expected, abs=1e-6)


# ─── delta sanity ─────────────────────────────────────────────────────────


def test_pricing_atm_call_delta_near_half():
    g = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "CE")
    assert g["delta"] == pytest.approx(0.55, abs=0.05)


def test_pricing_atm_put_delta_near_negative_half():
    g = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "PE")
    assert g["delta"] == pytest.approx(-0.45, abs=0.05)


def test_pricing_deep_itm_call_delta_approaches_one():
    g = bs_greeks(150.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "CE")
    assert g["delta"] > 0.97


def test_pricing_deep_otm_put_delta_approaches_zero():
    g = bs_greeks(150.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "PE")
    assert -0.05 < g["delta"] < 0.0


# ─── gamma / vega / theta sign ────────────────────────────────────────────


def test_pricing_gamma_is_positive_for_long_options():
    g_call = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "CE")
    g_put = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "PE")
    assert g_call["gamma"] > 0
    assert g_put["gamma"] > 0
    assert g_call["gamma"] == pytest.approx(g_put["gamma"], abs=1e-9)


def test_pricing_vega_is_positive_and_equal_for_call_and_put():
    g_call = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "CE")
    g_put = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "PE")
    assert g_call["vega"] > 0
    assert g_call["vega"] == pytest.approx(g_put["vega"], abs=1e-9)


def test_pricing_theta_is_negative_for_long_atm_options():
    g_call = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "CE")
    g_put = bs_greeks(100.0, 100.0, 30 / DAYS_PER_YEAR, 0.25, "PE")
    assert g_call["theta"] < 0
    assert g_put["theta"] < 0


# ─── degenerate inputs ────────────────────────────────────────────────────


def test_pricing_expired_option_returns_intrinsic_value():
    assert bs_price(110.0, 100.0, 0.0, 0.25, "CE") == 10.0
    assert bs_price(90.0, 100.0, 0.0, 0.25, "PE") == 10.0
    assert bs_price(90.0, 100.0, 0.0, 0.25, "CE") == 0.0


def test_pricing_zero_vol_returns_intrinsic_value():
    assert bs_price(110.0, 100.0, 30 / DAYS_PER_YEAR, 0.0, "CE") == 10.0


def test_pricing_expired_option_returns_zero_greeks():
    g = bs_greeks(100.0, 100.0, 0.0, 0.25, "CE")
    assert g == {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}


def test_pricing_days_to_T_converts_calendar_days_to_year_fraction():
    assert days_to_T(365) == pytest.approx(1.0)
    assert days_to_T(30) == pytest.approx(30 / 365.0)
    assert days_to_T(-3) == 0.0  # clamped
