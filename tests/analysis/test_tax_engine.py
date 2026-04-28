"""T1: FY26 tax-engine unit tests.

Boundaries verified:
  • holding period exactly 365 days  → STCG (day 366 is the first LTCG day)
  • LTCG ₹1.25L exemption applied once per fiscal year (caller passes used)
  • UNKNOWN acquisition date treated as STCG (worst-case fallback)
  • Net negative buckets are clamped to zero — no cross-bucket offset
"""
from __future__ import annotations

import datetime as dt

import pytest

from marketmind.analysis.tax_engine import (
    GAIN_LTCG,
    GAIN_STCG,
    GAIN_UNKNOWN,
    LTCG_EXEMPTION_INR,
    LTCG_RATE,
    STCG_RATE,
    RealizedGain,
    classify_holding_period,
    compute_tax,
    realize_gain,
)

# ─── classify_holding_period ─────────────────────────────────────────────


def test_tax_engine_classify_period_below_boundary_is_stcg() -> None:
    days, kind = classify_holding_period(dt.date(2025, 1, 1), dt.date(2025, 7, 1))
    assert kind == GAIN_STCG
    assert days == 181


def test_tax_engine_classify_period_at_exactly_365_days_is_stcg() -> None:
    """Day 365 stays in the short-term bucket; LTCG begins at day 366."""
    days, kind = classify_holding_period(dt.date(2025, 1, 1), dt.date(2026, 1, 1))
    assert days == 365
    assert kind == GAIN_STCG


def test_tax_engine_classify_period_at_366_days_is_ltcg() -> None:
    days, kind = classify_holding_period(dt.date(2025, 1, 1), dt.date(2026, 1, 2))
    assert days == 366
    assert kind == GAIN_LTCG


def test_tax_engine_classify_unknown_acquisition_is_unknown() -> None:
    days, kind = classify_holding_period(None, dt.date(2026, 4, 27))
    assert days is None
    assert kind == GAIN_UNKNOWN


def test_tax_engine_classify_negative_window_raises() -> None:
    with pytest.raises(ValueError):
        classify_holding_period(dt.date(2026, 1, 1), dt.date(2025, 1, 1))


def test_tax_engine_classify_accepts_datetime_and_date_alike() -> None:
    """datetime and date inputs must yield the same classification."""
    d_acq = dt.date(2025, 1, 1)
    dt_now = dt.datetime(2026, 1, 2, 14, 30, 0)
    days_dt, kind_dt = classify_holding_period(d_acq, dt_now)
    days_d, kind_d = classify_holding_period(d_acq, dt.date(2026, 1, 2))
    assert (days_dt, kind_dt) == (days_d, kind_d) == (366, GAIN_LTCG)


# ─── realize_gain ────────────────────────────────────────────────────────


def test_tax_engine_realize_gain_profit_is_positive() -> None:
    g = realize_gain(
        symbol="RELIANCE",
        quantity=10,
        cost_basis=2000.0,
        sell_price=2500.0,
        acquisition_date=dt.date(2024, 1, 1),
        as_of=dt.date(2026, 1, 2),  # > 365 days
    )
    assert g.gain_inr == pytest.approx(5_000.0)
    assert g.gain_type == GAIN_LTCG
    assert g.holding_period_days == 732


def test_tax_engine_realize_gain_loss_is_negative() -> None:
    g = realize_gain(
        symbol="TCS",
        quantity=5,
        cost_basis=3500.0,
        sell_price=3000.0,
        acquisition_date=dt.date(2026, 1, 1),
        as_of=dt.date(2026, 4, 1),
    )
    assert g.gain_inr == pytest.approx(-2_500.0)
    assert g.gain_type == GAIN_STCG


def test_tax_engine_realize_gain_unknown_acquisition_is_unknown_type() -> None:
    g = realize_gain(
        symbol="INFY",
        quantity=3,
        cost_basis=1500.0,
        sell_price=1800.0,
        acquisition_date=None,
        as_of=dt.date(2026, 4, 27),
    )
    assert g.gain_type == GAIN_UNKNOWN
    assert g.holding_period_days is None
    assert g.gain_inr == pytest.approx(900.0)


def test_tax_engine_realize_gain_negative_quantity_rejected() -> None:
    with pytest.raises(ValueError):
        realize_gain(
            symbol="X",
            quantity=-1,
            cost_basis=100.0,
            sell_price=120.0,
            acquisition_date=dt.date(2025, 1, 1),
            as_of=dt.date(2026, 1, 1),
        )


def test_tax_engine_realize_gain_zero_quantity_yields_zero_gain() -> None:
    """Edge: a zero-quantity sale is degenerate but not invalid."""
    g = realize_gain(
        symbol="X",
        quantity=0,
        cost_basis=100.0,
        sell_price=120.0,
        acquisition_date=dt.date(2025, 1, 1),
        as_of=dt.date(2026, 4, 27),
    )
    assert g.gain_inr == 0.0


# ─── compute_tax ─────────────────────────────────────────────────────────


def _gain(kind: str, amount: float, sym: str = "X") -> RealizedGain:
    """Test helper: synthesise a RealizedGain of a given bucket and amount."""
    return RealizedGain(
        symbol=sym,
        quantity=1.0,
        cost_basis=0.0,
        sell_price=amount,
        holding_period_days=400 if kind == GAIN_LTCG else 100,
        gain_inr=amount,
        gain_type=kind,
    )


def test_tax_engine_compute_tax_pure_stcg_applies_15_percent() -> None:
    bill = compute_tax([_gain(GAIN_STCG, 100_000.0)])
    assert bill.stcg_tax_inr == pytest.approx(15_000.0)
    assert bill.ltcg_tax_inr == 0.0
    assert bill.total_tax_inr == pytest.approx(15_000.0)


def test_tax_engine_compute_tax_pure_ltcg_under_exemption_is_zero() -> None:
    bill = compute_tax([_gain(GAIN_LTCG, 50_000.0)])
    assert bill.ltcg_taxable_inr == 0.0
    assert bill.ltcg_tax_inr == 0.0
    assert bill.ltcg_exemption_consumed_inr == pytest.approx(50_000.0)


def test_tax_engine_compute_tax_pure_ltcg_above_exemption_taxes_overage() -> None:
    """LTCG of ₹2L → first ₹1.25L exempt, ₹75K taxed at 12.5% = ₹9,375."""
    bill = compute_tax([_gain(GAIN_LTCG, 200_000.0)])
    assert bill.ltcg_taxable_inr == pytest.approx(75_000.0)
    assert bill.ltcg_tax_inr == pytest.approx(9_375.0)
    assert bill.ltcg_exemption_consumed_inr == pytest.approx(LTCG_EXEMPTION_INR)


def test_tax_engine_compute_tax_partial_exemption_with_prior_ltcg_used() -> None:
    """Caller already realised ₹50K LTCG this FY; only ₹75K of exemption left."""
    bill = compute_tax([_gain(GAIN_LTCG, 200_000.0)], ltcg_used_inr=50_000.0)
    # remaining exemption = 1.25L − 0.5L = 0.75L; taxable = 2L − 0.75L = 1.25L
    assert bill.ltcg_exemption_consumed_inr == pytest.approx(75_000.0)
    assert bill.ltcg_taxable_inr == pytest.approx(125_000.0)
    assert bill.ltcg_tax_inr == pytest.approx(125_000.0 * LTCG_RATE)


def test_tax_engine_compute_tax_exhausted_exemption_taxes_full_ltcg() -> None:
    bill = compute_tax([_gain(GAIN_LTCG, 100_000.0)], ltcg_used_inr=200_000.0)
    assert bill.ltcg_exemption_consumed_inr == 0.0
    assert bill.ltcg_taxable_inr == pytest.approx(100_000.0)


def test_tax_engine_compute_tax_unknown_classified_as_stcg() -> None:
    bill = compute_tax([_gain(GAIN_UNKNOWN, 100_000.0)])
    assert bill.stcg_tax_inr == pytest.approx(15_000.0)
    assert bill.ltcg_tax_inr == 0.0


def test_tax_engine_compute_tax_negative_stcg_clamped_to_zero() -> None:
    """A net STCG loss must not trigger a refund; tax is 0, not negative."""
    bill = compute_tax([_gain(GAIN_STCG, -50_000.0)])
    assert bill.stcg_realized_inr == pytest.approx(-50_000.0)
    assert bill.stcg_tax_inr == 0.0


def test_tax_engine_compute_tax_loss_does_not_offset_other_bucket() -> None:
    """Conservative: STCL does NOT reduce LTCG tax in this engine."""
    bill = compute_tax([
        _gain(GAIN_STCG, -50_000.0),
        _gain(GAIN_LTCG, 200_000.0),
    ])
    # LTCG above exemption = 75K → tax = 9,375. STCL is ignored.
    assert bill.stcg_tax_inr == 0.0
    assert bill.ltcg_tax_inr == pytest.approx(9_375.0)


def test_tax_engine_compute_tax_mixed_gains_aggregate_per_bucket() -> None:
    bill = compute_tax([
        _gain(GAIN_STCG, 30_000.0, "A"),
        _gain(GAIN_STCG, 70_000.0, "B"),
        _gain(GAIN_LTCG, 200_000.0, "C"),
        _gain(GAIN_LTCG, -50_000.0, "D"),
    ])
    # net STCG = 100K → 15K tax
    # net LTCG = 150K → 1.25L exempt → 25K taxable → 3,125 tax
    assert bill.stcg_tax_inr == pytest.approx(15_000.0)
    assert bill.ltcg_taxable_inr == pytest.approx(25_000.0)
    assert bill.ltcg_tax_inr == pytest.approx(3_125.0)
    assert bill.total_tax_inr == pytest.approx(18_125.0)


def test_tax_engine_compute_tax_empty_iterable_yields_zero_bill() -> None:
    bill = compute_tax([])
    assert bill.total_tax_inr == 0.0
    assert bill.stcg_realized_inr == 0.0
    assert bill.ltcg_realized_inr == 0.0


def test_tax_engine_compute_tax_negative_ltcg_used_rejected() -> None:
    with pytest.raises(ValueError):
        compute_tax([_gain(GAIN_LTCG, 100_000.0)], ltcg_used_inr=-1.0)


def test_tax_engine_compute_tax_uses_published_rates() -> None:
    """Guard against accidental rate drift in the constants module."""
    assert STCG_RATE == 0.15
    assert LTCG_RATE == 0.125
    assert LTCG_EXEMPTION_INR == 125_000.0
