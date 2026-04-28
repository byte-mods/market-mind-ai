"""T4: Tax-loss-harvest detector tests.

Verifies `find_harvest_candidates` and the `harvest_losses=True` integration
on `recommend_tax_optimal_rebalance`.
"""
from __future__ import annotations

import datetime as dt

import pytest

from marketmind.analysis.tax_engine import GAIN_LTCG, GAIN_STCG, GAIN_UNKNOWN
from marketmind.analysis.tax_lots import TaxLot
from marketmind.analysis.tax_rebalancer import (
    CurrentHolding,
    HarvestCandidate,
    find_harvest_candidates,
    recommend_tax_optimal_rebalance,
)


AS_OF = dt.date(2026, 4, 27)


def _h(sym, qty, px, avg, lots=None):
    return CurrentHolding(symbol=sym, quantity=qty, last_price=px, avg_cost=avg, lots=lots)


# ─── threshold validation ────────────────────────────────────────────────


def test_tax_harvest_negative_thresholds_rejected() -> None:
    with pytest.raises(ValueError):
        find_harvest_candidates([], AS_OF, min_loss_inr=-1)
    with pytest.raises(ValueError):
        find_harvest_candidates([], AS_OF, min_loss_pct=-1)


# ─── basic candidate detection ──────────────────────────────────────────


def test_tax_harvest_no_holdings_returns_empty() -> None:
    assert find_harvest_candidates([], AS_OF) == []


def test_tax_harvest_only_gains_returns_empty() -> None:
    h = _h("A", 10, 200, 100,
           lots=[TaxLot("A", 10, 100.0, dt.date(2025, 1, 1))])
    assert find_harvest_candidates([h], AS_OF) == []


def test_tax_harvest_loss_above_threshold_returned() -> None:
    """16.7%/₹2000 loss above default thresholds (₹1000 / 5%) → returned."""
    h = _h("A", 100, 100, 120,
           lots=[TaxLot("A", 100, 120.0, dt.date(2026, 1, 1))])
    cands = find_harvest_candidates([h], AS_OF)
    assert len(cands) == 1
    assert cands[0].symbol == "A"
    assert cands[0].est_loss_inr == pytest.approx(2_000.0)


def test_tax_harvest_loss_below_min_inr_filtered() -> None:
    """Loss ₹50 (10 × ₹5) is below default ₹1000 threshold → filtered."""
    h = _h("A", 10, 95, 100,
           lots=[TaxLot("A", 10, 100.0, dt.date(2026, 1, 1))])
    cands = find_harvest_candidates([h], AS_OF)
    # 10 × 5 = ₹50 loss; below default ₹1000
    assert cands == []


def test_tax_harvest_loss_below_min_pct_filtered() -> None:
    """High-rupee but low-percent loss filtered when min_loss_pct enforced."""
    # 100 shares × (200 → 195) → ₹500 loss = 2.5% — below 5% threshold
    h = _h("A", 100, 195, 200,
           lots=[TaxLot("A", 100, 200.0, dt.date(2026, 1, 1))])
    # min_loss_inr=100 to clear the rupee gate; pct gate at default 5% should filter
    cands = find_harvest_candidates([h], AS_OF, min_loss_inr=100, min_loss_pct=5.0)
    assert cands == []


def test_tax_harvest_zero_cost_basis_lot_skipped() -> None:
    """Defensive: a 0-cost-basis lot can't be harvested (loss_pct undefined)."""
    h = _h("A", 10, 100, 0,
           lots=[TaxLot("A", 10, 0.0, dt.date(2026, 1, 1))])
    assert find_harvest_candidates([h], AS_OF) == []


def test_tax_harvest_zero_last_price_skipped() -> None:
    """A holding with no usable mark-to-market price can't be assessed."""
    h = _h("A", 10, 0.0, 100, lots=[TaxLot("A", 10, 100.0, dt.date(2026, 1, 1))])
    assert find_harvest_candidates([h], AS_OF) == []


# ─── lot-level granularity ──────────────────────────────────────────────


def test_tax_harvest_walks_at_lot_level_not_position_level() -> None:
    """One holding with a gain lot AND a loss lot must surface only the loss."""
    h = _h("A", 20, 110, 110, lots=[
        TaxLot("A", 10, 80.0, dt.date(2026, 1, 1)),    # gain lot
        TaxLot("A", 10, 200.0, dt.date(2026, 2, 1)),   # loss lot (₹90/sh × 10 = ₹900)
    ])
    cands = find_harvest_candidates([h], AS_OF, min_loss_inr=500.0)
    assert len(cands) == 1
    assert cands[0].cost_basis == 200.0
    assert cands[0].est_loss_inr == pytest.approx(900.0)


def test_tax_harvest_no_lots_uses_unknown_bucket() -> None:
    """Holding without ledger is bucketed as UNKNOWN (treated as STCG)."""
    h = _h("A", 10, 50, 100, lots=None)  # ₹500 loss = 50% — well above thresholds
    cands = find_harvest_candidates([h], AS_OF, min_loss_inr=400.0)
    assert len(cands) == 1
    assert cands[0].gain_type == GAIN_UNKNOWN
    assert cands[0].holding_period_days is None


def test_tax_harvest_classifies_ltcg_loss_correctly() -> None:
    h = _h("A", 100, 50, 100,
           lots=[TaxLot("A", 100, 100.0, dt.date(2024, 1, 1))])  # >365 days
    cands = find_harvest_candidates([h], AS_OF)
    assert cands[0].gain_type == GAIN_LTCG
    assert cands[0].holding_period_days is not None
    assert cands[0].holding_period_days > 365


def test_tax_harvest_classifies_stcg_loss_correctly() -> None:
    h = _h("A", 100, 50, 100,
           lots=[TaxLot("A", 100, 100.0, dt.date(2026, 1, 1))])  # <365 days
    cands = find_harvest_candidates([h], AS_OF)
    assert cands[0].gain_type == GAIN_STCG


# ─── ordering ───────────────────────────────────────────────────────────


def test_tax_harvest_results_sorted_by_loss_size_desc() -> None:
    holdings = [
        _h("A", 10, 50, 100, lots=[TaxLot("A", 10, 100.0, dt.date(2026, 1, 1))]),  # ₹500
        _h("B", 10, 80, 100, lots=[TaxLot("B", 10, 100.0, dt.date(2026, 1, 1))]),  # ₹200
        _h("C", 10, 60, 100, lots=[TaxLot("C", 10, 100.0, dt.date(2026, 1, 1))]),  # ₹400
    ]
    cands = find_harvest_candidates(holdings, AS_OF, min_loss_inr=100.0)
    assert [c.symbol for c in cands] == ["A", "C", "B"]
    assert all(
        cands[i].est_loss_inr >= cands[i + 1].est_loss_inr
        for i in range(len(cands) - 1)
    )


def test_tax_harvest_advisory_text_warns_about_30_day_repurchase() -> None:
    h = _h("A", 100, 50, 100,
           lots=[TaxLot("A", 100, 100.0, dt.date(2026, 1, 1))])
    cands = find_harvest_candidates([h], AS_OF)
    assert "30 days" in cands[0].advisory


# ─── integration with recommend_tax_optimal_rebalance ───────────────────


def test_tax_harvest_recommend_with_harvest_off_returns_no_candidates() -> None:
    """Default `harvest_losses=False` → harvest_candidates is empty even when
    losses exist."""
    holdings = [
        _h("A", 10, 50, 100, lots=[TaxLot("A", 10, 100.0, dt.date(2026, 1, 1))]),
        _h("B", 10, 100, 90, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, {"A": 0.5, "B": 0.5}, AS_OF,
    )
    assert rec.harvest_candidates == []


def test_tax_harvest_recommend_with_harvest_on_returns_candidates() -> None:
    holdings = [
        _h("A", 100, 50, 100, lots=[TaxLot("A", 100, 100.0, dt.date(2026, 1, 1))]),
        _h("B", 100, 100, 90, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, {"A": 0.5, "B": 0.5}, AS_OF,
        harvest_losses=True,
    )
    assert len(rec.harvest_candidates) == 1
    assert rec.harvest_candidates[0].symbol == "A"
    assert isinstance(rec.harvest_candidates[0], HarvestCandidate)


def test_tax_harvest_recommend_thresholds_passed_through() -> None:
    """Custom thresholds applied in recommend(...) reach find_harvest_candidates."""
    holdings = [
        _h("A", 10, 95, 100, lots=[TaxLot("A", 10, 100.0, dt.date(2026, 1, 1))]),  # ₹50 loss = 5%
        _h("B", 10, 100, 90, lots=None),
    ]
    # Default thresholds reject (₹50 < ₹1000); relaxed thresholds accept.
    rec_default = recommend_tax_optimal_rebalance(
        holdings, {"A": 0.5, "B": 0.5}, AS_OF, harvest_losses=True,
    )
    rec_relaxed = recommend_tax_optimal_rebalance(
        holdings, {"A": 0.5, "B": 0.5}, AS_OF,
        harvest_losses=True,
        harvest_min_loss_inr=10.0,
        harvest_min_loss_pct=1.0,
    )
    assert rec_default.harvest_candidates == []
    assert len(rec_relaxed.harvest_candidates) == 1
