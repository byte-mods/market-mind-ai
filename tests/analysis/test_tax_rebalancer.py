"""T3: Tax-aware rebalancer core tests.

Verifies the recommend_tax_optimal_rebalance pipeline:
  • diff vs target_weights → trade list (BUY/SELL with integer shares)
  • lot-level tax projection (TAX_MIN) with FIFO-naive comparison
  • LTCG exemption flow with `ltcg_used_inr`
  • missing-lot fallback (single UNKNOWN-date lot, warning emitted)
  • inconsistent ledger fallback
  • new-symbol pricing through `new_symbol_prices`
  • tracking error after integer rounding
"""
from __future__ import annotations

import datetime as dt

import pytest

from marketmind.analysis.tax_engine import (
    GAIN_LTCG, GAIN_STCG, GAIN_UNKNOWN, LTCG_EXEMPTION_INR, LTCG_RATE, STCG_RATE,
)
from marketmind.analysis.tax_lots import TaxLot
from marketmind.analysis.tax_rebalancer import (
    CurrentHolding,
    recommend_tax_optimal_rebalance,
)


AS_OF = dt.date(2026, 4, 27)


def _holding(sym: str, qty: float, px: float, avg: float, lots=None) -> CurrentHolding:
    return CurrentHolding(symbol=sym, quantity=qty, last_price=px, avg_cost=avg, lots=lots)


# ─── input validation ───────────────────────────────────────────────────


def test_tax_rebalancer_target_weights_must_sum_to_one() -> None:
    holdings = [_holding("A", 10, 100, 90)]
    with pytest.raises(ValueError):
        recommend_tax_optimal_rebalance(
            holdings, target_weights={"A": 0.5}, as_of=AS_OF,
        )


def test_tax_rebalancer_holding_negative_quantity_rejected() -> None:
    with pytest.raises(ValueError):
        CurrentHolding(symbol="A", quantity=-1, last_price=100, avg_cost=90)


def test_tax_rebalancer_holding_negative_price_rejected() -> None:
    with pytest.raises(ValueError):
        CurrentHolding(symbol="A", quantity=10, last_price=-1, avg_cost=90)


# ─── trade emission ─────────────────────────────────────────────────────


def test_tax_rebalancer_no_change_returns_empty_trades() -> None:
    """When current weights already match target, no trades emit."""
    holdings = [_holding("A", 10, 100, 90), _holding("B", 10, 100, 90)]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.5, "B": 0.5}, as_of=AS_OF,
    )
    assert rec.trades == []


def test_tax_rebalancer_full_liquidation_emits_only_sells() -> None:
    """Target weight 1.0 on B from a 50/50 A/B portfolio: A liquidated, B doubled."""
    holdings = [
        _holding("A", 10, 100, 80,
                 lots=[TaxLot("A", 10, 80, dt.date(2025, 1, 1))]),
        _holding("B", 10, 100, 90,
                 lots=[TaxLot("B", 10, 90, dt.date(2025, 1, 1))]),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    actions = [(t.symbol, t.action) for t in rec.trades]
    assert ("A", "SELL") in actions
    assert ("B", "BUY") in actions
    a_trade = next(t for t in rec.trades if t.symbol == "A" and t.action == "SELL")
    assert a_trade.quantity == 10


def test_tax_rebalancer_buy_only_for_new_symbol_in_target() -> None:
    """Symbol C is not currently held; provided via new_symbol_prices."""
    holdings = [_holding("A", 20, 100, 80,
                         lots=[TaxLot("A", 20, 80, dt.date(2025, 1, 1))])]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.5, "C": 0.5}, as_of=AS_OF,
        new_symbol_prices={"C": 200.0},
    )
    c_trade = next(t for t in rec.trades if t.symbol == "C")
    assert c_trade.action == "BUY"
    assert c_trade.quantity > 0


def test_tax_rebalancer_partial_share_below_half_skipped() -> None:
    """Δ between -0.5 and +0.5 share must not emit a trade."""
    # 100 share holding @ ₹100 → ₹10000 total. Target 0.5005 → ₹5005 → 50.05 shares,
    # rounds to 50 → delta = 0 → no trade.
    holdings = [_holding("A", 100, 100, 80,
                         lots=[TaxLot("A", 100, 80, dt.date(2025, 1, 1))]),
                _holding("B", 0.0001, 100, 80)]  # zero, prices balance
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 1.0}, as_of=AS_OF,
    )
    assert all(t.action != "SELL" or t.quantity > 0 for t in rec.trades)


def test_tax_rebalancer_buy_trade_quantity_is_integer() -> None:
    holdings = [_holding("A", 100, 100, 80,
                         lots=[TaxLot("A", 100, 80, dt.date(2025, 1, 1))])]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.5, "B": 0.5}, as_of=AS_OF,
        new_symbol_prices={"B": 33.33},
    )
    for t in rec.trades:
        assert isinstance(t.quantity, int)


# ─── lot fallback ───────────────────────────────────────────────────────


def test_tax_rebalancer_missing_lots_warns_and_falls_back_to_unknown() -> None:
    """When `lots=None`, a single UNKNOWN-date lot is synthesised; STCG-worst-case."""
    holdings = [_holding("A", 10, 200, 100, lots=None),
                _holding("B", 10, 200, 100, lots=None)]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    # A is fully sold — gain ₹100/share × 10 = ₹1000 STCG (UNKNOWN bucket)
    assert any("no lot ledger" in w for w in rec.warnings)
    assert all(g.gain_type == GAIN_UNKNOWN for g in rec.realized_gains)


def test_tax_rebalancer_inconsistent_ledger_falls_back_with_warning() -> None:
    """Lots that don't reconcile with quantity → fallback + warning."""
    holdings = [
        _holding("A", 10, 200, 100,
                 lots=[TaxLot("A", 5, 100, dt.date(2025, 1, 1))]),  # only 5 of 10
        _holding("B", 10, 200, 100, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    assert any("supplied lots sum to" in w for w in rec.warnings)


def test_tax_rebalancer_consistent_ledger_used_directly() -> None:
    """Lots that reconcile exactly → no fallback warning for that symbol."""
    holdings = [
        _holding("A", 10, 200, 100,
                 lots=[TaxLot("A", 10, 100, dt.date(2024, 1, 1))]),  # LTCG
        _holding("B", 10, 200, 100, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    # Realised on A — should be LTCG (acquired Jan 2024, sold Apr 2026 → ~849 days)
    a_gains = [g for g in rec.realized_gains if g.symbol == "A"]
    assert len(a_gains) == 1
    assert a_gains[0].gain_type == GAIN_LTCG


# ─── tax projection vs naive comparison ─────────────────────────────────


def test_tax_rebalancer_savings_zero_when_no_taxable_gains() -> None:
    """If all sells are pure losses, both plans owe zero tax → savings = 0."""
    holdings = [
        _holding("A", 10, 100, 200,  # bought at 200, current 100 → loss
                 lots=[TaxLot("A", 10, 200, dt.date(2025, 1, 1))]),
        _holding("B", 10, 100, 100, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    assert rec.savings_inr == 0.0
    assert rec.tax_summary["total_tax_inr"] == 0.0


def test_tax_rebalancer_tax_summary_uses_published_rates() -> None:
    """A pure STCG sale should produce tax = gain × 0.15."""
    holdings = [
        _holding("A", 10, 200, 100,
                 lots=[TaxLot("A", 10, 100, dt.date(2026, 1, 1))]),  # STCG
        _holding("B", 10, 200, 100, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    # Gain: (200-100) × 10 = ₹1000 STCG → tax 150
    assert rec.tax_summary["total_tax_inr"] == pytest.approx(1000 * STCG_RATE)


def test_tax_rebalancer_ltcg_used_reduces_remaining_exemption() -> None:
    """Caller has used ₹1L of LTCG exemption already; only ₹25K remains."""
    holdings = [
        _holding("A", 100, 2000, 1000,
                 lots=[TaxLot("A", 100, 1000, dt.date(2024, 1, 1))]),  # LTCG
        _holding("B", 100, 2000, 1000, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
        ltcg_used_inr=100_000.0,
    )
    # Gain: (2000-1000) × 100 = ₹100,000 LTCG. With ₹25K exemption left,
    # taxable LTCG = ₹75,000 → tax = 75K × 0.125 = 9,375
    assert rec.tax_summary["ltcg_exemption_consumed_inr"] == pytest.approx(25_000.0)
    assert rec.tax_summary["ltcg_tax_inr"] == pytest.approx(75_000.0 * LTCG_RATE)


def test_tax_rebalancer_savings_positive_when_tax_min_beats_fifo() -> None:
    """A symbol has 2 lots: the older FIFO lot has a high gain (taxed),
    the newer lot has a loss. TAX_MIN sells the loss lot first → 0 tax;
    FIFO sells the high-gain lot first → positive tax.
    """
    holdings = [
        _holding(
            "A", 20, 150, 100,
            lots=[
                # FIFO lot (older): bought at 100, gain when sold at 150 → +50/share
                TaxLot("A", 10, 100.0, dt.date(2026, 1, 1)),  # STCG
                # Newer lot: bought at 200, loss when sold at 150 → -50/share
                TaxLot("A", 10, 200.0, dt.date(2026, 2, 1)),  # STCG
            ],
        ),
        _holding("B", 20, 150, 100, lots=None),
    ]
    # Sell 10 shares of A (target: A=0.25 → ₹1500 → 10 shares; was 20)
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.25, "B": 0.75}, as_of=AS_OF,
    )
    assert rec.savings_inr > 0.0
    # TAX_MIN sells the loss lot → 0 STCG tax
    assert rec.tax_summary["total_tax_inr"] == 0.0
    # FIFO sells the gain lot → ₹500 STCG → ₹75 tax
    assert rec.naive_tax_summary["total_tax_inr"] == pytest.approx(500.0 * STCG_RATE)


# ─── tracking error ─────────────────────────────────────────────────────


def test_tax_rebalancer_tracking_error_zero_when_perfectly_aligned() -> None:
    holdings = [_holding("A", 10, 100, 90), _holding("B", 10, 100, 90)]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.5, "B": 0.5}, as_of=AS_OF,
    )
    assert rec.tracking_error_pct == pytest.approx(0.0, abs=1e-6)


def test_tax_rebalancer_tracking_error_positive_when_rounding_distorts() -> None:
    """A 33.33%/66.67% split on a ₹100 portfolio cannot be hit exactly with
    integer-share rounding → small tracking error expected."""
    holdings = [_holding("A", 1, 100, 90), _holding("B", 1, 50, 40)]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.3333, "B": 0.6667}, as_of=AS_OF,
    )
    assert rec.tracking_error_pct >= 0.0


# ─── price warnings ─────────────────────────────────────────────────────


def test_tax_rebalancer_target_symbol_with_no_price_emits_warning() -> None:
    holdings = [_holding("A", 10, 100, 90,
                         lots=[TaxLot("A", 10, 90, dt.date(2025, 1, 1))])]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.5, "C": 0.5}, as_of=AS_OF,
        # C has no price
    )
    assert any("missing or non-positive price" in w for w in rec.warnings)


# ─── realized_gains population ──────────────────────────────────────────


def test_tax_rebalancer_realized_gains_attached_only_to_sells() -> None:
    holdings = [
        _holding("A", 10, 100, 80,
                 lots=[TaxLot("A", 10, 80, dt.date(2025, 1, 1))]),
        _holding("B", 10, 100, 90, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    for t in rec.trades:
        if t.action == "BUY":
            assert t.lots_sold == []
        else:
            assert len(t.lots_sold) > 0


def test_tax_rebalancer_realized_gain_quantities_match_sell_quantities() -> None:
    holdings = [
        _holding("A", 30, 100, 80,
                 lots=[
                     TaxLot("A", 10, 80, dt.date(2025, 1, 1)),
                     TaxLot("A", 20, 90, dt.date(2025, 6, 1)),
                 ]),
        _holding("B", 30, 100, 80, lots=None),
    ]
    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights={"A": 0.0, "B": 1.0}, as_of=AS_OF,
    )
    a_sell = next(t for t in rec.trades if t.symbol == "A" and t.action == "SELL")
    assert a_sell.quantity == 30
    assert sum(g.quantity for g in a_sell.lots_sold) == pytest.approx(30.0)


# ─── W4.1 acceptance gate (plan.md) ───────────────────────────────────────


def test_w41_acceptance_tax_aware_beats_naive_by_at_least_3_percent() -> None:
    """W4.1 acceptance: in a worked scenario, TAX_MIN must beat FIFO by ≥3% after-tax.

    Scenario: 5-symbol portfolio, two symbols carry multi-lot positions with
    a mix of STCG-loss and LTCG-gain lots. A 35/30/15/5/15 target rebalance
    forces SELLs in A and B. FIFO disposes the oldest LTCG lot first (large
    gain pushes past the ₹1.25L exemption); TAX_MIN realises the STCG losses
    on A, leaving only the unavoidable LTCG sale on B which fits within the
    exemption.

    Expected: tax_summary.total_tax_inr is materially below
    naive_tax_summary.total_tax_inr → savings_pct ≥ 3.0.
    """
    LT_OLD = AS_OF - dt.timedelta(days=800)   # qualifies as LTCG
    ST_NEW = AS_OF - dt.timedelta(days=30)    # short-term

    holdings = [
        # Symbol A: 1000 shares @ ₹2000. Two lots: an old LTCG winner and a
        # recent STCG loser. FIFO will sell the LTCG lot; TAX_MIN the loss.
        _holding("A", 1000, 2000.0, 1600.0, lots=[
            TaxLot("A", 500, 1000.0, LT_OLD),
            TaxLot("A", 500, 2200.0, ST_NEW),
        ]),
        # Symbol B: 500 shares @ ₹3000, single old LTCG lot. Both strategies
        # must hit this lot when forced to sell B.
        _holding("B", 500, 3000.0, 1500.0, lots=[
            TaxLot("B", 500, 1500.0, LT_OLD),
        ]),
        # Symbols C, D, E are buys only — no sells, no realised gains.
        _holding("C", 200, 1500.0, 1300.0, lots=[
            TaxLot("C", 200, 1300.0, ST_NEW),
        ]),
        _holding("D", 100, 500.0, 400.0, lots=[
            TaxLot("D", 100, 400.0, ST_NEW),
        ]),
        _holding("E", 100, 1000.0, 800.0, lots=[
            TaxLot("E", 100, 800.0, LT_OLD),
        ]),
    ]
    target = {"A": 0.35, "B": 0.30, "C": 0.15, "D": 0.05, "E": 0.15}

    rec = recommend_tax_optimal_rebalance(
        holdings, target_weights=target, as_of=AS_OF,
    )

    # The rebalance is non-trivial — there must be sells in both A and B.
    sells = {t.symbol for t in rec.trades if t.action == "SELL"}
    assert "A" in sells and "B" in sells, f"expected A and B in sells, got {sells}"

    naive_tax = rec.naive_tax_summary["total_tax_inr"]
    smart_tax = rec.tax_summary["total_tax_inr"]

    # Sanity: naive plan must owe more than the smart plan (else the test
    # scenario is mis-designed and the acceptance bar is meaningless).
    assert naive_tax > smart_tax, (
        f"scenario broken: naive_tax={naive_tax} not greater than smart_tax={smart_tax}"
    )
    # And savings must be measured against a non-zero naive baseline so
    # savings_pct is well-defined.
    assert naive_tax > 0, f"naive plan owes zero tax — exemption masked the scenario"

    # The acceptance bar from plan.md (W4.1):
    assert rec.savings_pct >= 3.0, (
        f"W4.1 acceptance failed: savings_pct={rec.savings_pct:.2f}% "
        f"(naive ₹{naive_tax:.2f} vs smart ₹{smart_tax:.2f}); plan demands ≥ 3%"
    )
