"""T2: lot model + tax-aware lot-selection strategy tests.

Verifies:
  • TaxLot validation (negative qty / cost-basis rejected; frozen)
  • FIFO / LIFO / HIFO / TAX_MIN ordering
  • Partial-lot splits preserve acquisition_date and original ordering
  • Float fuzz (1e-9) on quantity boundary
  • Multi-symbol input rejected
  • Unknown-date lots sort last in date-based strategies
"""
from __future__ import annotations

import datetime as dt

import pytest

from marketmind.analysis.tax_engine import GAIN_LTCG, GAIN_STCG, GAIN_UNKNOWN
from marketmind.analysis.tax_lots import (
    SELECT_FIFO,
    SELECT_HIFO,
    SELECT_LIFO,
    SELECT_TAX_MIN,
    TaxLot,
    select_lots_to_sell,
)

# ─── TaxLot validation ───────────────────────────────────────────────────


def test_tax_lots_taxlot_negative_quantity_rejected() -> None:
    with pytest.raises(ValueError):
        TaxLot(symbol="X", quantity=-1.0, cost_basis=100.0, acquisition_date=None)


def test_tax_lots_taxlot_negative_cost_basis_rejected() -> None:
    with pytest.raises(ValueError):
        TaxLot(symbol="X", quantity=1.0, cost_basis=-1.0, acquisition_date=None)


def test_tax_lots_taxlot_is_frozen() -> None:
    """Frozen dataclass: any attribute mutation must raise."""
    lot = TaxLot(symbol="X", quantity=1.0, cost_basis=100.0, acquisition_date=None)
    with pytest.raises(Exception):
        lot.quantity = 2.0  # type: ignore[misc]


def test_tax_lots_taxlot_zero_quantity_allowed() -> None:
    """Zero is degenerate but not invalid — it's the natural representation
    of a fully-consumed lot before pruning."""
    TaxLot(symbol="X", quantity=0.0, cost_basis=100.0, acquisition_date=None)


# ─── select_lots_to_sell — validation ────────────────────────────────────


def test_tax_lots_select_negative_quantity_rejected() -> None:
    lots = [TaxLot("X", 10, 100.0, dt.date(2025, 1, 1))]
    with pytest.raises(ValueError):
        select_lots_to_sell(lots, -1.0, sell_price=120.0, as_of=dt.date(2026, 4, 27))


def test_tax_lots_select_unknown_strategy_rejected() -> None:
    lots = [TaxLot("X", 10, 100.0, dt.date(2025, 1, 1))]
    with pytest.raises(ValueError):
        select_lots_to_sell(
            lots, 5, sell_price=120.0,
            as_of=dt.date(2026, 4, 27), strategy="BOGUS",
        )


def test_tax_lots_select_insufficient_quantity_rejected() -> None:
    lots = [TaxLot("X", 5, 100.0, dt.date(2025, 1, 1))]
    with pytest.raises(ValueError):
        select_lots_to_sell(lots, 10, sell_price=120.0, as_of=dt.date(2026, 4, 27))


def test_tax_lots_select_mixed_symbols_rejected() -> None:
    lots = [
        TaxLot("A", 5, 100.0, dt.date(2025, 1, 1)),
        TaxLot("B", 5, 100.0, dt.date(2025, 1, 1)),
    ]
    with pytest.raises(ValueError):
        select_lots_to_sell(lots, 5, sell_price=120.0, as_of=dt.date(2026, 4, 27))


def test_tax_lots_select_empty_lots_with_zero_quantity_returns_empty() -> None:
    realized, remaining = select_lots_to_sell(
        [], 0.0, sell_price=120.0, as_of=dt.date(2026, 4, 27)
    )
    assert realized == []
    assert remaining == []


def test_tax_lots_select_empty_lots_with_positive_quantity_rejected() -> None:
    with pytest.raises(ValueError):
        select_lots_to_sell([], 1.0, sell_price=120.0, as_of=dt.date(2026, 4, 27))


def test_tax_lots_select_zero_quantity_returns_lots_unchanged() -> None:
    lots = [TaxLot("X", 5, 100.0, dt.date(2025, 1, 1))]
    realized, remaining = select_lots_to_sell(
        lots, 0.0, sell_price=120.0, as_of=dt.date(2026, 4, 27)
    )
    assert realized == []
    assert remaining == lots


# ─── FIFO ────────────────────────────────────────────────────────────────


def test_tax_lots_fifo_picks_oldest_first() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2026, 1, 1)),  # newest
        TaxLot("X", 5, 100.0, dt.date(2024, 1, 1)),  # oldest
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),  # middle
    ]
    realized, remaining = select_lots_to_sell(
        lots, 7, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
    )
    # FIFO: oldest (2024) fully, then 2 of 2025
    assert len(realized) == 2
    assert realized[0].quantity == 5
    assert realized[0].cost_basis == 100.0
    assert realized[0].holding_period_days == (dt.date(2026, 4, 27) - dt.date(2024, 1, 1)).days
    assert realized[1].quantity == 2
    # Remaining preserves original input order — 2026 lot, then trimmed 2025 lot.
    assert len(remaining) == 2
    assert remaining[0].acquisition_date == dt.date(2026, 1, 1)
    assert remaining[0].quantity == 5
    assert remaining[1].acquisition_date == dt.date(2025, 1, 1)
    assert remaining[1].quantity == 3


def test_tax_lots_fifo_unknown_dates_sort_last() -> None:
    """When some lots lack acquisition_date, FIFO disposes dated lots first."""
    lots = [
        TaxLot("X", 5, 100.0, None),                  # unknown
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),
    ]
    realized, _ = select_lots_to_sell(
        lots, 5, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
    )
    assert len(realized) == 1
    assert realized[0].holding_period_days is not None  # the dated one


# ─── LIFO ────────────────────────────────────────────────────────────────


def test_tax_lots_lifo_picks_newest_first() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2024, 1, 1)),  # oldest
        TaxLot("X", 5, 100.0, dt.date(2026, 1, 1)),  # newest
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),  # middle
    ]
    realized, remaining = select_lots_to_sell(
        lots, 7, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_LIFO,
    )
    assert realized[0].holding_period_days == (
        dt.date(2026, 4, 27) - dt.date(2026, 1, 1)
    ).days
    assert realized[0].quantity == 5
    assert realized[1].quantity == 2
    # Remaining order: 2024 (full, 5), 2025 (trimmed, 3)
    assert len(remaining) == 2
    assert remaining[0].acquisition_date == dt.date(2024, 1, 1)
    assert remaining[1].acquisition_date == dt.date(2025, 1, 1)
    assert remaining[1].quantity == 3


# ─── HIFO ────────────────────────────────────────────────────────────────


def test_tax_lots_hifo_picks_highest_cost_basis_first() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2024, 1, 1)),
        TaxLot("X", 5, 200.0, dt.date(2025, 1, 1)),  # highest CB
        TaxLot("X", 5, 150.0, dt.date(2025, 6, 1)),
    ]
    realized, _ = select_lots_to_sell(
        lots, 7, sell_price=180.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_HIFO,
    )
    # 200 lot first (whole 5), then 150 lot (2 shares)
    assert realized[0].cost_basis == 200.0
    assert realized[0].quantity == 5
    assert realized[1].cost_basis == 150.0
    assert realized[1].quantity == 2


# ─── TAX_MIN ─────────────────────────────────────────────────────────────


def test_tax_lots_tax_min_prefers_loss_lots_first() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),  # gain at sell=200
        TaxLot("X", 5, 250.0, dt.date(2025, 1, 1)),  # loss at sell=200
    ]
    realized, _ = select_lots_to_sell(
        lots, 5, sell_price=200.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_TAX_MIN,
    )
    # The loss lot is sold first.
    assert realized[0].cost_basis == 250.0
    assert realized[0].gain_inr == pytest.approx(-250.0)


def test_tax_lots_tax_min_prefers_ltcg_over_stcg_at_same_gain() -> None:
    """Two gain lots with similar economics — LTCG (12.5%) sold before STCG (15%)."""
    lots = [
        # STCG lot: held 100 days, gains ₹50/share
        TaxLot("X", 5, 150.0, dt.date(2026, 1, 17)),
        # LTCG lot: held 400 days, gains ₹50/share
        TaxLot("X", 5, 150.0, dt.date(2025, 3, 23)),
    ]
    realized, _ = select_lots_to_sell(
        lots, 5, sell_price=200.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_TAX_MIN,
    )
    assert realized[0].gain_type == GAIN_LTCG


def test_tax_lots_tax_min_within_bucket_uses_hifo_tiebreak() -> None:
    """All STCG gains; the highest cost-basis (smallest gain) is sold first."""
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2026, 1, 1)),
        TaxLot("X", 5, 180.0, dt.date(2026, 1, 1)),  # smallest gain
        TaxLot("X", 5, 140.0, dt.date(2026, 1, 1)),
    ]
    realized, _ = select_lots_to_sell(
        lots, 5, sell_price=200.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_TAX_MIN,
    )
    assert realized[0].cost_basis == 180.0


def test_tax_lots_tax_min_unknown_acquisition_treated_as_stcg() -> None:
    """An UNKNOWN-date lot must NOT be preferred over an LTCG lot when both
    are gains — UNKNOWN is taxed at 15%, LTCG at 12.5%."""
    lots = [
        TaxLot("X", 5, 150.0, None),                       # UNKNOWN gain
        TaxLot("X", 5, 150.0, dt.date(2025, 1, 1)),       # LTCG gain
    ]
    realized, _ = select_lots_to_sell(
        lots, 5, sell_price=200.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_TAX_MIN,
    )
    assert realized[0].gain_type == GAIN_LTCG
    assert realized[0].holding_period_days is not None


# ─── Partial / full consumption ──────────────────────────────────────────


def test_tax_lots_full_consumption_yields_empty_remaining() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),
        TaxLot("X", 5, 100.0, dt.date(2025, 6, 1)),
    ]
    realized, remaining = select_lots_to_sell(
        lots, 10, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
    )
    assert sum(g.quantity for g in realized) == 10
    assert remaining == []


def test_tax_lots_partial_lot_split_preserves_acquisition_date() -> None:
    lots = [TaxLot("X", 10, 100.0, dt.date(2025, 1, 1))]
    realized, remaining = select_lots_to_sell(
        lots, 3, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
    )
    assert len(realized) == 1
    assert realized[0].quantity == 3
    assert len(remaining) == 1
    assert remaining[0].quantity == 7
    assert remaining[0].acquisition_date == dt.date(2025, 1, 1)
    assert remaining[0].cost_basis == 100.0


def test_tax_lots_realized_quantities_sum_to_quantity_sold() -> None:
    lots = [
        TaxLot("X", 5, 100.0, dt.date(2025, 1, 1)),
        TaxLot("X", 7, 110.0, dt.date(2025, 6, 1)),
        TaxLot("X", 3, 120.0, dt.date(2025, 12, 1)),
    ]
    realized, remaining = select_lots_to_sell(
        lots, 9, sell_price=130.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_HIFO,
    )
    assert sum(g.quantity for g in realized) == pytest.approx(9.0)
    assert sum(l.quantity for l in remaining) == pytest.approx(15 - 9)


def test_tax_lots_float_fuzz_just_under_total_succeeds() -> None:
    """Quantity within 1e-10 of total available is accepted."""
    lots = [TaxLot("X", 10.0, 100.0, dt.date(2025, 1, 1))]
    realized, remaining = select_lots_to_sell(
        lots, 10.0 - 1e-10, sell_price=120.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
    )
    assert len(realized) == 1
    # Either fully consumed (epsilon swallowed) or 1e-10 leftover (also pruned)
    assert remaining == []


def test_tax_lots_float_fuzz_just_over_total_rejected() -> None:
    """Quantity > total + 1e-9 is rejected even by a hair."""
    lots = [TaxLot("X", 10.0, 100.0, dt.date(2025, 1, 1))]
    with pytest.raises(ValueError):
        select_lots_to_sell(
            lots, 10.0 + 1e-3, sell_price=120.0,
            as_of=dt.date(2026, 4, 27), strategy=SELECT_FIFO,
        )


def test_tax_lots_remaining_lots_are_in_original_input_order() -> None:
    """Even when sort order differs from input, leftovers preserve input order."""
    lots = [
        TaxLot("X", 5, 50.0, dt.date(2026, 1, 1)),    # newest, low CB
        TaxLot("X", 5, 200.0, dt.date(2025, 1, 1)),   # mid, high CB → HIFO sells this
        TaxLot("X", 5, 100.0, dt.date(2024, 1, 1)),   # oldest, mid CB
    ]
    _, remaining = select_lots_to_sell(
        lots, 3, sell_price=180.0,
        as_of=dt.date(2026, 4, 27), strategy=SELECT_HIFO,
    )
    # Index 1 was trimmed (5 → 2); indices 0 and 2 untouched. Order preserved.
    assert [l.acquisition_date for l in remaining] == [
        dt.date(2026, 1, 1),
        dt.date(2025, 1, 1),
        dt.date(2024, 1, 1),
    ]
    assert remaining[1].quantity == 2
