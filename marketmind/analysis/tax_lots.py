"""
MarketMind AI — Tax Lots (W4.1)

Lot-level position model + tax-aware lot-selection strategies.

A `TaxLot` is an immutable, single-symbol parcel acquired at one price on one
date. A position in a symbol is a list of TaxLots. When the rebalancer wants
to sell `q` shares of a symbol, `select_lots_to_sell` decides *which* lots
(or fragments thereof) to dispose of, under one of four strategies:

  • FIFO     — oldest lot first
  • LIFO     — newest lot first
  • HIFO     — highest cost-basis first  (typically minimises gains)
  • TAX_MIN  — minimise projected FY26 tax: realise losses first, then LTCG
               gains (12.5%), then STCG (15%); within ties prefer higher
               cost-basis.

Quantities may be fractional during solver iteration; rounding to integer
shares happens at the trade-emission boundary in `tax_rebalancer.py`.

The functions are pure: no I/O, no globals, no mutation of inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

from marketmind.analysis.tax_engine import (
    GAIN_LTCG,
    RealizedGain,
    classify_holding_period,
    realize_gain,
)

DateLike = Union[date, datetime]

# ─── Strategy tags ────────────────────────────────────────────────────────

SELECT_FIFO: str = "FIFO"
SELECT_LIFO: str = "LIFO"
SELECT_HIFO: str = "HIFO"
SELECT_TAX_MIN: str = "TAX_MIN"

VALID_STRATEGIES = frozenset({SELECT_FIFO, SELECT_LIFO, SELECT_HIFO, SELECT_TAX_MIN})

# Float-fuzz threshold for "fully consumed" / "any leftover".
# Smaller than 1 share, larger than typical SLSQP residuals.
_QTY_EPS: float = 1e-9


# ─── Lot model ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaxLot:
    """A parcel acquired at one price on one date.

    `acquisition_date` may be None — Kite's holdings() does not return per-lot
    acquisition timestamps, so the rebalancer's fallback synthesises a
    single-lot position from `average_price × quantity` with date=None.
    Such lots are bucketed as STCG-worst-case by `tax_engine.compute_tax`.
    """

    symbol: str
    quantity: float
    cost_basis: float
    acquisition_date: Optional[DateLike] = None

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(
                f"TaxLot quantity must be non-negative, got {self.quantity}"
            )
        if self.cost_basis < 0:
            raise ValueError(
                f"TaxLot cost_basis must be non-negative, got {self.cost_basis}"
            )


# ─── Sort keys ────────────────────────────────────────────────────────────


def _date_sort_key(lot: TaxLot) -> Tuple[bool, date]:
    """Sort key for date-based strategies. Unknown dates sort *last* in FIFO
    (so explicit dates dispose first when present); reversed for LIFO."""
    if lot.acquisition_date is None:
        return (True, date.min)
    d = lot.acquisition_date
    return (False, d.date() if isinstance(d, datetime) else d)


def _tax_min_key(lot: TaxLot, sell_price: float, as_of: DateLike) -> Tuple[int, float, float]:
    """Sort key for TAX_MIN. Lower keys are sold first.

    Bucket order:
      0 — losses   (sell first; deeper loss → smaller key → first)
      1 — LTCG gains  (12.5%)
      2 — STCG / UNKNOWN gains  (15%)

    Tiebreaker within bucket: higher cost-basis first → smaller realised gain.
    """
    _, kind = classify_holding_period(lot.acquisition_date, as_of)
    gain_per_share = sell_price - lot.cost_basis

    if gain_per_share < 0:
        # Deeper loss first: ascending gain_per_share (most negative is smallest).
        return (0, gain_per_share, 0.0)

    bucket = 1 if kind == GAIN_LTCG else 2
    # Higher cost-basis first → ascending negative cost-basis.
    return (bucket, -lot.cost_basis, 0.0)


def _sort_indexed(
    indexed: List[Tuple[int, TaxLot]],
    strategy: str,
    sell_price: float,
    as_of: DateLike,
) -> List[Tuple[int, TaxLot]]:
    """Sort `(index, lot)` pairs by strategy. Returns a new list."""
    if strategy == SELECT_FIFO:
        return sorted(indexed, key=lambda p: _date_sort_key(p[1]))
    if strategy == SELECT_LIFO:
        # Reverse: newest first; unknown-date lots sort last in both directions
        # because they convey no chronological signal.
        return sorted(
            indexed,
            key=lambda p: (p[1].acquisition_date is None,
                            -((_date_sort_key(p[1])[1]).toordinal())),
        )
    if strategy == SELECT_HIFO:
        return sorted(indexed, key=lambda p: -p[1].cost_basis)
    if strategy == SELECT_TAX_MIN:
        return sorted(
            indexed, key=lambda p: _tax_min_key(p[1], sell_price, as_of)
        )
    raise ValueError(f"unknown lot-selection strategy: {strategy!r}")


# ─── Public API ───────────────────────────────────────────────────────────


def select_lots_to_sell(
    lots: List[TaxLot],
    quantity_to_sell: float,
    sell_price: float,
    as_of: DateLike,
    strategy: str = SELECT_TAX_MIN,
) -> Tuple[List[RealizedGain], List[TaxLot]]:
    """Pick lots to satisfy `quantity_to_sell` under `strategy`.

    Returns (realized_gains, remaining_lots) where:
      • realized_gains — list of partial-or-whole lot dispositions, in the
        order they were consumed (i.e. strategy order).
      • remaining_lots — lots NOT sold, plus the trimmed fragment of any
        partially-sold lot, preserved in the *original* input ordering.

    Raises:
      ValueError — if quantity_to_sell is negative, if the input list spans
      multiple symbols, if requested quantity exceeds total available, or if
      `strategy` is not one of FIFO/LIFO/HIFO/TAX_MIN.
    """
    if quantity_to_sell < 0:
        raise ValueError(
            f"quantity_to_sell must be non-negative, got {quantity_to_sell}"
        )
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"unknown lot-selection strategy: {strategy!r}")

    if not lots:
        if quantity_to_sell > _QTY_EPS:
            raise ValueError(
                f"insufficient quantity: requested {quantity_to_sell}, "
                f"available 0 (no lots provided)"
            )
        return [], []

    symbols = {l.symbol for l in lots}
    if len(symbols) > 1:
        raise ValueError(
            f"select_lots_to_sell expects a single symbol, got {sorted(symbols)}"
        )

    total_available = sum(l.quantity for l in lots)
    if quantity_to_sell > total_available + _QTY_EPS:
        raise ValueError(
            f"insufficient quantity: requested {quantity_to_sell}, "
            f"available {total_available}"
        )

    if quantity_to_sell <= _QTY_EPS:
        return [], list(lots)

    indexed = list(enumerate(lots))
    sorted_indexed = _sort_indexed(indexed, strategy, sell_price, as_of)

    consumed_by_idx: Dict[int, float] = {}
    remaining = quantity_to_sell
    realized: List[RealizedGain] = []

    for idx, lot in sorted_indexed:
        if remaining <= _QTY_EPS:
            break
        take = min(lot.quantity, remaining)
        if take <= 0:
            continue
        realized.append(realize_gain(
            symbol=lot.symbol,
            quantity=take,
            cost_basis=lot.cost_basis,
            sell_price=sell_price,
            acquisition_date=lot.acquisition_date,
            as_of=as_of,
        ))
        consumed_by_idx[idx] = take
        remaining -= take

    # Reconstruct remaining_lots in original input order.
    remaining_lots: List[TaxLot] = []
    for idx, orig in enumerate(lots):
        used = consumed_by_idx.get(idx, 0.0)
        leftover = orig.quantity - used
        if leftover > _QTY_EPS:
            remaining_lots.append(replace(orig, quantity=leftover))

    return realized, remaining_lots
