"""
MarketMind AI — Tax-aware Rebalancer (W4.1)

Compose `tax_engine` + `tax_lots` into a recommend-trades engine.

Inputs:
  • `holdings`         — list of CurrentHolding (qty, last_price, avg_cost,
                         optional lot ledger).
  • `target_weights`   — dict[sym, weight] summing to ~1.0; weights drive
                         integer-share targets.
  • `as_of`            — date used for STCG / LTCG classification.
  • `ltcg_used_inr`    — LTCG already realised this FY (subtracts from
                         the ₹1.25L exemption).

Outputs `RebalanceRecommendation` with:
  • A trade list (BUY/SELL with notional, lot dispositions on sells).
  • A tax projection under the tax-aware (TAX_MIN) plan.
  • A naive (FIFO) projection for comparison → savings_inr / savings_pct.
  • Tracking error vs target weights (%).
  • A warnings list for missing lot ledgers, missing prices, weights drift.

The engine assumes self-financing (no external cash injection); total
portfolio value is the current mark-to-market sum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

from marketmind.analysis.tax_engine import (
    RealizedGain,
    TaxBill,
    classify_holding_period,
    compute_tax,
)
from marketmind.analysis.tax_lots import (
    SELECT_FIFO,
    SELECT_TAX_MIN,
    TaxLot,
    select_lots_to_sell,
)

DateLike = Union[date, datetime]

# Half-share threshold below which a delta is treated as "no trade".
_DELTA_EPS_SHARES: float = 0.5
# Lot-reconciliation tolerance (Kite quantities are integer; allow tiny float fuzz).
_LOT_RECONCILE_EPS: float = 1e-6
# Acceptable deviation of sum(target_weights) from 1.0.
_WEIGHT_SUM_TOL: float = 0.01


# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CurrentHolding:
    """Current position in one symbol.

    `lots` is optional. When None or unreconcileable, the rebalancer
    synthesises a single TaxLot at `avg_cost` with `acquisition_date=None`,
    which the tax engine buckets as STCG worst-case (15%) and the
    recommendation surfaces a warning.
    """

    symbol: str
    quantity: float
    last_price: float
    avg_cost: float
    lots: Optional[List[TaxLot]] = None

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(f"holding quantity must be non-negative, got {self.quantity}")
        if self.last_price < 0 or self.avg_cost < 0:
            raise ValueError("prices must be non-negative")


@dataclass(frozen=True)
class TradeRec:
    symbol: str
    action: str            # 'BUY' | 'SELL'
    quantity: int
    est_price: float
    notional_inr: float
    lots_sold: List[RealizedGain] = field(default_factory=list)


@dataclass(frozen=True)
class HarvestCandidate:
    """A lot currently underwater — eligible for tax-loss harvesting.

    `est_loss_inr` is reported as a POSITIVE rupee amount (the magnitude of
    the unrealised loss). Harvesting realises this loss to offset other gains
    in the same fiscal year.
    """

    symbol: str
    quantity: float
    cost_basis: float
    current_price: float
    est_loss_inr: float
    holding_period_days: Optional[int]
    gain_type: str
    advisory: str = (
        "Avoid re-purchase within 30 days to keep the loss recognition robust"
        " against future wash-sale-style scrutiny."
    )


@dataclass(frozen=True)
class RebalanceRecommendation:
    trades: List[TradeRec]
    realized_gains: List[RealizedGain]
    tax_summary: Dict[str, float]       # tax-aware (TAX_MIN) plan
    naive_tax_summary: Dict[str, float]  # FIFO baseline
    savings_inr: float
    savings_pct: float
    tracking_error_pct: float
    harvest_candidates: List[HarvestCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ─── Internals ────────────────────────────────────────────────────────────


def _materialise_lots(h: CurrentHolding) -> Tuple[List[TaxLot], Optional[str]]:
    """Return (lots, warning_or_None). Falls back to a single UNKNOWN-date
    lot when the supplied ledger is absent or inconsistent with `quantity`."""
    if h.lots:
        ledger_qty = sum(l.quantity for l in h.lots)
        if abs(ledger_qty - h.quantity) <= _LOT_RECONCILE_EPS:
            return list(h.lots), None
        warn = (
            f"{h.symbol}: supplied lots sum to {ledger_qty:.4f} but holding "
            f"quantity is {h.quantity:.4f}; falling back to UNKNOWN single-lot"
        )
    else:
        warn = f"{h.symbol}: no lot ledger supplied — taxed as STCG worst-case"
    fallback = TaxLot(
        symbol=h.symbol, quantity=h.quantity, cost_basis=h.avg_cost,
        acquisition_date=None,
    )
    return [fallback], warn


def _bill_to_dict(bill: TaxBill) -> Dict[str, float]:
    return {
        "stcg_realized_inr": bill.stcg_realized_inr,
        "ltcg_realized_inr": bill.ltcg_realized_inr,
        "ltcg_taxable_inr": bill.ltcg_taxable_inr,
        "stcg_tax_inr": bill.stcg_tax_inr,
        "ltcg_tax_inr": bill.ltcg_tax_inr,
        "total_tax_inr": bill.total_tax_inr,
        "ltcg_exemption_consumed_inr": bill.ltcg_exemption_consumed_inr,
    }


def _compute_target_quantities(
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    total_value: float,
    warnings: List[str],
) -> Dict[str, int]:
    """Convert target weights → integer-share targets at current prices.

    Symbols with no available price are skipped with a warning."""
    target_qty: Dict[str, int] = {}
    for sym, w in target_weights.items():
        px = prices.get(sym, 0.0)
        if px <= 0:
            warnings.append(f"{sym}: missing or non-positive price; skipped")
            continue
        target_qty[sym] = int(round((w * total_value) / px))
    return target_qty


def _tracking_error_pct(
    target_qty: Dict[str, int],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
) -> float:
    """Half-sum-of-absolute-deviations between rounded actual and target weights."""
    realised_value = sum(target_qty.get(s, 0) * prices.get(s, 0.0) for s in target_qty)
    if realised_value <= 0:
        return 0.0
    actual_w = {
        s: (target_qty.get(s, 0) * prices.get(s, 0.0)) / realised_value
        for s in target_weights
    }
    deviation = sum(abs(actual_w.get(s, 0.0) - target_weights[s]) for s in target_weights)
    return (deviation / 2.0) * 100.0


# ─── Public API ───────────────────────────────────────────────────────────


def find_harvest_candidates(
    holdings: List[CurrentHolding],
    as_of: DateLike,
    min_loss_inr: float = 1_000.0,
    min_loss_pct: float = 5.0,
) -> List[HarvestCandidate]:
    """Identify lots currently underwater above two thresholds.

    A lot qualifies when:
      • current_price < cost_basis (unrealised loss),
      • |loss_inr| ≥ min_loss_inr,
      • |loss_pct| ≥ min_loss_pct (relative to cost-basis).

    Lots are walked at the LOT level (not aggregated per holding), so a
    holding with a profitable old lot and a loss-bearing new lot will only
    surface the loss-bearing lot. Holdings without a ledger fall back to a
    single UNKNOWN-date lot via `_materialise_lots`.

    Returned in descending order of `est_loss_inr` (deepest first).
    """
    if min_loss_inr < 0 or min_loss_pct < 0:
        raise ValueError(
            f"thresholds must be non-negative: "
            f"min_loss_inr={min_loss_inr}, min_loss_pct={min_loss_pct}"
        )

    out: List[HarvestCandidate] = []
    for h in holdings:
        if h.last_price <= 0:
            continue
        lots, _ = _materialise_lots(h)
        for lot in lots:
            if lot.cost_basis <= 0:
                continue
            loss_per_share = lot.cost_basis - h.last_price
            if loss_per_share <= 0:
                continue
            est_loss = loss_per_share * lot.quantity
            if est_loss < min_loss_inr:
                continue
            loss_pct = (loss_per_share / lot.cost_basis) * 100.0
            if loss_pct < min_loss_pct:
                continue
            days, kind = classify_holding_period(lot.acquisition_date, as_of)
            out.append(HarvestCandidate(
                symbol=h.symbol,
                quantity=lot.quantity,
                cost_basis=lot.cost_basis,
                current_price=h.last_price,
                est_loss_inr=est_loss,
                holding_period_days=days,
                gain_type=kind,
            ))

    out.sort(key=lambda c: -c.est_loss_inr)
    return out


def recommend_tax_optimal_rebalance(
    holdings: List[CurrentHolding],
    target_weights: Dict[str, float],
    as_of: DateLike,
    ltcg_used_inr: float = 0.0,
    new_symbol_prices: Optional[Dict[str, float]] = None,
    selection_strategy: str = SELECT_TAX_MIN,
    naive_strategy: str = SELECT_FIFO,
    harvest_losses: bool = False,
    harvest_min_loss_inr: float = 1_000.0,
    harvest_min_loss_pct: float = 5.0,
) -> RebalanceRecommendation:
    """Build the trade list + tax projection for a self-financing rebalance.

    `target_weights` must sum to ~1.0 (within ±1%). New symbols (in target
    but not currently held) require a price in `new_symbol_prices`.

    Returns a recommendation with both the TAX_MIN plan and an FIFO-naive
    comparison; `savings_inr` is `naive_total_tax − tax_min_total_tax`.
    """
    weight_sum = sum(target_weights.values())
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(
            f"target_weights must sum to ~1.0 (±{_WEIGHT_SUM_TOL}), got {weight_sum:.4f}"
        )

    new_symbol_prices = new_symbol_prices or {}
    warnings: List[str] = []

    # Build price + qty maps; new symbols inherit price from new_symbol_prices.
    prices: Dict[str, float] = {h.symbol: h.last_price for h in holdings}
    for sym, px in new_symbol_prices.items():
        prices.setdefault(sym, px)
    current_qty: Dict[str, float] = {h.symbol: h.quantity for h in holdings}
    holdings_by_sym: Dict[str, CurrentHolding] = {h.symbol: h for h in holdings}

    # Symbols in target but absent from prices map → skipped with warning
    for sym in target_weights:
        if sym not in prices or prices[sym] <= 0:
            warnings.append(f"{sym}: missing or non-positive price; skipped")

    total_value = sum(h.quantity * h.last_price for h in holdings)
    target_qty = _compute_target_quantities(
        target_weights, prices, total_value, warnings=[],  # warnings handled above
    )

    trades: List[TradeRec] = []
    realized: List[RealizedGain] = []
    naive_realized: List[RealizedGain] = []
    universe = set(current_qty) | set(target_qty)

    for sym in sorted(universe):
        cur = current_qty.get(sym, 0.0)
        tgt = float(target_qty.get(sym, 0))
        delta = tgt - cur
        px = prices.get(sym, 0.0)

        if abs(delta) < _DELTA_EPS_SHARES or px <= 0:
            continue

        if delta < 0:
            qty_to_sell = -delta
            holding = holdings_by_sym.get(sym)
            if holding is None:
                # Should not happen — sells imply a current holding — but be defensive.
                continue
            lots, warn = _materialise_lots(holding)
            if warn is not None:
                warnings.append(warn)
            tax_min_realized, _ = select_lots_to_sell(
                lots, qty_to_sell, sell_price=px, as_of=as_of,
                strategy=selection_strategy,
            )
            naive_lots_realized, _ = select_lots_to_sell(
                lots, qty_to_sell, sell_price=px, as_of=as_of,
                strategy=naive_strategy,
            )
            realized.extend(tax_min_realized)
            naive_realized.extend(naive_lots_realized)
            trades.append(TradeRec(
                symbol=sym, action="SELL",
                quantity=int(round(qty_to_sell)),
                est_price=px,
                notional_inr=qty_to_sell * px,
                lots_sold=tax_min_realized,
            ))
        else:
            trades.append(TradeRec(
                symbol=sym, action="BUY",
                quantity=int(round(delta)),
                est_price=px,
                notional_inr=delta * px,
                lots_sold=[],
            ))

    tax_bill = compute_tax(realized, ltcg_used_inr=ltcg_used_inr)
    naive_bill = compute_tax(naive_realized, ltcg_used_inr=ltcg_used_inr)

    savings_inr = naive_bill.total_tax_inr - tax_bill.total_tax_inr
    savings_pct = (
        (savings_inr / naive_bill.total_tax_inr * 100.0)
        if naive_bill.total_tax_inr > 0 else 0.0
    )

    harvest_candidates: List[HarvestCandidate] = []
    if harvest_losses:
        harvest_candidates = find_harvest_candidates(
            holdings, as_of=as_of,
            min_loss_inr=harvest_min_loss_inr,
            min_loss_pct=harvest_min_loss_pct,
        )

    return RebalanceRecommendation(
        trades=trades,
        realized_gains=realized,
        tax_summary=_bill_to_dict(tax_bill),
        naive_tax_summary=_bill_to_dict(naive_bill),
        savings_inr=savings_inr,
        savings_pct=savings_pct,
        tracking_error_pct=_tracking_error_pct(target_qty, target_weights, prices),
        harvest_candidates=harvest_candidates,
        warnings=warnings,
    )
