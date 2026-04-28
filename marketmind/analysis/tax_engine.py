"""
MarketMind AI — Tax Engine (W4.1)

Pure-function FY26 Indian listed-equity capital-gains rules.

Constants reflect the FY26 spec stated in plan.md (W4.1):
  • STCG 15%   for holding period ≤ 365 days
  • LTCG 12.5% for holding period >  365 days, with ₹1,25,000 exemption per
    fiscal year applied across ALL LTCG (not per security).

Loss treatment is intentionally simplified — IT Act §70/§71 carry-forward and
intra-head set-off are out of scope for the in-session rebalancer; we only
zero-out a negative net bucket rather than offsetting against the other.
This is conservative for the tax estimate: it never under-states tax owed.

The engine is stateless. All inputs flow through dataclasses; no I/O, no
globals, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Tuple, Union

# ─── FY26 constants ────────────────────────────────────────────────────────

STCG_RATE: float = 0.15
LTCG_RATE: float = 0.125
LTCG_EXEMPTION_INR: float = 125_000.0
LT_HOLDING_DAYS: int = 365  # > LT_HOLDING_DAYS qualifies as long-term

# Gain-type tags. UNKNOWN is used when acquisition_date is missing — Kite's
# holdings() endpoint returns only a weighted average price, no per-lot date,
# so the rebalancer falls back to STCG-worst-case and emits a warning.
GAIN_STCG = "STCG"
GAIN_LTCG = "LTCG"
GAIN_UNKNOWN = "UNKNOWN"

DateLike = Union[date, datetime]


# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RealizedGain:
    """A single (partial or whole) lot disposition.

    `gain_inr` is signed: negative = realised loss.
    `holding_period_days` is None when acquisition date was unknown.
    `gain_type` is one of GAIN_STCG / GAIN_LTCG / GAIN_UNKNOWN.
    UNKNOWN gains are treated as STCG by `compute_tax`.
    """

    symbol: str
    quantity: float
    cost_basis: float          # per-share INR at acquisition
    sell_price: float          # per-share INR at sale
    holding_period_days: Optional[int]
    gain_inr: float
    gain_type: str


@dataclass(frozen=True)
class TaxBill:
    """Aggregated tax summary for a set of realised gains.

    `*_realized_inr` are net signed amounts (gains − losses) per bucket.
    `*_tax_inr` are the rupee-amounts owed at FY26 rates after applying the
    LTCG exemption. Negative net buckets are clamped to zero for tax
    purposes (see module docstring on §70/§71 simplification).
    """

    stcg_realized_inr: float
    ltcg_realized_inr: float
    ltcg_taxable_inr: float
    stcg_tax_inr: float
    ltcg_tax_inr: float
    total_tax_inr: float
    ltcg_exemption_consumed_inr: float


# ─── Pure helpers ─────────────────────────────────────────────────────────


def _to_date(d: DateLike) -> date:
    return d.date() if isinstance(d, datetime) else d


def classify_holding_period(
    acquisition_date: Optional[DateLike],
    as_of: DateLike,
) -> Tuple[Optional[int], str]:
    """Return (holding_period_days, gain_type).

    Boundary: holding period of *exactly* LT_HOLDING_DAYS (365) is STCG;
    LTCG begins at day 366 (i.e. `days > LT_HOLDING_DAYS`).
    """
    if acquisition_date is None:
        return None, GAIN_UNKNOWN

    acq = _to_date(acquisition_date)
    asof = _to_date(as_of)
    days = (asof - acq).days
    if days < 0:
        raise ValueError(
            f"as_of {asof.isoformat()} predates acquisition "
            f"{acq.isoformat()}: holding period is negative"
        )
    return days, (GAIN_LTCG if days > LT_HOLDING_DAYS else GAIN_STCG)


def realize_gain(
    symbol: str,
    quantity: float,
    cost_basis: float,
    sell_price: float,
    acquisition_date: Optional[DateLike],
    as_of: DateLike,
) -> RealizedGain:
    """Materialise a single lot sale into a RealizedGain record.

    Quantity must be non-negative (a "negative sale" is a buy — not modelled).
    Loss is expressed as negative `gain_inr`.
    """
    if quantity < 0:
        raise ValueError(f"quantity must be non-negative, got {quantity}")

    days, kind = classify_holding_period(acquisition_date, as_of)
    gain = (sell_price - cost_basis) * quantity
    return RealizedGain(
        symbol=symbol,
        quantity=quantity,
        cost_basis=cost_basis,
        sell_price=sell_price,
        holding_period_days=days,
        gain_inr=gain,
        gain_type=kind,
    )


def compute_tax(
    gains: Iterable[RealizedGain],
    ltcg_used_inr: float = 0.0,
) -> TaxBill:
    """Aggregate `gains` and apply FY26 STCG/LTCG rates.

    `ltcg_used_inr` lets the caller account for LTCG already realised earlier
    in the same fiscal year, so the ₹1.25L exemption is not double-counted.
    Must be non-negative; values larger than the exemption simply clamp the
    remaining exemption to 0.

    Negative net buckets are clamped to zero — neither bucket subsidises the
    other (conservative, matches the docstring policy).
    """
    if ltcg_used_inr < 0:
        raise ValueError(
            f"ltcg_used_inr must be non-negative, got {ltcg_used_inr}"
        )

    stcg = 0.0
    ltcg = 0.0
    for g in gains:
        if g.gain_type == GAIN_LTCG:
            ltcg += g.gain_inr
        else:
            # STCG and UNKNOWN both fall in the short-term bucket.
            stcg += g.gain_inr

    stcg_for_tax = max(stcg, 0.0)
    ltcg_for_tax = max(ltcg, 0.0)

    remaining_exemption = max(LTCG_EXEMPTION_INR - ltcg_used_inr, 0.0)
    exemption_used = min(ltcg_for_tax, remaining_exemption)
    ltcg_taxable = ltcg_for_tax - exemption_used

    stcg_tax = stcg_for_tax * STCG_RATE
    ltcg_tax = ltcg_taxable * LTCG_RATE

    return TaxBill(
        stcg_realized_inr=stcg,
        ltcg_realized_inr=ltcg,
        ltcg_taxable_inr=ltcg_taxable,
        stcg_tax_inr=stcg_tax,
        ltcg_tax_inr=ltcg_tax,
        total_tax_inr=stcg_tax + ltcg_tax,
        ltcg_exemption_consumed_inr=exemption_used,
    )
