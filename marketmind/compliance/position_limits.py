"""Position-limit checks (W5.3 T3).

Concentration-only this section. Computes post-trade symbol weight and
warns if it exceeds ``max_concentration_pct`` of the portfolio. Pure
compute — no fetcher, no Mongo. F&O ban-period checks are a separate
concern deferred to W5.3 Section 4 (needs an NSE endpoint with its own
10-min Mongo TTL cache).

A "warning" does not block the trade — pretrade_check.PretradeChecker
maps warnings to ``DECISION_WARN`` and errors to ``DECISION_BLOCK``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

DEFAULT_MAX_CONCENTRATION_PCT = 25.0


@dataclass(frozen=True)
class PositionLimitStatus:
    """Outcome of the concentration check."""

    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _holding_value(h: Dict[str, Any]) -> float:
    """Best-effort current value: last_price × quantity."""
    try:
        qty = float(h.get("quantity", 0) or 0)
        last = float(h.get("last_price", 0) or h.get("average_price", 0) or 0)
        return qty * last
    except (TypeError, ValueError):
        return 0.0


def check_position_limits(
    symbol: str,
    transaction_type: str,
    quantity: float,
    price: float,
    holdings: List[Dict[str, Any]],
    max_concentration_pct: float = DEFAULT_MAX_CONCENTRATION_PCT,
) -> PositionLimitStatus:
    """Decide whether the proposed trade exceeds concentration limits.

    Errors: zero/negative qty or price → BLOCK-worthy.
    Warnings: post-trade single-symbol weight > ``max_concentration_pct``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    sym_norm = symbol.upper().strip() if isinstance(symbol, str) else ""
    txn = transaction_type.upper().strip() if isinstance(transaction_type, str) else ""

    if not sym_norm:
        errors.append("symbol must be a non-empty string")
    if txn not in {"BUY", "SELL"}:
        errors.append(f"transaction_type must be BUY or SELL, got {transaction_type!r}")
    try:
        q = float(quantity)
        p = float(price)
    except (TypeError, ValueError):
        errors.append("quantity and price must be numeric")
        return PositionLimitStatus(ok=False, warnings=warnings, errors=errors)
    if q <= 0:
        errors.append(f"quantity must be > 0, got {q}")
    if p <= 0:
        errors.append(f"price must be > 0, got {p}")
    if errors:
        return PositionLimitStatus(ok=False, warnings=warnings, errors=errors)

    trade_value = q * p
    portfolio_value = sum(_holding_value(h) for h in holdings)
    sym_holdings = [
        h for h in holdings
        if (h.get("tradingsymbol") or h.get("symbol", "")).upper() == sym_norm
    ]
    current_sym_value = sum(_holding_value(h) for h in sym_holdings)
    current_sym_qty = 0.0
    for h in sym_holdings:
        try:
            current_sym_qty += float(h.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            continue

    # Over-sell guard — broker would reject this anyway, but a compliance
    # gate that lets it through is misleading. Cannot SELL more than held;
    # cannot SELL a symbol you don't own at all.
    if txn == "SELL" and q > current_sym_qty:
        errors.append(
            f"cannot SELL {q} {sym_norm} — only {current_sym_qty} held"
        )
        return PositionLimitStatus(ok=False, warnings=warnings, errors=errors)

    if txn == "BUY":
        new_sym_value = current_sym_value + trade_value
        new_portfolio = portfolio_value + trade_value
    else:  # SELL
        new_sym_value = max(0.0, current_sym_value - trade_value)
        new_portfolio = max(0.0, portfolio_value - trade_value)

    if new_portfolio <= 0:
        # All-out sell — concentration undefined post-trade; nothing to warn on.
        return PositionLimitStatus(ok=True, warnings=warnings, errors=errors)

    concentration_pct = (new_sym_value / new_portfolio) * 100.0
    if concentration_pct > max_concentration_pct:
        warnings.append(
            f"post-trade {sym_norm} concentration "
            f"{concentration_pct:.1f}% > limit {max_concentration_pct:.1f}%"
        )

    return PositionLimitStatus(ok=True, warnings=warnings, errors=errors)
