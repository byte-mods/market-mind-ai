"""W5.3 T3 verification: position-limit concentration check."""
from __future__ import annotations

import pytest

from marketmind.compliance.position_limits import (
    DEFAULT_MAX_CONCENTRATION_PCT,
    PositionLimitStatus,
    check_position_limits,
)


def _h(symbol: str, qty: float, last: float) -> dict:
    return {"tradingsymbol": symbol, "quantity": qty, "last_price": last}


# ─── Errors block ────────────────────────────────────────────────────────


def test_position_limits_zero_qty_errors() -> None:
    s = check_position_limits("RELIANCE", "BUY", 0, 100.0, [])
    assert s.ok is False
    assert any("quantity must be > 0" in e for e in s.errors)


def test_position_limits_zero_price_errors() -> None:
    s = check_position_limits("RELIANCE", "BUY", 1, 0.0, [])
    assert s.ok is False
    assert any("price must be > 0" in e for e in s.errors)


def test_position_limits_invalid_txn_type_errors() -> None:
    s = check_position_limits("RELIANCE", "MAYBE", 1, 100.0, [])
    assert s.ok is False
    assert any("transaction_type must be" in e for e in s.errors)


def test_position_limits_empty_symbol_errors() -> None:
    s = check_position_limits("", "BUY", 1, 100.0, [])
    assert s.ok is False
    assert any("symbol must be" in e for e in s.errors)


def test_position_limits_non_numeric_qty_errors() -> None:
    s = check_position_limits("RELIANCE", "BUY", "ten", 100.0, [])  # type: ignore[arg-type]
    assert s.ok is False
    assert any("must be numeric" in e for e in s.errors)


# ─── Concentration warnings ──────────────────────────────────────────────


def test_position_limits_buy_into_empty_portfolio_warns() -> None:
    # First buy — 100% concentration → must warn.
    s = check_position_limits("RELIANCE", "BUY", 10, 2500.0, [])
    assert s.ok is True
    assert s.errors == []
    assert len(s.warnings) == 1
    assert "100.0%" in s.warnings[0]


def test_position_limits_buy_under_limit_no_warning() -> None:
    holdings = [
        _h("RELIANCE", 1, 1000),
        _h("TCS", 1, 1000),
        _h("INFY", 1, 1000),
        _h("WIPRO", 1, 1000),
        _h("HDFC", 1, 1000),
    ]
    # Buy ₹100 of RELIANCE → portfolio 5100, RELIANCE 1100, conc ≈21.6% < 25%.
    s = check_position_limits("RELIANCE", "BUY", 1, 100.0, holdings)
    assert s.warnings == []


def test_position_limits_buy_breach_warns_with_pct() -> None:
    holdings = [_h("RELIANCE", 10, 100), _h("TCS", 100, 10)]  # 1000 + 1000 = 2000
    # Buy ₹2000 RELIANCE → port 4000, RELIANCE 3000 → 75% → warn.
    s = check_position_limits("RELIANCE", "BUY", 20, 100.0, holdings)
    assert len(s.warnings) == 1
    assert "75.0%" in s.warnings[0]


def test_position_limits_sell_no_concentration_warning() -> None:
    holdings = [_h("RELIANCE", 100, 100), _h("TCS", 1, 100)]  # 10000 + 100
    # Sell ₹5000 RELIANCE → port 5100, RELIANCE 5000 → ~98% still warns.
    s = check_position_limits("RELIANCE", "SELL", 50, 100.0, holdings)
    assert len(s.warnings) == 1


def test_position_limits_sell_full_position_no_division_by_zero() -> None:
    holdings = [_h("RELIANCE", 10, 100)]
    # Sell entire position → portfolio = 0 → no concentration warning.
    s = check_position_limits("RELIANCE", "SELL", 10, 100.0, holdings)
    assert s.warnings == []
    assert s.errors == []


def test_position_limits_custom_max_concentration() -> None:
    # Loosen to 80% — same trade no longer warns.
    holdings = [_h("RELIANCE", 10, 100), _h("TCS", 100, 10)]
    s = check_position_limits(
        "RELIANCE", "BUY", 20, 100.0, holdings,
        max_concentration_pct=80.0,
    )
    assert s.warnings == []


# ─── Symbol case-insensitivity in holdings match ─────────────────────────


def test_position_limits_holdings_symbol_case_insensitive() -> None:
    # Holdings stored lowercase; trade in uppercase — must match the position.
    holdings = [{"tradingsymbol": "reliance", "quantity": 10, "last_price": 100}]
    s = check_position_limits("RELIANCE", "BUY", 5, 100.0, holdings)
    # Existing pos 1000 + trade 500 = 1500, port 1500 → 100% concentration warn.
    assert any("100.0%" in w for w in s.warnings)


def test_position_limits_sell_more_than_held_errors() -> None:
    holdings = [_h("RELIANCE", 10, 100)]
    s = check_position_limits("RELIANCE", "SELL", 11, 100.0, holdings)
    assert s.ok is False
    assert any("only 10" in e for e in s.errors)


def test_position_limits_sell_non_owned_symbol_errors() -> None:
    holdings = [_h("TCS", 10, 100)]
    s = check_position_limits("RELIANCE", "SELL", 1, 100.0, holdings)
    assert s.ok is False
    assert any("only 0" in e for e in s.errors)


def test_position_limits_default_constant_exposed() -> None:
    assert DEFAULT_MAX_CONCENTRATION_PCT == 25.0


def test_position_limits_status_is_frozen() -> None:
    s = PositionLimitStatus(ok=True)
    with pytest.raises(Exception):
        s.ok = False  # type: ignore[misc]
