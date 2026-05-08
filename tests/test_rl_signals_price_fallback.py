"""Verify the RL Signals price-resolution chain: Kite live → NSE live → historical close.

These pin the contract introduced when we made the Multi-Timeframe RL Signals card
display fresh prices on Refresh. Before this fix, the helper fell straight from
"Kite missing" to "yesterday's close", so Entry/Target/SL anchored on a stale value
whenever the Kite session was thin or offline. The chain MUST now insert NSE-live
between those two layers.

The helper under test is `server._resolve_rl_cp` — a pure function lifted out of
`rl_multiframe` precisely so the chain can be unit-tested without booting the full
controller, the Kite client, or the historical candle store.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from server import _resolve_rl_cp


class _StubPF:
    """Stand-in for `PriceFetcher`. Records calls; configurable response.

    `response` is what `get_stock_price` returns. `raise_exc` causes it to raise
    instead — used to verify the helper swallows network/parse failures and
    falls through to the historical close.
    """

    def __init__(self, response: Optional[Dict[str, Any]] = None,
                 raise_exc: Optional[Exception] = None) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    def get_stock_price(self, sym: str) -> Optional[Dict[str, Any]]:
        self.calls.append(sym)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _hist_df(last_close: float = 1234.5) -> pd.DataFrame:
    """Two-row df is enough — only `df['close'].iloc[-1]` is read."""
    return pd.DataFrame({"close": [1200.0, last_close]})


# ─── Layer 1: Kite live LTP wins when present ──────────────────────────────


def test_resolve_cp_kite_live_takes_precedence_over_nse_and_history() -> None:
    """When Kite returned a positive LTP, NSE must not be hit and history is ignored."""
    pf = _StubPF(response={"current_price": 999.99})  # would be wrong if used
    kite_prices = {"RELIANCE": {"current_price": 1500.25}}
    df = _hist_df(last_close=1100.0)

    cp = _resolve_rl_cp("RELIANCE", kite_prices, pf, df)

    assert cp == 1500.25
    assert pf.calls == [], "NSE fallback must not be invoked when Kite LTP is present"


# ─── Layer 2: Kite-miss → NSE-live anchor (the bug fix) ─────────────────────


def test_resolve_cp_falls_to_nse_live_when_kite_misses_symbol() -> None:
    """Kite returned no entry for the symbol → NSE-live is the next anchor.

    This is the headline fix: previously the helper would jump straight to
    `df['close'].iloc[-1]` (yesterday's close) and Target/SL would compute off
    that. The NSE-live layer now sits between the two.
    """
    pf = _StubPF(response={"current_price": 1444.5})
    kite_prices: Dict[str, Dict[str, Any]] = {}  # Kite offline / didn't return RELIANCE
    df = _hist_df(last_close=1100.0)

    cp = _resolve_rl_cp("RELIANCE", kite_prices, pf, df)

    assert cp == 1444.5, "must use NSE live, not the stale historical close"
    assert pf.calls == ["RELIANCE"]


def test_resolve_cp_falls_to_nse_live_when_kite_returns_zero() -> None:
    """Kite returned an entry but with zero LTP → treat as miss, descend to NSE."""
    pf = _StubPF(response={"current_price": 808.0})
    kite_prices = {"TCS": {"current_price": 0}}  # zero is a known Kite-degraded shape
    df = _hist_df(last_close=750.0)

    cp = _resolve_rl_cp("TCS", kite_prices, pf, df)

    assert cp == 808.0


# ─── Layer 3: NSE failures → historical close fallthrough ──────────────────


def test_resolve_cp_falls_to_history_when_nse_raises() -> None:
    """NSE raising must not 5xx the route — helper swallows and uses history.

    The route runs against 25 symbols per call; one network blip on a single
    symbol must not abort the scan or escape as an exception. The contract is:
    NSE failure is logged at debug level and the helper drops to the last
    cached close.
    """
    pf = _StubPF(raise_exc=RuntimeError("NSE connection reset"))
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = _hist_df(last_close=1234.5)

    cp = _resolve_rl_cp("INFY", kite_prices, pf, df)

    assert cp == 1234.5
    assert pf.calls == ["INFY"], "NSE was attempted exactly once before fallthrough"


def test_resolve_cp_falls_to_history_when_nse_returns_none() -> None:
    """NSE call succeeded but returned None (e.g. unknown symbol) → use history."""
    pf = _StubPF(response=None)
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = _hist_df(last_close=2050.75)

    cp = _resolve_rl_cp("UNKNOWN", kite_prices, pf, df)

    assert cp == 2050.75


def test_resolve_cp_falls_to_history_when_nse_returns_zero_price() -> None:
    """NSE returned a dict with `current_price=0` (degraded payload) → use history.

    Zero is the canonical "no quote" value from NSE on illiquid or halted symbols.
    Treat it the same as a miss — descend to the historical close rather than
    publishing 0 as the entry price.
    """
    pf = _StubPF(response={"current_price": 0, "symbol": "X"})
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = _hist_df(last_close=42.5)

    cp = _resolve_rl_cp("X", kite_prices, pf, df)

    assert cp == 42.5


# ─── Adversarial pins: malformed inputs the helper must survive ─────────────


def test_resolve_cp_handles_kite_entry_explicitly_none() -> None:
    """`kite_prices[sym]` may be `None` (not a missing key) — must not raise.

    Pinning this prevents a future "fix" that drops the `(... or {})` guard,
    which currently absorbs both missing-key and None-value shapes.
    """
    pf = _StubPF(response={"current_price": 100.0})
    kite_prices = {"X": None}  # value is None, not a dict
    df = _hist_df(last_close=50.0)

    cp = _resolve_rl_cp("X", kite_prices, pf, df)

    assert cp == 100.0  # descended past the None to NSE


def test_resolve_cp_handles_empty_history_df() -> None:
    """`df['close'].iloc[-1]` raises `IndexError` on an empty df — caught, returns 0.

    The route's `if cp <= 0: continue` then skips the symbol, which is the
    intended terminal behaviour when literally no price source is available.
    """
    pf = _StubPF(response=None)
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = pd.DataFrame({"close": []})  # empty

    cp = _resolve_rl_cp("X", kite_prices, pf, df)

    assert cp == 0.0


def test_resolve_cp_handles_nse_non_numeric_current_price() -> None:
    """NSE payload with a non-numeric `current_price` (string, garbage) must not crash.

    The helper's `float(...)` cast raises `TypeError`/`ValueError`; the broad
    except in the NSE layer catches it and the chain descends to history.
    Pinning this contract prevents a future narrowing of the except clause from
    silently regressing the swallow.
    """
    pf = _StubPF(response={"current_price": "not-a-number"})
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = _hist_df(last_close=77.7)

    cp = _resolve_rl_cp("X", kite_prices, pf, df)

    assert cp == 77.7


def test_resolve_cp_returns_zero_when_every_layer_fails() -> None:
    """All three sources empty/zero → helper returns 0.0; caller skips the symbol.

    Pinning this is important because the route's `if cp <= 0: continue` guard
    depends on a non-positive sentinel rather than an exception or None.
    """
    pf = _StubPF(response={"current_price": 0})
    kite_prices: Dict[str, Dict[str, Any]] = {}
    df = pd.DataFrame({"close": [0.0, 0.0]})

    cp = _resolve_rl_cp("DEAD", kite_prices, pf, df)

    assert cp == 0.0
