"""Tests for Kite-based option chain fallback."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pytest

from marketmind.core.options_fetcher import OptionsFetcher


class _StubKite:
    """Minimal KiteClient stub for option chain fallback tests."""

    def __init__(self, connected: bool = True, instruments: List[Dict] = None,
                 quotes: Dict[str, Any] = None, ltp: Dict[str, Any] = None) -> None:
        self.is_connected = connected
        self._instruments = instruments or []
        self._quotes = quotes or {}
        self._ltp = ltp or {}

    def get_instruments(self, exchange: str) -> List[Dict]:
        return self._instruments

    def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        return {s: self._quotes.get(s, {}) for s in symbols}

    def get_ltp(self, symbols: List[str]) -> Dict[str, Any]:
        return self._ltp


def _make_instruments(symbol: str = "RELIANCE") -> List[Dict]:
    """Generate stub NFO instruments for RELIANCE options."""
    expiry = date.today() + timedelta(days=7)
    strikes = [2400, 2420, 2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580, 2600]
    insts = []
    for strike in strikes:
        for typ in ('CE', 'PE'):
            insts.append({
                'tradingsymbol': f'{symbol}{expiry.strftime("%y%b").upper()}{int(strike)}{typ}',
                'name': symbol,
                'strike': float(strike),
                'expiry': expiry,
                'instrument_type': typ,
                'segment': 'NFO-OPT',
                'lot_size': 250,
                'exchange': 'NFO',
            })
    return insts


def _make_quotes(instruments: List[Dict]) -> Dict[str, Any]:
    """Generate stub quotes matching the instruments."""
    quotes = {}
    for inst in instruments:
        key = f"NFO:{inst['tradingsymbol']}"
        strike = inst['strike']
        is_ce = inst['instrument_type'] == 'CE'
        base_price = max(0.5, abs(2500 - strike) * 0.3 + (5 if is_ce else 3))
        quotes[key] = {
            'last_price': round(base_price, 2),
            'oi': int(50000 + abs(2500 - strike) * 1000),
            'volume': int(10000 + abs(2500 - strike) * 500),
            'depth': {
                'buy': [{'price': round(base_price - 0.5, 2), 'quantity': 100}],
                'sell': [{'price': round(base_price + 0.5, 2), 'quantity': 100}],
            },
        }
    return quotes


def test_kite_fallback_builds_chain() -> None:
    """When NSE is empty, Kite fallback produces a valid chain."""
    fetcher = OptionsFetcher()
    insts = _make_instruments("RELIANCE")
    quotes = _make_quotes(insts)
    ltp = {"NSE:RELIANCE": {"last_price": 2500.0}}
    kite = _StubKite(instruments=insts, quotes=quotes, ltp=ltp)

    result = fetcher.get_option_chain_from_kite("RELIANCE", kite)
    assert result is not None
    assert result["symbol"] == "RELIANCE"
    assert result["underlying"] == 2500.0
    assert result["source"] == "kite"
    assert len(result["calls"]) > 0
    assert len(result["puts"]) > 0
    assert result["atm_strike"] == 2500.0
    assert "pcr" in result
    assert "sentiment" in result


def test_kite_fallback_not_connected_returns_none() -> None:
    """If Kite is not connected, fallback returns None."""
    fetcher = OptionsFetcher()
    kite = _StubKite(connected=False)
    assert fetcher.get_option_chain_from_kite("RELIANCE", kite) is None


def test_kite_fallback_index_mapping() -> None:
    """Index names are mapped correctly (NIFTY → NIFTY 50)."""
    fetcher = OptionsFetcher()
    expiry = date.today() + timedelta(days=7)
    insts = [
        {
            'tradingsymbol': f'NIFTY{expiry.strftime("%y%b").upper()}25000CE',
            'name': 'NIFTY 50',
            'strike': 25000.0,
            'expiry': expiry,
            'instrument_type': 'CE',
            'segment': 'NFO-OPT',
            'lot_size': 50,
            'exchange': 'NFO',
        },
        {
            'tradingsymbol': f'NIFTY{expiry.strftime("%y%b").upper()}25000PE',
            'name': 'NIFTY 50',
            'strike': 25000.0,
            'expiry': expiry,
            'instrument_type': 'PE',
            'segment': 'NFO-OPT',
            'lot_size': 50,
            'exchange': 'NFO',
        },
    ]
    quotes = {
        'NFO:NIFTY' + expiry.strftime("%y%b").upper() + '25000CE': {
            'last_price': 150.0, 'oi': 100000, 'volume': 50000,
            'depth': {'buy': [{'price': 149.5, 'quantity': 100}], 'sell': [{'price': 150.5, 'quantity': 100}]},
        },
        'NFO:NIFTY' + expiry.strftime("%y%b").upper() + '25000PE': {
            'last_price': 120.0, 'oi': 80000, 'volume': 40000,
            'depth': {'buy': [{'price': 119.5, 'quantity': 100}], 'sell': [{'price': 120.5, 'quantity': 100}]},
        },
    }
    ltp = {"NSE:NIFTY": {"last_price": 25000.0}}
    kite = _StubKite(instruments=insts, quotes=quotes, ltp=ltp)

    result = fetcher.get_option_chain_from_kite("NIFTY", kite)
    assert result is not None
    assert result["symbol"] == "NIFTY"
    assert result["underlying"] == 25000.0


def test_kite_fallback_no_options_returns_none() -> None:
    """If no matching options exist in Kite's instrument list, return None."""
    fetcher = OptionsFetcher()
    kite = _StubKite(instruments=[], connected=True)
    assert fetcher.get_option_chain_from_kite("UNKNOWN", kite) is None
