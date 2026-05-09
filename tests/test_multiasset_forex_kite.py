"""Tests for MultiAssetFetcher._fetch_forex_kite — Kite CDS path."""
from __future__ import annotations

from datetime import date, timedelta

from marketmind.core.multiasset_fetcher import FOREX_SYMBOLS, MultiAssetFetcher


class _StubKite:
    def __init__(self, instruments=None, ohlc=None, connected=True):
        self.is_connected = connected
        self._insts = instruments or []
        self._ohlc = ohlc or {}

    def get_instruments(self, exchange):
        return self._insts

    def get_ohlc(self, keys):
        return {k: self._ohlc.get(k, {}) for k in keys}


def test_forex_kite_picks_nearest_future_and_returns_change():
    today = date.today()
    near = today + timedelta(days=10)
    far = today + timedelta(days=40)
    insts = [
        {'name': 'USDINR', 'tradingsymbol': 'USDINRFUT_FAR',
         'instrument_type': 'FUT', 'expiry': far, 'segment': 'CDS-FUT'},
        {'name': 'USDINR', 'tradingsymbol': 'USDINRFUT_NEAR',
         'instrument_type': 'FUT', 'expiry': near, 'segment': 'CDS-FUT'},
    ]
    ohlc = {
        'CDS:USDINRFUT_NEAR': {
            'last_price': 84.65,
            'ohlc': {'close': 84.50},
        }
    }
    kite = _StubKite(instruments=insts, ohlc=ohlc)
    f = MultiAssetFetcher()
    out = f._fetch_forex_kite(FOREX_SYMBOLS['USDINR'], kite)
    assert out is not None
    assert out['source'] == 'kite'
    assert out['rate'] == 84.65
    assert out['prev'] == 84.50
    assert out['change'] == 0.15
    assert out['change_pct'] == round(0.15 / 84.50 * 100, 3)


def test_forex_kite_returns_none_when_no_futures():
    kite = _StubKite(instruments=[])
    f = MultiAssetFetcher()
    out = f._fetch_forex_kite(FOREX_SYMBOLS['USDINR'], kite)
    assert out is None


def test_forex_kite_returns_none_when_kite_disconnected_path_skipped():
    """get_forex skips Kite path when not connected — Kite-only test verifies
    the helper still returns None when ohlc is empty even if connected."""
    today = date.today()
    insts = [{'name': 'USDINR', 'tradingsymbol': 'USDINRFUT',
              'instrument_type': 'FUT', 'expiry': today + timedelta(days=10),
              'segment': 'CDS-FUT'}]
    kite = _StubKite(instruments=insts, ohlc={})
    f = MultiAssetFetcher()
    assert f._fetch_forex_kite(FOREX_SYMBOLS['USDINR'], kite) is None


def test_forex_kite_falls_back_to_ohlc_close_when_market_closed():
    """When CDS market is shut, Kite returns last_price=0 but the OHLC envelope
    still carries the prior session's close. The helper must use that close
    instead of returning None and tripping the (broken) NSE fallback."""
    today = date.today()
    insts = [{'name': 'USDINR', 'tradingsymbol': 'USDINRFUT',
              'instrument_type': 'FUT', 'expiry': today + timedelta(days=10),
              'segment': 'CDS-FUT'}]
    ohlc = {
        'CDS:USDINRFUT': {
            'last_price': 0,
            'ohlc': {'close': 91.34},
        }
    }
    kite = _StubKite(instruments=insts, ohlc=ohlc)
    f = MultiAssetFetcher()
    out = f._fetch_forex_kite(FOREX_SYMBOLS['USDINR'], kite)
    assert out is not None
    assert out['rate'] == 91.34
    assert out['prev'] == 91.34
    assert out['change_pct'] == 0
    assert out['source'] == 'kite'


def test_forex_kite_skips_past_expiries():
    today = date.today()
    insts = [
        {'name': 'USDINR', 'tradingsymbol': 'EXPIRED',
         'instrument_type': 'FUT', 'expiry': today - timedelta(days=5),
         'segment': 'CDS-FUT'},
    ]
    kite = _StubKite(instruments=insts)
    f = MultiAssetFetcher()
    assert f._fetch_forex_kite(FOREX_SYMBOLS['USDINR'], kite) is None
