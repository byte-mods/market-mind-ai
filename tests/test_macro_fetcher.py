"""Tests for MacroFetcher — Nifty PE, USD/INR delegation, breadth."""
from unittest.mock import patch
from marketmind.core.macro_fetcher import MacroFetcher


def test_nifty_pe_reads_dy_field_for_dividend_yield():
    """NSE allIndices returns the dividend yield under `dy`, not `divYield`.
    Regression: code used to read `divYield`, always falling through to 1.2."""
    f = MacroFetcher()
    f._cache.clear()
    payload = {
        'data': [
            {'index': 'NIFTY 500', 'pe': 23.5, 'pb': 3.65, 'dy': 1.11},
        ],
    }
    with patch.object(f, '_nse_get', return_value=payload):
        out = f.get_nifty_pe()
    assert out['div_yield'] == 1.11
    assert out['pe'] == 23.5
    assert out['pb'] == 3.65


def test_nifty_pe_legacy_divYield_alias_still_works():
    """Forward-compat: older fixtures using `divYield` should still parse."""
    f = MacroFetcher()
    f._cache.clear()
    payload = {'data': [{'index': 'NIFTY 500', 'pe': 22, 'pb': 4, 'divYield': 1.5}]}
    with patch.object(f, '_nse_get', return_value=payload):
        out = f.get_nifty_pe()
    assert out['div_yield'] == 1.5


def test_get_usdinr_delegates_to_multiasset_fetcher():
    """Macro USD/INR should resolve via the working multiasset forex path,
    not the legacy quote-derivative.quoteVFO path that returned empty."""
    f = MacroFetcher()
    f._cache.clear()
    fake_forex = {'usdinr': {'rate': 84.92, 'prev': 84.50, 'source': 'kite'}}

    class StubMaf:
        def get_forex(self, kite_client=None):
            return fake_forex

    with patch('marketmind.core.multiasset_fetcher.get_multiasset_fetcher',
               return_value=StubMaf()):
        out = f.get_usdinr(kite_client=None)
    assert out['rate'] == 84.92
    assert out['prev'] == 84.50
    assert out['source'] == 'kite'
    assert out['change'] == 0.42
    assert out['change_pct'] > 0


def test_get_usdinr_falls_back_to_estimate_when_no_rate_available():
    f = MacroFetcher()
    f._cache.clear()

    class StubMaf:
        def get_forex(self, kite_client=None):
            return {'usdinr': {'rate': 0, 'source': 'estimate'}}

    with patch('marketmind.core.multiasset_fetcher.get_multiasset_fetcher',
               return_value=StubMaf()):
        out = f.get_usdinr()
    assert out['source'] == 'estimate'
    assert out['rate'] == 84.5  # macro-level fallback


def test_get_all_passes_kite_client_through():
    """`/api/macro` route forwards controller.kite — verify get_all threads it
    into get_usdinr so the Kite CDS path is reachable."""
    f = MacroFetcher()
    captured = {}

    def stub_usdinr(kite_client=None):
        captured['kite'] = kite_client
        return {'rate': 1.0}

    with patch.object(f, 'get_usdinr', side_effect=stub_usdinr), \
         patch.object(f, 'get_india_vix', return_value={}), \
         patch.object(f, 'get_nifty_pe', return_value={}), \
         patch.object(f, 'get_market_breadth', return_value={}):
        sentinel = object()
        f.get_all(kite_client=sentinel)
    assert captured['kite'] is sentinel
