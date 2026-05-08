"""Tests for MultiAssetFetcher (W4.2) — forex, crypto, commodities, correlations."""
import pytest
import json
import time
from unittest.mock import patch, MagicMock, PropertyMock
from marketmind.core.multiasset_fetcher import (
    MultiAssetFetcher, get_multiasset_fetcher, COINGECKO_CACHE_TTL,
    MCX_COMMODITIES, FOREX_SYMBOLS, CRYPTO_ASSETS,
)


class TestMultiAssetFetcherSingleton:
    def test_get_multiasset_fetcher_returns_same_instance(self):
        a = get_multiasset_fetcher()
        b = get_multiasset_fetcher()
        assert a is b


class TestForexNSEParsing:
    """Test _fetch_forex_single against realistic NSE quote-derivative payloads."""

    def make_nse_response(self, last_price, prev_close):
        return {
            "quoteVFO": {
                "lastPrice": last_price,
                "prevClose": prev_close,
            }
        }

    def test_usdinr_parses_live_nse_response(self):
        maf = MultiAssetFetcher()
        payload = self.make_nse_response(84.75, 84.50)
        with patch.object(maf, '_nse_get', return_value=payload):
            result = maf._fetch_forex_single(FOREX_SYMBOLS['USDINR'])
        assert result['rate'] == 84.75
        assert result['prev'] == 84.50
        assert result['change'] == pytest.approx(0.25)
        assert result['change_pct'] == pytest.approx(0.296, abs=0.01)
        assert result['source'] == 'nse'
        assert result['base'] == 'USD'

    def test_eurinr_parses_live_nse_response(self):
        maf = MultiAssetFetcher()
        payload = self.make_nse_response(91.20, 91.50)
        with patch.object(maf, '_nse_get', return_value=payload):
            result = maf._fetch_forex_single(FOREX_SYMBOLS['EURINR'])
        assert result['rate'] == 91.20
        assert result['change'] == pytest.approx(-0.30)
        assert result['source'] == 'nse'

    def test_forex_falls_back_to_estimate_on_nse_failure(self):
        maf = MultiAssetFetcher()
        with patch.object(maf, '_nse_get', return_value=None):
            result = maf._fetch_forex_single(FOREX_SYMBOLS['USDINR'])
        assert result['rate'] == 84.5  # fallback rate
        assert result['source'] == 'estimate'

    def test_forex_zero_prev_close_does_not_divide_by_zero(self):
        maf = MultiAssetFetcher()
        payload = self.make_nse_response(84.75, 0)
        with patch.object(maf, '_nse_get', return_value=payload):
            result = maf._fetch_forex_single(FOREX_SYMBOLS['USDINR'])
        assert result['change_pct'] == 0

    def test_get_forex_returns_both_pairs(self):
        maf = MultiAssetFetcher()
        maf._cache.clear()
        usd_payload = {"quoteVFO": {"lastPrice": 84.75, "prevClose": 84.50}}
        eur_payload = {"quoteVFO": {"lastPrice": 91.20, "prevClose": 91.50}}

        def fake_nse_get(path):
            if 'USDINR' in path:
                return usd_payload
            if 'EURINR' in path:
                return eur_payload
            return None

        with patch.object(maf, '_nse_get', side_effect=fake_nse_get):
            result = maf.get_forex()
        assert 'usdinr' in result
        assert 'eurinr' in result
        assert result['usdinr']['rate'] == 84.75
        assert result['eurinr']['rate'] == 91.20


class TestCryptoCoinGeckoParsing:
    """Test CoinGecko response parsing and caching."""

    def make_cg_price_response(self, btc_inr, btc_24h, eth_inr, eth_24h):
        return {
            "bitcoin": {"inr": btc_inr, "inr_24h_change": btc_24h},
            "ethereum": {"inr": eth_inr, "inr_24h_change": eth_24h},
        }

    def test_crypto_parses_coingecko_response(self):
        maf = MultiAssetFetcher()
        maf._cg_cache.clear()
        payload = self.make_cg_price_response(7200000, 2.5, 520000, -1.3)
        with patch.object(maf, '_coingecko_get', return_value=payload):
            result = maf.get_crypto()
        assert result['btc']['price_inr'] == 7200000
        assert result['btc']['change_24h_pct'] == 2.5
        assert result['eth']['price_inr'] == 520000
        assert result['eth']['change_24h_pct'] == -1.3
        assert result['btc']['source'] == 'coingecko'

    def test_crypto_marks_unavailable_on_none_response(self):
        maf = MultiAssetFetcher()
        maf._cg_cache.clear()
        with patch.object(maf, '_coingecko_get', return_value=None):
            result = maf.get_crypto()
        assert result['btc']['source'] == 'unavailable'
        assert result['eth']['source'] == 'unavailable'
        assert result['btc']['price_inr'] is None

    def test_crypto_returns_cached_data_within_ttl(self):
        maf = MultiAssetFetcher()
        maf._cg_cache.clear()
        payload = self.make_cg_price_response(7200000, 2.5, 520000, -1.3)
        call_count = [0]

        def counting_get(path, cache_key):
            call_count[0] += 1
            return payload

        with patch.object(maf, '_coingecko_get', side_effect=counting_get):
            r1 = maf.get_crypto()
            r2 = maf.get_crypto()
        assert call_count[0] == 1  # second call served from cache
        assert r1 == r2


class TestMCXCommodities:
    """Test commodity price resolution via Kite."""

    def test_commodities_returns_unavailable_when_kite_disconnected(self):
        maf = MultiAssetFetcher()
        maf._cache.clear()
        mock_kite = MagicMock()
        type(mock_kite).is_connected = PropertyMock(return_value=False)
        result = maf.get_commodities(mock_kite)
        assert result['gold']['source'] == 'unavailable'
        assert result['gold']['price'] is None

    def test_commodities_parses_kite_ltp_response(self):
        maf = MultiAssetFetcher()
        maf._cache.clear()
        mock_kite = MagicMock()
        type(mock_kite).is_connected = PropertyMock(return_value=True)
        mock_kite.get_ltp.return_value = {
            'MCX:GOLDM': {'last_price': 78250.0},
            'MCX:SILVERM': {'last_price': 98500.0},
            'MCX:CRUDEOILM': {'last_price': 6750.0},
            'MCX:ZINCM': {'last_price': 312.5},
        }
        mock_kite.get_ohlc.return_value = {
            'MCX:GOLDM': {'ohlc': {'close': 78000.0}},
            'MCX:SILVERM': {'ohlc': {'close': 99000.0}},
            'MCX:CRUDEOILM': {'ohlc': {'close': 6800.0}},
            'MCX:ZINCM': {'ohlc': {'close': 310.0}},
        }
        result = maf.get_commodities(mock_kite)
        assert result['gold']['price'] == 78250.0
        assert result['gold']['prev_close'] == 78000.0
        assert result['gold']['change_pct'] == pytest.approx(0.32, abs=0.1)
        assert result['gold']['source'] == 'kite'

    def test_commodities_zero_ltp_marks_unavailable(self):
        maf = MultiAssetFetcher()
        maf._cache.clear()
        mock_kite = MagicMock()
        type(mock_kite).is_connected = PropertyMock(return_value=True)
        mock_kite.get_ltp.return_value = {}
        mock_kite.get_ohlc.return_value = {}
        result = maf.get_commodities(mock_kite)
        assert result['gold']['source'] == 'unavailable'
        assert result['gold']['price'] is None


class TestCrossAssetCorrelations:
    """Test correlation matrix computation."""

    def test_empty_correlations_with_no_data(self):
        maf = MultiAssetFetcher()
        mock_kite = MagicMock()
        mock_kite.is_connected = False
        mock_pf = MagicMock()
        mock_pf.get_historical_data.return_value = None

        with patch.object(maf, 'get_crypto_history', return_value=[]):
            result = maf.get_cross_asset_correlations(mock_kite, mock_pf, [30])
        assert '30' in result
        assert 'correlation_matrix' in result['30']
        # When no data is available, matrix is empty
        assert result['30']['correlation_matrix'] == {}

    def test_correlations_with_sufficient_data(self):
        maf = MultiAssetFetcher()
        import pandas as pd
        import numpy as np

        # Create 60 days of correlated price data
        dates = pd.date_range('2026-01-01', periods=60, freq='B')
        np.random.seed(42)
        nifty_returns = np.random.normal(0.0005, 0.01, 60)
        gold_returns = nifty_returns * 0.3 + np.random.normal(0.0002, 0.008, 60)  # ~0.3 corr

        nifty_prices = 22000 * np.cumprod(1 + nifty_returns)
        gold_prices = 78000 * np.cumprod(1 + gold_returns)

        nifty_df = pd.DataFrame({'close': nifty_prices}, index=dates)
        gold_df = pd.DataFrame({'close': gold_prices}, index=dates)

        mock_pf = MagicMock()
        mock_pf.get_historical_data.return_value = nifty_df

        mock_kite = MagicMock()
        mock_kite.is_connected = True
        mock_kite.get_historical_data.return_value = gold_df

        with patch.object(maf, 'get_crypto_history', return_value=[]):
            result = maf.get_cross_asset_correlations(mock_kite, mock_pf, [30])

        assert '30' in result
        matrix = result['30']['correlation_matrix']
        assert 'nifty500' in matrix
        assert 'gold' in matrix
        # Correlation should exist between nifty500 and gold
        corr_val = matrix['nifty500'].get('gold')
        assert corr_val is not None
        assert -1.0 <= corr_val <= 1.0

    def test_crypto_history_parses_market_chart(self):
        maf = MultiAssetFetcher()
        now_ms = int(time.time() * 1000)
        day_ms = 86400 * 1000
        payload = {
            'prices': [
                [now_ms - 30 * day_ms, 7000000],
                [now_ms - 29 * day_ms, 7100000],
                [now_ms - 28 * day_ms, 7050000],
            ]
        }
        with patch.object(maf, '_coingecko_get', return_value=payload):
            history = maf.get_crypto_history('bitcoin', days=30)
        assert len(history) == 3
        assert history[0]['price'] == 7000000
        assert 'date' in history[0]

    def test_get_all_aggregates(self):
        maf = MultiAssetFetcher()
        maf._cache.clear()
        maf._cg_cache.clear()
        mock_kite = MagicMock()
        type(mock_kite).is_connected = PropertyMock(return_value=False)
        mock_pf = MagicMock()

        usd_payload = {"quoteVFO": {"lastPrice": 84.75, "prevClose": 84.50}}
        cg_payload = {"bitcoin": {"inr": 7200000, "inr_24h_change": 2.5},
                       "ethereum": {"inr": 520000, "inr_24h_change": -1.3}}

        with patch.object(maf, '_nse_get', return_value=usd_payload):
            with patch.object(maf, '_coingecko_get', return_value=cg_payload):
                result = maf.get_all(mock_kite, mock_pf)

        assert 'commodities' in result
        assert 'forex' in result
        assert 'crypto' in result
        assert 'timestamp' in result
