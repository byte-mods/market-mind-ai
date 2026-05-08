import numpy as np
import pandas as pd
import pytest
from marketmind.analysis.portfolio_optimizer import PortfolioOptimizer


class FakePriceFetcher:
    def __init__(self, returns_df):
        self._returns = returns_df

    def get_historical_data(self, symbol, days=252):
        rets = self._returns[symbol]
        prices = 100 * (1 + rets).cumprod()
        df = pd.DataFrame({
            'date': pd.date_range(end='2024-01-01', periods=len(prices), freq='B'),
            'close': prices.values,
        })
        return df


def _make_optimizer(returns_df):
    return PortfolioOptimizer(price_fetcher=FakePriceFetcher(returns_df))


def _sample_returns(n_assets=5, n_days=252, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0005, 0.015, n_days)
    data = {}
    for i in range(n_assets):
        noise = rng.normal(0, 0.01, n_days)
        data[f'SYM{i}'] = base + noise * (0.5 + i * 0.2)
    return pd.DataFrame(data)


class TestHRP:
    def test_hrp_weights_sum_to_one(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        result = opt.optimize(list(rets.columns), objective='hrp', days=252)
        assert 'error' not in result
        weights = list(result['weights'].values())
        assert abs(sum(weights) - 100.0) < 0.1

    def test_hrp_no_negative_weights(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        result = opt.optimize(list(rets.columns), objective='hrp', days=252)
        assert 'error' not in result
        weights = list(result['weights'].values())
        assert all(w >= 0 for w in weights)

    def test_hrp_does_not_concentrate_excessively(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        result = opt.optimize(list(rets.columns), objective='hrp', days=252)
        assert 'error' not in result
        weights = result['weights']
        # HRP should avoid putting >90% in one asset on a diversified panel
        assert max(weights.values()) < 90.0

    def test_hrp_returns_valid_stats(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        result = opt.optimize(list(rets.columns), objective='hrp', days=252)
        assert 'error' not in result
        assert 'expected_annual_return_pct' in result
        assert 'annual_volatility_pct' in result
        assert 'sharpe_ratio' in result
        assert result['annual_volatility_pct'] >= 0

    def test_hrp_fallback_on_internal_error(self, monkeypatch):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        monkeypatch.setattr(
            opt, '_get_quasi_diag',
            lambda link: (_ for _ in ()).throw(ValueError('forced'))
        )
        result = opt.optimize(list(rets.columns), objective='hrp', days=252)
        assert 'error' not in result
        expected = 100.0 / len(rets.columns)
        for w in result['weights'].values():
            assert abs(w - expected) < 0.1


class TestBlackLitterman:
    def test_bl_no_views_returns_market_prior(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        result = opt.optimize(list(rets.columns), objective='black_litterman', days=252)
        assert 'error' not in result
        # No views → equal-weight fallback (market prior)
        expected = 100.0 / len(rets.columns)
        for w in result['weights'].values():
            assert abs(w - expected) < 0.1

    def test_bl_absolute_view_shifts_weight(self):
        rets = _sample_returns()
        syms = list(rets.columns)
        opt = _make_optimizer(rets)
        # Without views
        base = opt.optimize(syms, objective='black_litterman', days=252)
        # With a strong positive view on SYM0
        views = [{'assets': ['SYM0'], 'type': 'absolute', 'magnitude': 0.30, 'confidence': 0.8}]
        result = opt.optimize(syms, objective='black_litterman', days=252, views=views)
        assert 'error' not in result
        assert result['weights']['SYM0'] > base['weights']['SYM0']

    def test_bl_relative_view_shifts_weight(self):
        rets = _sample_returns()
        syms = list(rets.columns)
        opt = _make_optimizer(rets)
        views = [{'assets': ['SYM0', 'SYM1'], 'type': 'relative', 'magnitude': 0.05, 'confidence': 0.7}]
        result = opt.optimize(syms, objective='black_litterman', days=252, views=views)
        assert 'error' not in result
        # SYM0 should get more weight than SYM1 due to positive relative view
        assert result['weights']['SYM0'] > result['weights']['SYM1']

    def test_bl_invalid_view_asset_ignored(self):
        rets = _sample_returns()
        syms = list(rets.columns)
        opt = _make_optimizer(rets)
        views = [{'assets': ['INVALID'], 'type': 'absolute', 'magnitude': 0.20, 'confidence': 0.5}]
        result = opt.optimize(syms, objective='black_litterman', days=252, views=views)
        assert 'error' not in result
        expected = 100.0 / len(syms)
        for w in result['weights'].values():
            assert abs(w - expected) < 0.1

    def test_bl_weights_sum_to_one(self):
        rets = _sample_returns()
        syms = list(rets.columns)
        opt = _make_optimizer(rets)
        views = [
            {'assets': ['SYM0'], 'type': 'absolute', 'magnitude': 0.20, 'confidence': 0.6},
            {'assets': ['SYM1', 'SYM2'], 'type': 'relative', 'magnitude': 0.03, 'confidence': 0.5},
        ]
        result = opt.optimize(syms, objective='black_litterman', days=252, views=views)
        assert 'error' not in result
        weights = list(result['weights'].values())
        assert abs(sum(weights) - 100.0) < 0.1
        assert all(w >= 0 for w in weights)

    def test_bl_fallback_on_internal_error(self, monkeypatch):
        rets = _sample_returns()
        syms = list(rets.columns)
        opt = _make_optimizer(rets)
        monkeypatch.setattr(
            opt, '_max_sharpe',
            lambda mean_rets, cov, n: (_ for _ in ()).throw(ValueError('forced'))
        )
        views = [{'assets': ['SYM0'], 'type': 'absolute', 'magnitude': 0.10, 'confidence': 0.5}]
        result = opt.optimize(syms, objective='black_litterman', days=252, views=views)
        assert 'error' not in result
        expected = 100.0 / len(syms)
        for w in result['weights'].values():
            assert abs(w - expected) < 0.1


class TestIntegration:
    def test_default_objective_is_hrp(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        # Call without explicit objective — should use new default
        result = opt.optimize(list(rets.columns), days=252)
        assert result['objective'] == 'hrp'

    def test_compare_strategies_includes_hrp_and_bl(self):
        rets = _sample_returns()
        opt = _make_optimizer(rets)
        results = opt.compare_strategies(list(rets.columns), days=252)
        strategies = [r['strategy'] for r in results]
        assert 'hrp' in strategies
        assert 'black_litterman' in strategies
        assert 'max_sharpe' in strategies
        # All rows should have the same expected keys
        for r in results:
            assert 'return_pct' in r
            assert 'vol_pct' in r
            assert 'sharpe' in r
            assert 'top_weight' in r
            assert 'top_pct' in r
