"""Tests for FactorEngine — cross-sectional Z-score factor exposures."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from marketmind.analysis.factor_model import FactorEngine, _z_score


# ─── Helpers ────────────────────────────────────────────────────────────────


class _FakePriceFetcher:
    """Deterministic price fetcher for testing."""

    def __init__(self, fundamentals: Dict[str, Dict], prices: Dict[str, float]) -> None:
        self.fundamentals = fundamentals
        self.prices = prices

    def _get_screener_fundamentals(self, symbol: str):
        return self.fundamentals.get(symbol.upper())

    def get_historical_data(self, symbol: str, days: int = 365):
        import pandas as pd
        price = self.prices.get(symbol.upper(), 100.0)
        # Generate a simple price series with some drift
        rng = np.random.default_rng(hash(symbol) % 2**31)
        returns = rng.normal(0.0003, 0.015, days)
        close = price * (1 + returns).cumprod()
        dates = pd.date_range(end='2024-01-01', periods=days, freq='B')
        return pd.DataFrame({
            'date': dates,
            'open': close * 0.99,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.ones(days) * 100000,
        })


def _make_engine(fundamentals: Dict[str, Dict], prices: Dict[str, float]) -> FactorEngine:
    return FactorEngine(price_fetcher=_FakePriceFetcher(fundamentals, prices))


# ─── Z-score ────────────────────────────────────────────────────────────────


def test_z_score_median_zero() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _z_score(arr)
    assert abs(float(np.median(z))) < 0.01
    assert z[0] < z[-1]


def test_z_score_constant_returns_zeros() -> None:
    arr = np.array([5.0, 5.0, 5.0])
    z = _z_score(arr)
    assert np.allclose(z, 0.0)


# ─── Universe snapshot ──────────────────────────────────────────────────────


def test_build_universe_snapshot_skips_missing() -> None:
    fund = {
        "A": {"pb_ratio": 2.0, "roe": 15.0},
        "B": {},  # no fundamentals, but price history gives momentum
    }
    engine = _make_engine(fund, {"A": 100, "B": 50})
    snap = engine.build_universe_snapshot(["A", "B"])
    assert "A" in snap
    # B may still appear if momentum could be computed from price history
    if "B" in snap:
        assert "pb" not in snap["B"]  # no fundamentals → no pb


def test_build_universe_snapshot_computes_momentum() -> None:
    fund = {"A": {"pb_ratio": 2.0, "roe": 15.0, "market_cap": 1e12}}
    engine = _make_engine(fund, {"A": 100})
    snap = engine.build_universe_snapshot(["A"])
    assert "A" in snap
    assert "momentum_12m" in snap["A"]


# ─── Factor exposures ───────────────────────────────────────────────────────


def test_compute_exposures_value_inverts_pb() -> None:
    """Lower P/B → higher value exposure (negative Z of P/B)."""
    snapshot = {
        "CHEAP": {"pb": 0.5, "roe": 10.0, "market_cap": 1e12},
        "EXPENSIVE": {"pb": 5.0, "roe": 10.0, "market_cap": 1e12},
    }
    engine = _make_engine({}, {})
    exposures = engine.compute_exposures(snapshot)
    assert exposures["CHEAP"]["value"] > exposures["EXPENSIVE"]["value"]


def test_compute_exposures_size_from_market_cap() -> None:
    snapshot = {
        "LARGE": {"pb": 2.0, "roe": 10.0, "market_cap": 1e13},
        "SMALL": {"pb": 2.0, "roe": 10.0, "market_cap": 1e10},
    }
    engine = _make_engine({}, {})
    exposures = engine.compute_exposures(snapshot)
    assert exposures["LARGE"]["size"] > exposures["SMALL"]["size"]


def test_compute_exposures_profitability_from_roe() -> None:
    snapshot = {
        "HIGH_ROE": {"pb": 2.0, "roe": 25.0, "market_cap": 1e12},
        "LOW_ROE": {"pb": 2.0, "roe": 5.0, "market_cap": 1e12},
    }
    engine = _make_engine({}, {})
    exposures = engine.compute_exposures(snapshot)
    assert exposures["HIGH_ROE"]["profitability"] > exposures["LOW_ROE"]["profitability"]


def test_compute_exposures_quality_penalises_leverage() -> None:
    snapshot = {
        "LOW_LEV": {"pb": 2.0, "roe": 15.0, "roce": 20.0, "market_cap": 1e12, "debt_equity": 0.1},
        "HIGH_LEV": {"pb": 2.0, "roe": 15.0, "roce": 20.0, "market_cap": 1e12, "debt_equity": 2.0},
    }
    engine = _make_engine({}, {})
    exposures = engine.compute_exposures(snapshot)
    assert exposures["LOW_LEV"]["quality"] > exposures["HIGH_LEV"]["quality"]


def test_compute_exposures_empty_snapshot() -> None:
    engine = _make_engine({}, {})
    assert engine.compute_exposures({}) == {}


# ─── Single stock exposure ──────────────────────────────────────────────────


def test_get_stock_exposure_returns_percentiles() -> None:
    fund = {
        "A": {"pb_ratio": 1.0, "roe": 20.0, "market_cap": 1e12},
        "B": {"pb_ratio": 3.0, "roe": 10.0, "market_cap": 1e11},
        "C": {"pb_ratio": 5.0, "roe": 5.0, "market_cap": 1e10},
    }
    engine = _make_engine(fund, {"A": 100, "B": 100, "C": 100})
    result = engine.get_stock_exposure("A", universe=["A", "B", "C"])
    assert result["status"] == "ready"
    assert "value" in result["exposures"]
    assert "value" in result["percentiles"]
    assert 0 <= result["percentiles"]["value"] <= 100


def test_get_stock_exposure_adds_symbol_to_universe() -> None:
    fund = {
        "A": {"pb_ratio": 1.0, "roe": 20.0, "market_cap": 1e12},
    }
    engine = _make_engine(fund, {"A": 100})
    result = engine.get_stock_exposure("A", universe=[])
    assert result["status"] == "ready"


# ─── Portfolio attribution ──────────────────────────────────────────────────


def test_portfolio_attribution_equal_weight_fallback() -> None:
    fund = {
        "A": {"pb_ratio": 1.0, "roe": 20.0, "market_cap": 1e12},
        "B": {"pb_ratio": 3.0, "roe": 10.0, "market_cap": 1e11},
    }
    engine = _make_engine(fund, {"A": 100, "B": 100})
    result = engine.portfolio_attribution(
        [{"symbol": "A"}, {"symbol": "B"}]
    )
    assert result["status"] == "ready"
    assert "portfolio_factors" in result
    assert "factor_drift" in result


def test_portfolio_attribution_weighted_by_value() -> None:
    fund = {
        "A": {"pb_ratio": 1.0, "roe": 20.0, "market_cap": 1e12},
        "B": {"pb_ratio": 5.0, "roe": 5.0, "market_cap": 1e10},
    }
    engine = _make_engine(fund, {"A": 100, "B": 100})
    result = engine.portfolio_attribution([
        {"symbol": "A", "current_value": 80000},
        {"symbol": "B", "current_value": 20000},
    ])
    # 80% in A (cheap, high value) → portfolio should tilt value-positive
    pf_val = result["portfolio_factors"].get("value", 0)
    assert pf_val > 0


def test_portfolio_attribution_no_holdings() -> None:
    engine = _make_engine({}, {})
    result = engine.portfolio_attribution([])
    assert result["status"] == "no_holdings"


# ─── Factor momentum ────────────────────────────────────────────────────────


def test_factor_momentum_returns_correlations() -> None:
    fund = {
        s: {"pb_ratio": 2.0 + i * 0.1, "roe": 10.0 + i, "market_cap": 1e12}
        for i, s in enumerate("ABCDEFGHIJ")
    }
    engine = _make_engine(fund, {s: 100 + i * 10 for i, s in enumerate("ABCDEFGHIJ")})
    result = engine.factor_momentum(universe=list("ABCDEFGHIJ"))
    assert isinstance(result, dict)
    for factor in result:
        assert "correlation" in result[factor]
        assert "spread_pct" in result[factor]
        assert result[factor]["regime"] in ("positive", "negative", "neutral")


def test_factor_momentum_too_few_stocks() -> None:
    fund = {"A": {"pb_ratio": 2.0, "roe": 10.0, "market_cap": 1e12}}
    engine = _make_engine(fund, {"A": 100})
    result = engine.factor_momentum(universe=["A"])
    assert result == {}


# ─── Factor summary ─────────────────────────────────────────────────────────


def test_factor_summary_structure() -> None:
    fund = {
        "A": {"pb_ratio": 1.0, "roe": 20.0, "market_cap": 1e12},
        "B": {"pb_ratio": 3.0, "roe": 10.0, "market_cap": 1e11},
    }
    engine = _make_engine(fund, {"A": 100, "B": 100})
    result = engine.factor_summary(universe=["A", "B"])
    assert "universe_size" in result
    assert "exposures" in result
    assert "momentum" in result
    assert "stats" in result
    assert "value" in result["stats"]
