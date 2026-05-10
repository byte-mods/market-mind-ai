"""Tests for volatility analytics — IV history, rank, surface, term structure, skew."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from marketmind.ml.options.vol_analytics import (
    IVHistoryCollector,
    iv_history,
    iv_rank,
    skew_metrics,
    term_structure,
    vol_surface,
)


class _FakeMongoCol:
    """In-memory stand-in for a pymongo Collection."""

    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        self.indexes: List[Any] = []

    def replace_one(self, filter_doc: Dict, replacement: Dict, upsert: bool = False) -> None:
        key = filter_doc.get("_id")
        self.docs = [d for d in self.docs if d.get("_id") != key]
        self.docs.append(replacement)

    def find(self, query: Dict | None = None) -> "_FakeMongoCol":
        q = query or {}
        symbol = q.get("symbol")
        ts_spec = q.get("ts", {})
        cutoff = ts_spec.get("$gte")
        result: List[Dict] = []
        for d in self.docs:
            if symbol and d.get("symbol") != symbol:
                continue
            if cutoff and d.get("ts") and d["ts"] < cutoff:
                continue
            result.append(d)
        # Sort newest-first by ts
        result.sort(key=lambda x: x.get("ts", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        self._last_cursor = result
        return self

    def sort(self, key: str, direction: int = -1) -> "_FakeMongoCol":
        # Already sorted in find; no-op for compatibility
        return self

    def __iter__(self):
        return iter(self._last_cursor)

    def create_index(self, keys: Any, **kwargs: Any) -> None:
        self.indexes.append((keys, kwargs))


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mongo() -> _FakeMongoCol:
    return _FakeMongoCol()


@pytest.fixture
def sample_chain() -> Dict:
    return {
        "symbol": "NIFTY",
        "underlying": 22500,
        "atm_strike": 22500,
        "expiry_dates": ["15-May-2026"],
        "days_to_expiry": 5,
        "calls": [
            {"strike": 22500, "iv": 18.5, "ltp": 120.0, "oi": 100000},
            {"strike": 22600, "iv": 17.0, "ltp": 80.0, "oi": 80000},
        ],
        "puts": [
            {"strike": 22500, "iv": 19.0, "ltp": 95.0, "oi": 120000},
            {"strike": 22400, "iv": 20.5, "ltp": 60.0, "oi": 90000},
        ],
        "pcr": 1.2,
        "max_pain": 22500,
    }


@pytest.fixture
def multi_expiry_raw() -> Dict:
    """Raw NSE-shaped response with two expiries and enough strikes for skew."""
    return {
        "records": {
            "underlyingValue": 22500,
            "expiryDates": ["15-May-2026", "22-May-2026"],
            "data": [
                {
                    "strikePrice": 22500,
                    "expiryDate": "15-May-2026",
                    "CE": {
                        "openInterest": 100000,
                        "impliedVolatility": 18.5,
                        "lastPrice": 120.0,
                        "bidprice": 119.5,
                        "askPrice": 120.5,
                    },
                    "PE": {
                        "openInterest": 120000,
                        "impliedVolatility": 19.0,
                        "lastPrice": 95.0,
                        "bidprice": 94.5,
                        "askPrice": 95.5,
                    },
                },
                {
                    "strikePrice": 22600,
                    "expiryDate": "15-May-2026",
                    "CE": {
                        "openInterest": 80000,
                        "impliedVolatility": 17.0,
                        "lastPrice": 80.0,
                        "bidprice": 79.5,
                        "askPrice": 80.5,
                    },
                    "PE": {
                        "openInterest": 60000,
                        "impliedVolatility": 16.5,
                        "lastPrice": 110.0,
                        "bidprice": 109.5,
                        "askPrice": 110.5,
                    },
                },
                # 21400 ≈ 95% moneyness — put skew candidate
                {
                    "strikePrice": 21400,
                    "expiryDate": "15-May-2026",
                    "CE": {
                        "openInterest": 10000,
                        "impliedVolatility": 16.0,
                        "lastPrice": 5.0,
                        "bidprice": 4.5,
                        "askPrice": 5.5,
                    },
                    "PE": {
                        "openInterest": 150000,
                        "impliedVolatility": 25.0,
                        "lastPrice": 180.0,
                        "bidprice": 179.5,
                        "askPrice": 180.5,
                    },
                },
                # 23600 ≈ 105% moneyness — call wing
                {
                    "strikePrice": 23600,
                    "expiryDate": "15-May-2026",
                    "CE": {
                        "openInterest": 70000,
                        "impliedVolatility": 16.5,
                        "lastPrice": 15.0,
                        "bidprice": 14.5,
                        "askPrice": 15.5,
                    },
                    "PE": {
                        "openInterest": 5000,
                        "impliedVolatility": 15.0,
                        "lastPrice": 250.0,
                        "bidprice": 249.5,
                        "askPrice": 250.5,
                    },
                },
                {
                    "strikePrice": 22500,
                    "expiryDate": "22-May-2026",
                    "CE": {
                        "openInterest": 50000,
                        "impliedVolatility": 19.5,
                        "lastPrice": 150.0,
                        "bidprice": 149.5,
                        "askPrice": 150.5,
                    },
                    "PE": {
                        "openInterest": 60000,
                        "impliedVolatility": 20.0,
                        "lastPrice": 130.0,
                        "bidprice": 129.5,
                        "askPrice": 130.5,
                    },
                },
            ],
        }
    }


# ─── IV History Collector ───────────────────────────────────────────────────


def test_collector_saves_valid_chain(mongo: _FakeMongoCol, sample_chain: Dict) -> None:
    _id = IVHistoryCollector.save(sample_chain, mongo)
    assert _id is not None
    assert len(mongo.docs) == 1
    doc = mongo.docs[0]
    assert doc["symbol"] == "NIFTY"
    assert doc["atm_iv"] == pytest.approx(0.1875, abs=1e-4)  # (18.5+19.0)/2 / 100
    assert doc["underlying"] == 22500


def test_collector_rapid_saves_have_distinct_ids(mongo: _FakeMongoCol, sample_chain: Dict) -> None:
    """Two saves in rapid succession must not silently overwrite each other."""
    id1 = IVHistoryCollector.save(sample_chain, mongo)
    id2 = IVHistoryCollector.save(sample_chain, mongo)
    assert id1 is not None
    assert id2 is not None
    assert id1 != id2
    assert len(mongo.docs) == 2


def test_collector_noop_on_none_mongo(sample_chain: Dict) -> None:
    assert IVHistoryCollector.save(sample_chain, None) is None


def test_collector_noop_on_unavailable_chain(mongo: _FakeMongoCol) -> None:
    chain = {"symbol": "NIFTY", "unavailable": True}
    assert IVHistoryCollector.save(chain, mongo) is None
    assert len(mongo.docs) == 0


def test_collector_noop_on_zero_iv(mongo: _FakeMongoCol) -> None:
    chain = {
        "symbol": "NIFTY",
        "atm_strike": 22500,
        "calls": [{"strike": 22500, "iv": 0}],
        "puts": [{"strike": 22500, "iv": 0}],
    }
    assert IVHistoryCollector.save(chain, mongo) is None


def test_collector_ensure_indexes(mongo: _FakeMongoCol) -> None:
    IVHistoryCollector.ensure_indexes(mongo)
    assert len(mongo.indexes) == 3


# ─── IV Rank ────────────────────────────────────────────────────────────────


def test_iv_rank_no_history(mongo: _FakeMongoCol) -> None:
    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "no_history"
    assert result["iv_rank"] is None
    assert result["history_days"] == 0


def test_iv_rank_collecting_insufficient_points(mongo: _FakeMongoCol) -> None:
    # Seed 10 points — below _MIN_HISTORY_POINTS (20)
    for i in range(10):
        mongo.docs.append({
            "_id": f"NIFTY:{i}",
            "ts": datetime.now(timezone.utc),
            "symbol": "NIFTY",
            "atm_iv": 0.15 + i * 0.001,
        })
    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "collecting"
    assert result["history_days"] == 10


def test_iv_rank_ready_with_sufficient_history(mongo: _FakeMongoCol) -> None:
    # Seed 25 points with a clear range
    base = datetime.now(timezone.utc)
    for i in range(25):
        # IV range: newest = 0.30 (high), oldest = 0.10 (low)
        iv = 0.30 - (i / 24) * 0.20
        mongo.docs.append({
            "_id": f"NIFTY:{i}",
            "ts": base - __import__('datetime').timedelta(days=i),
            "symbol": "NIFTY",
            "atm_iv": iv,
        })
    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "ready"
    assert result["history_days"] == 25
    # Current IV is the newest (i=0) → 0.30
    assert result["current_iv"] == pytest.approx(30.0, abs=0.1)
    # Rank should be near 100 (current ≈ max)
    assert result["iv_rank"] == pytest.approx(100.0, abs=5.0)
    # Percentile should also be high
    assert result["iv_percentile"] > 80.0
    assert result["history_high"] == pytest.approx(30.0, abs=0.1)
    assert result["history_low"] == pytest.approx(10.0, abs=0.1)


def test_iv_rank_flat_history_defaults_to_50(mongo: _FakeMongoCol) -> None:
    # All same IV → rank = 50 by convention; percentile = 100 (all <= current)
    base = datetime.now(timezone.utc)
    for i in range(25):
        mongo.docs.append({
            "_id": f"NIFTY:{i}",
            "ts": base - __import__('datetime').timedelta(days=i),
            "symbol": "NIFTY",
            "atm_iv": 0.20,
        })
    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "ready"
    assert result["iv_rank"] == 50.0
    assert result["iv_percentile"] == 100.0


# ─── Vol Surface ────────────────────────────────────────────────────────────


def test_vol_surface_multi_expiry(multi_expiry_raw: Dict) -> None:
    surf = vol_surface(multi_expiry_raw)
    assert not surf["unavailable"]
    assert surf["underlying"] == 22500
    assert len(surf["points"]) > 0
    # Should have points for both expiries
    expiries = {p["expiry"] for p in surf["points"]}
    assert "15-May-2026" in expiries
    assert "22-May-2026" in expiries


def test_vol_surface_atm_curve(multi_expiry_raw: Dict) -> None:
    surf = vol_surface(multi_expiry_raw)
    atm_curve = surf["atm_curve"]
    assert len(atm_curve) >= 1
    # Front expiry should have lower ATM IV than back (contango in this fixture)
    front = next(c for c in atm_curve if c["expiry"] == "15-May-2026")
    back = next(c for c in atm_curve if c["expiry"] == "22-May-2026")
    # (18.5+19.0)/2 = 18.75 vs (19.5+20.0)/2 = 19.75
    assert front["atm_iv_pct"] == pytest.approx(18.75, abs=0.1)
    assert back["atm_iv_pct"] == pytest.approx(19.75, abs=0.1)


def test_vol_surface_empty_data() -> None:
    surf = vol_surface({"records": {"data": []}})
    assert surf["unavailable"] is True
    assert surf["points"] == []


# ─── Term Structure ─────────────────────────────────────────────────────────


def test_term_structure_contango(multi_expiry_raw: Dict) -> None:
    surf = vol_surface(multi_expiry_raw)
    ts = term_structure(surf)
    assert ts["slope"] == "contango"
    assert ts["carry_bps_per_day"] > 0


def test_term_structure_single_expiry() -> None:
    surf = vol_surface({
        "records": {
            "underlyingValue": 22500,
            "expiryDates": ["15-May-2026"],
            "data": [
                {
                    "strikePrice": 22500,
                    "expiryDate": "15-May-2026",
                    "CE": {"impliedVolatility": 18.5, "lastPrice": 120.0, "openInterest": 100000},
                    "PE": {"impliedVolatility": 19.0, "lastPrice": 95.0, "openInterest": 120000},
                },
            ],
        }
    })
    ts = term_structure(surf)
    assert ts["slope"] == "single_expiry"
    assert ts["carry_bps_per_day"] is None


def test_term_structure_backwardation() -> None:
    surf = vol_surface({
        "records": {
            "underlyingValue": 22500,
            "expiryDates": ["15-May-2026", "22-May-2026"],
            "data": [
                {
                    "strikePrice": 22500,
                    "expiryDate": "15-May-2026",
                    "CE": {"impliedVolatility": 22.0, "lastPrice": 120.0, "openInterest": 100000},
                    "PE": {"impliedVolatility": 22.5, "lastPrice": 95.0, "openInterest": 120000},
                },
                {
                    "strikePrice": 22500,
                    "expiryDate": "22-May-2026",
                    "CE": {"impliedVolatility": 18.0, "lastPrice": 150.0, "openInterest": 50000},
                    "PE": {"impliedVolatility": 18.5, "lastPrice": 130.0, "openInterest": 60000},
                },
            ],
        }
    })
    ts = term_structure(surf)
    assert ts["slope"] == "backwardation"
    assert ts["carry_bps_per_day"] < 0


def test_term_structure_flat() -> None:
    surf = vol_surface({
        "records": {
            "underlyingValue": 22500,
            "expiryDates": ["15-May-2026", "22-May-2026"],
            "data": [
                {
                    "strikePrice": 22500,
                    "expiryDate": "15-May-2026",
                    "CE": {"impliedVolatility": 20.0, "lastPrice": 120.0, "openInterest": 100000},
                    "PE": {"impliedVolatility": 20.0, "lastPrice": 95.0, "openInterest": 120000},
                },
                {
                    "strikePrice": 22500,
                    "expiryDate": "22-May-2026",
                    "CE": {"impliedVolatility": 20.1, "lastPrice": 150.0, "openInterest": 50000},
                    "PE": {"impliedVolatility": 20.1, "lastPrice": 130.0, "openInterest": 60000},
                },
            ],
        }
    })
    ts = term_structure(surf)
    assert ts["slope"] == "flat"
    assert abs(ts["carry_bps_per_day"]) < 5.0


# ─── Skew Metrics ───────────────────────────────────────────────────────────


def test_skew_metrics_put_skew(multi_expiry_raw: Dict) -> None:
    sk = skew_metrics(multi_expiry_raw)
    assert not sk["unavailable"]
    assert sk["smile_shape"] == "put_skew"
    # Put skew index should be > 1 (puts are pricier than ATM)
    assert sk["put_skew_index"] > 1.0


def test_skew_metrics_unavailable_on_empty() -> None:
    sk = skew_metrics({"records": {"data": [], "underlyingValue": 0}})
    assert sk["unavailable"] is True


def test_skew_metrics_malformed_nse_data() -> None:
    """Non-dict CE/PE and string IV values must not crash."""
    raw = {
        "records": {
            "underlyingValue": "22500",
            "data": [
                {
                    "strikePrice": 22500,
                    "expiryDate": "15-May-2026",
                    "CE": "-",
                    "PE": {"impliedVolatility": "19.0", "lastPrice": 95.0},
                },
                {
                    "strikePrice": 22600,
                    "expiryDate": "15-May-2026",
                    "CE": {"impliedVolatility": 17.0, "lastPrice": 80.0},
                    "PE": {"impliedVolatility": "-", "lastPrice": 110.0},
                },
            ],
        }
    }
    # Should not raise
    surf = vol_surface(raw)
    assert not surf["unavailable"]
    sk = skew_metrics(raw)
    assert not sk["unavailable"]
    assert sk["atm_iv_pct"] == pytest.approx(19.0, abs=0.1)


def test_skew_metrics_risk_reversal_sign(multi_expiry_raw: Dict) -> None:
    sk = skew_metrics(multi_expiry_raw)
    # With put skew, risk reversal (call_105_iv - put_95_iv) should be negative
    assert sk["risk_reversal"] < 0


# ─── Integration: collector → history → rank ────────────────────────────────


def test_ensure_indexes_none() -> None:
    """ensure_indexes with None mongo_col must not raise."""
    IVHistoryCollector.ensure_indexes(None)


def test_vol_surface_missing_expiry_date() -> None:
    """Records without expiryDate should still surface via fallback."""
    raw = {
        "records": {
            "underlyingValue": 22500,
            "data": [
                {
                    "strikePrice": 22500,
                    "CE": {"impliedVolatility": 18.5, "lastPrice": 120.0},
                    "PE": {"impliedVolatility": 19.0, "lastPrice": 95.0},
                },
            ],
        }
    }
    surf = vol_surface(raw)
    assert not surf["unavailable"]
    assert len(surf["points"]) == 2


def test_iv_rank_history_with_none_iv(mongo: _FakeMongoCol) -> None:
    """Docs with atm_iv=None should be skipped gracefully."""
    base = datetime.now(timezone.utc)
    for i in range(25):
        mongo.docs.append({
            "_id": f"NIFTY:{i}",
            "ts": base - __import__('datetime').timedelta(days=i),
            "symbol": "NIFTY",
            "atm_iv": None if i == 0 else 0.20,
        })
    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "ready"
    assert result["history_days"] == 25


def test_full_pipeline_collector_to_rank(mongo: _FakeMongoCol, sample_chain: Dict) -> None:
    # Save 25 distinct snapshots
    base = datetime.now(timezone.utc)
    for i in range(25):
        chain = {
            **sample_chain,
            "atm_iv": None,  # not used directly
            "calls": [
                {"strike": 22500, "iv": 10.0 + i * 2.0, "ltp": 100, "oi": 1000},
                {"strike": 22600, "iv": 9.0 + i * 2.0, "ltp": 80, "oi": 800},
            ],
            "puts": [
                {"strike": 22500, "iv": 11.0 + i * 2.0, "ltp": 90, "oi": 1200},
                {"strike": 22400, "iv": 12.0 + i * 2.0, "ltp": 60, "oi": 900},
            ],
        }
        IVHistoryCollector.save(chain, mongo)

    result = iv_rank("NIFTY", history_days=252, mongo_col=mongo)
    assert result["status"] == "ready"
    assert result["history_days"] == 25
