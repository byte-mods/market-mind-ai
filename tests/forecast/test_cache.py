"""F7 verification: forecast cache hits, TTL semantics, graceful degrade."""
from __future__ import annotations

import datetime as dt

import pytest

from marketmind.ml.forecast.base import ForecastResult
from marketmind.ml.forecast.cache import (
    FORECAST_TTL_S,
    ForecastCache,
    get_forecast_cache,
)


def _result(symbol: str = "RELIANCE", horizon: int = 5, model: str = "ensemble",
            as_of: dt.datetime | None = None) -> ForecastResult:
    return ForecastResult(
        symbol=symbol, horizon_days=horizon,
        as_of=as_of or dt.datetime(2026, 4, 27, 12, tzinfo=dt.timezone.utc),
        point=2950.0,
        lower_80=2820.0, upper_80=3080.0,
        lower_95=2750.0, upper_95=3150.0,
        model=model,
    )


def test_forecast_cache_hit(fake_mongo_col) -> None:
    cache = ForecastCache(mongo_col=fake_mongo_col)
    cache.set(_result(), interval="day")
    hit = cache.get("RELIANCE", 5, "ensemble", interval="day")
    assert hit is not None
    assert hit.symbol == "RELIANCE"
    assert hit.point == 2950.0


def test_forecast_cache_ttl_index_set(fake_mongo_col) -> None:
    cache = ForecastCache(mongo_col=fake_mongo_col)
    cache.set(_result(), interval="day")
    # TTL index must be set at the longest cache lifetime
    assert ("as_of", FORECAST_TTL_S) in fake_mongo_col.indexes


def test_forecast_cache_miss(fake_mongo_col) -> None:
    cache = ForecastCache(mongo_col=fake_mongo_col)
    assert cache.get("UNKNOWN", 5, "ensemble") is None


def test_forecast_cache_returns_none_when_mongo_disabled() -> None:
    cache = ForecastCache(mongo_col=None)
    cache.set(_result())  # no-op
    assert cache.get("RELIANCE", 5, "ensemble") is None


def test_forecast_cache_respects_effective_until(fake_mongo_col) -> None:
    """An entry whose effective_until has passed must be treated as a miss."""
    stale = _result(as_of=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2))
    cache = ForecastCache(mongo_col=fake_mongo_col)
    cache.set(stale, interval="1min")  # 5-minute TTL → already expired
    assert cache.get("RELIANCE", 5, "ensemble", interval="1min") is None


def test_forecast_cache_per_interval_keying(fake_mongo_col) -> None:
    """Same symbol+horizon+model on different intervals must NOT collide."""
    cache = ForecastCache(mongo_col=fake_mongo_col)
    daily = _result()
    intra = _result()
    cache.set(daily, interval="day")
    cache.set(intra, interval="5min")
    assert len(fake_mongo_col) == 2


def test_forecast_cache_re_set_overwrites(fake_mongo_col) -> None:
    cache = ForecastCache(mongo_col=fake_mongo_col)
    cache.set(_result(), interval="day")
    fresh = ForecastResult(
        symbol="RELIANCE", horizon_days=5,
        as_of=dt.datetime(2026, 4, 27, 13, tzinfo=dt.timezone.utc),
        point=3000.0, lower_80=2900, upper_80=3100,
        lower_95=2850, upper_95=3150, model="ensemble",
    )
    cache.set(fresh, interval="day")
    hit = cache.get("RELIANCE", 5, "ensemble", interval="day")
    assert hit is not None
    assert hit.point == 3000.0
    assert len(fake_mongo_col) == 1


def test_forecast_cache_singleton(fake_mongo_col) -> None:
    a = get_forecast_cache(mongo_col=fake_mongo_col)
    b = get_forecast_cache()
    assert a is b
