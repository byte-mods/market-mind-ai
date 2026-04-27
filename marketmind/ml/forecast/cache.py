"""
Forecast cache (Mongo-backed) — TTL-aware, graceful-degrade.

Key design:
    _id = "{symbol}:{horizon}:{model}:{interval}"
    TTL on `as_of` (BSON Date), expireAfterSeconds set per write based on interval.

Why per-write TTL rather than a single TTL index: Mongo TTL indexes apply one
expireAfterSeconds value globally to the whole collection. We use the same
convention as the alt-data aggregator — index `as_of` once at the longest
horizon (24h), and write `effective_until` into each doc so reads can self-
filter shorter intraday entries.

Failure semantics:
    mongo_col is None  →  read returns None, write is a no-op (no exceptions)
    upstream Mongo throws → logged at WARNING, treated as a miss
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, Optional

from marketmind.ml.forecast.base import ForecastResult

logger = logging.getLogger(__name__)

# Top-level TTL for the index (longest possible cache lifetime).
FORECAST_TTL_S: int = 24 * 3600

# Per-interval effective lifetimes (seconds).
INTERVAL_TTL_S: Dict[str, int] = {
    "1min":  60 * 5,        # intraday minute bars: 5min
    "5min":  60 * 30,
    "15min": 60 * 60,
    "60min": 60 * 60,
    "day":   24 * 3600,
    "daily": 24 * 3600,
}


class ForecastCache:
    """Thin Mongo wrapper. Pass ``mongo_col=None`` to disable persistence."""

    def __init__(self, mongo_col: Any = None) -> None:
        self.mongo_col = mongo_col
        self._ttl_ensured = False

    def get(
        self,
        symbol: str,
        horizon: int,
        model: str,
        interval: str = "day",
    ) -> Optional[ForecastResult]:
        if self.mongo_col is None:
            return None
        doc_id = self._key(symbol, horizon, model, interval)
        try:
            doc = self.mongo_col.find_one({"_id": doc_id})
        except Exception as e:  # noqa: BLE001
            logger.warning("forecast_cache get failed: %s", e)
            return None
        if not doc:
            return None
        # Honour effective_until (per-interval freshness)
        eu = doc.get("effective_until")
        now = _dt.datetime.now(_dt.timezone.utc)
        if isinstance(eu, _dt.datetime):
            eu_aware = eu if eu.tzinfo else eu.replace(tzinfo=_dt.timezone.utc)
            if eu_aware < now:
                return None
        return _doc_to_result(doc)

    def set(
        self,
        result: ForecastResult,
        interval: str = "day",
    ) -> None:
        if self.mongo_col is None or result is None:
            return
        if not self._ttl_ensured:
            try:
                self.mongo_col.create_index("as_of", expireAfterSeconds=FORECAST_TTL_S)
                self._ttl_ensured = True
            except Exception as e:  # noqa: BLE001
                logger.debug("forecast_cache TTL ensure failed: %s", e)

        ttl_s = INTERVAL_TTL_S.get(interval, INTERVAL_TTL_S["day"])
        doc_id = self._key(result.symbol, result.horizon_days, result.model, interval)
        doc = result.to_dict()
        doc["_id"] = doc_id
        # Mongo TTL needs real Date for `as_of` (overwrite the ISO string from to_dict).
        doc["as_of"] = result.as_of
        doc["effective_until"] = result.as_of + _dt.timedelta(seconds=ttl_s)
        doc["interval"] = interval
        try:
            self.mongo_col.replace_one({"_id": doc_id}, doc, upsert=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("forecast_cache set failed for %s: %s", doc_id, e)

    @staticmethod
    def _key(symbol: str, horizon: int, model: str, interval: str) -> str:
        return f"{symbol.upper()}:{horizon}:{model}:{interval}"


def _doc_to_result(doc: Dict[str, Any]) -> ForecastResult:
    """Round-trip a Mongo doc back into ``ForecastResult``.

    The cache layer is intentionally lossy on ``regime_conditional`` (we store
    the dict-form, not the dataclass form) since callers — the API route —
    serialise it back to JSON immediately. We don't reconstruct ``Band``
    instances from cached docs because no caller needs them.
    """
    as_of = doc.get("as_of")
    if isinstance(as_of, str):
        try:
            as_of = _dt.datetime.fromisoformat(as_of)
        except ValueError:
            as_of = _dt.datetime.now(_dt.timezone.utc)
    elif isinstance(as_of, _dt.datetime) and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=_dt.timezone.utc)
    return ForecastResult(
        symbol=doc.get("symbol", ""),
        horizon_days=int(doc.get("horizon_days", 1)),
        as_of=as_of,
        point=float(doc.get("point", 0.0)),
        lower_80=float(doc.get("lower_80", 0.0)),
        upper_80=float(doc.get("upper_80", 0.0)),
        lower_95=float(doc.get("lower_95", 0.0)),
        upper_95=float(doc.get("upper_95", 0.0)),
        model=str(doc.get("model", "ensemble")),
        regime_conditional=doc.get("regime_conditional"),
        components=dict(doc.get("components") or {}),
        calibration=dict(doc.get("calibration") or {}),
    )


_singleton: Optional[ForecastCache] = None


def get_forecast_cache(mongo_col: Any = None) -> ForecastCache:
    global _singleton
    if _singleton is None:
        _singleton = ForecastCache(mongo_col=mongo_col)
    elif mongo_col is not None and _singleton.mongo_col is None:
        _singleton.mongo_col = mongo_col
    return _singleton
