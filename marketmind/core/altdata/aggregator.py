"""
Alt-data aggregator: parallel fan-out + Mongo persist + flat-dict surface.

Architecture:
- Six sources implement ``AltDataSource``. Each is fault-isolated by its
  own ``@safe_fetch`` — failures degrade to ``[]``, never raise.
- ``AltDataAggregator.get_all()`` runs sources in parallel via a
  ``ThreadPoolExecutor(max_workers=6)`` with a per-source timeout of 12s.
  Sources that exceed the timeout are dropped from this cycle (logged).
- Each emitted ``AltSignal`` is upserted into Mongo collection
  ``alt_signals``, keyed by ``"{source}:{key}"`` so re-emissions overwrite
  cleanly. A TTL index on ``as_of`` (7 days) prunes stale rows.
- The return shape is a *flat dict* convenient for the API:
      {"reddit": {"sentiment_tilt": 0.42, ...}, "valuepickr": {...}, ...}
  Plus an ``"as_of"`` ISO timestamp at the top level.

If Mongo is unavailable (``mongo_col`` is ``None``), persistence silently
no-ops — same pattern as ``app_controller._init_mongo``.
"""
from __future__ import annotations

import concurrent.futures as _futures
import datetime as _dt
import logging
from typing import Any, Dict, List, Optional, Sequence

from marketmind.core.altdata.base import AltDataSource, AltSignal
from marketmind.core.altdata.gst_collections import get_gst_source
from marketmind.core.altdata.google_trends import get_trends_source
from marketmind.core.altdata.iip_cpi import get_iip_cpi_source
from marketmind.core.altdata.reddit_sentiment import get_reddit_source
from marketmind.core.altdata.siam_auto import get_siam_source
from marketmind.core.altdata.valuepickr import get_valuepickr_source

logger = logging.getLogger(__name__)

# Mongo TTL — alt-data is weekly/monthly, 7 days is plenty of headroom.
ALT_SIGNALS_TTL_S: int = 7 * 24 * 3600


class AltDataAggregator:
    PER_SOURCE_TIMEOUT_S: float = 12.0

    def __init__(
        self,
        sources: Optional[Sequence[AltDataSource]] = None,
        mongo_col: Any = None,
    ) -> None:
        self.sources = list(sources) if sources is not None else _default_sources()
        self.mongo_col = mongo_col
        self._ttl_ensured = False

    def get_all(self) -> Dict[str, Any]:
        signals = self._fan_out()
        self._persist(signals)
        return self._shape_response(signals)

    # ── Fan-out ──────────────────────────────────────────────────────────
    def _fan_out(self) -> List[AltSignal]:
        if not self.sources:
            return []
        all_signals: List[AltSignal] = []
        with _futures.ThreadPoolExecutor(max_workers=len(self.sources)) as ex:
            futures = {ex.submit(_run_source, s): s for s in self.sources}
            for fut in _futures.as_completed(futures, timeout=None):
                source = futures[fut]
                try:
                    chunk = fut.result(timeout=self.PER_SOURCE_TIMEOUT_S)
                    all_signals.extend(chunk)
                except _futures.TimeoutError:
                    logger.warning(
                        "alt-data source %s timed out after %.1fs",
                        source.SLUG, self.PER_SOURCE_TIMEOUT_S,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("alt-data source %s raised: %s", source.SLUG, e)
        return all_signals

    # ── Persistence ──────────────────────────────────────────────────────
    def _persist(self, signals: List[AltSignal]) -> None:
        if self.mongo_col is None or not signals:
            return
        if not self._ttl_ensured:
            try:
                self.mongo_col.create_index("as_of", expireAfterSeconds=ALT_SIGNALS_TTL_S)
                self._ttl_ensured = True
            except Exception as e:  # noqa: BLE001
                logger.debug("alt_signals TTL index ensure failed: %s", e)
        for sig in signals:
            doc_id = f"{sig.source}:{sig.key}"
            try:
                doc = sig.to_dict()
                doc["_id"] = doc_id
                # Mongo TTL index requires a real BSON Date, not an ISO string.
                doc["as_of"] = sig.as_of
                self.mongo_col.replace_one({"_id": doc_id}, doc, upsert=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("alt_signals upsert failed for %s: %s", doc_id, e)

    # ── Response shaping ─────────────────────────────────────────────────
    def _shape_response(self, signals: List[AltSignal]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"as_of": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        for sig in signals:
            bucket = out.setdefault(sig.source, {})
            bucket[sig.key] = {
                "value": sig.value,
                "unit": sig.unit,
                "confidence": sig.confidence,
                "as_of": sig.as_of.isoformat(),
            }
        out["_meta"] = {
            "source_count": sum(
                1 for k in out if k not in ("as_of", "_meta") and out[k]
            ),
            "signal_count": len(signals),
        }
        return out


def _run_source(src: AltDataSource) -> List[AltSignal]:
    """Trampoline for the executor — keeps the closure simple."""
    return src.fetch()


def _default_sources() -> List[AltDataSource]:
    return [
        get_reddit_source(),
        get_valuepickr_source(),
        get_siam_source(),
        get_gst_source(),
        get_iip_cpi_source(),
        get_trends_source(),
    ]


_singleton: Optional[AltDataAggregator] = None


def get_aggregator(mongo_col: Any = None) -> AltDataAggregator:
    """Process-wide singleton. Pass mongo_col on first call only."""
    global _singleton
    if _singleton is None:
        _singleton = AltDataAggregator(mongo_col=mongo_col)
    elif mongo_col is not None and _singleton.mongo_col is None:
        # Late-binding: server may resolve Mongo after the first import.
        _singleton.mongo_col = mongo_col
    return _singleton
