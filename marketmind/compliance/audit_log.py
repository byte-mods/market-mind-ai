"""Compliance audit-log store (W5.3 T1).

Append-only Mongo-backed record of every pre-trade decision and order
attempt. No TTL — this is a regulatory artefact and must persist. Reads
support symbol filter, since-cutoff filter, and a hard ``limit`` cap to
bound payload size for the route layer.

Failure semantics mirror ``forecast_cache``:
    mongo_col is None  →  append is a no-op, query returns []
    upstream Mongo throws → logged at WARNING, treated as miss

Schema (one Mongo doc per entry):
    _id              str   "{symbol}:{ts_iso}:{rand6}"  (write-collision-safe)
    ts               datetime (UTC, tz-aware)
    symbol           str   uppercase NSE ticker
    transaction_type str   BUY | SELL
    quantity         float
    price            float
    decision         str   ALLOW | BLOCK | WARN
    reasons          list[str]
    source           str   pretrade | order_attempt
"""
from __future__ import annotations

import datetime as _dt
import logging
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Public decision sentinels — matched by pretrade_check.PretradeDecision.
DECISION_ALLOW = "ALLOW"
DECISION_BLOCK = "BLOCK"
DECISION_WARN = "WARN"
_VALID_DECISIONS = {DECISION_ALLOW, DECISION_BLOCK, DECISION_WARN}

SOURCE_PRETRADE = "pretrade"
SOURCE_ORDER_ATTEMPT = "order_attempt"
_VALID_SOURCES = {SOURCE_PRETRADE, SOURCE_ORDER_ATTEMPT}

# Hard cap on `query(limit=...)` — protects route from runaway payloads.
MAX_QUERY_LIMIT = 1000


@dataclass(frozen=True)
class AuditLogEntry:
    """One audit row. Frozen so callers can't mutate after handing in."""

    ts: _dt.datetime
    symbol: str
    transaction_type: str
    quantity: float
    price: float
    decision: str
    reasons: List[str] = field(default_factory=list)
    source: str = SOURCE_PRETRADE

    def __post_init__(self) -> None:
        if self.decision not in _VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(_VALID_DECISIONS)}, got {self.decision!r}"
            )
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(_VALID_SOURCES)}, got {self.source!r}"
            )
        if self.ts.tzinfo is None:
            raise ValueError("ts must be tz-aware (use datetime.now(timezone.utc))")

    def to_doc(self) -> Dict[str, Any]:
        d = asdict(self)
        d["symbol"] = self.symbol.upper()
        d["transaction_type"] = self.transaction_type.upper()
        d["reasons"] = list(self.reasons)
        # Stable per-write _id; ts_iso prefix keeps natural sort by time.
        d["_id"] = f"{d['symbol']}:{self.ts.isoformat()}:{secrets.token_hex(3)}"
        return d


class AuditLogStore:
    """Thin Mongo wrapper. Pass ``mongo_col=None`` to disable persistence."""

    def __init__(self, mongo_col: Any = None) -> None:
        self.mongo_col = mongo_col
        self._indexes_ensured = False

    def append(self, entry: AuditLogEntry) -> Optional[str]:
        """Persist one entry; returns the assigned ``_id`` or None on no-op/error."""
        if self.mongo_col is None:
            return None
        self._ensure_indexes()
        doc = entry.to_doc()
        try:
            self.mongo_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            return doc["_id"]
        except Exception as e:  # noqa: BLE001
            logger.warning("audit_log append failed: %s", e)
            return None

    def query(
        self,
        symbol: Optional[str] = None,
        since: Optional[_dt.datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return entries newest-first. ``limit`` capped to ``MAX_QUERY_LIMIT``."""
        if self.mongo_col is None:
            return []
        if limit < 1:
            return []
        capped = min(limit, MAX_QUERY_LIMIT)
        if since is not None and since.tzinfo is None:
            raise ValueError("since must be tz-aware")
        try:
            docs: Iterable[Dict[str, Any]] = self.mongo_col.find({})
        except Exception as e:  # noqa: BLE001
            logger.warning("audit_log query failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        sym_norm = symbol.upper() if symbol else None
        for d in docs:
            if sym_norm and d.get("symbol") != sym_norm:
                continue
            ts = d.get("ts")
            if since is not None and isinstance(ts, _dt.datetime):
                ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)
                if ts_aware < since:
                    continue
            out.append(d)
        # Newest-first by ts; entries with non-datetime ts sink to the end.
        out.sort(key=lambda x: x.get("ts") or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc), reverse=True)
        return out[:capped]

    def _ensure_indexes(self) -> None:
        if self._indexes_ensured or self.mongo_col is None:
            return
        try:
            # No expireAfterSeconds — audit log is permanent. Index for query speed.
            self.mongo_col.create_index("symbol")
            self.mongo_col.create_index("ts")
            self._indexes_ensured = True
        except Exception as e:  # noqa: BLE001
            logger.debug("audit_log index ensure failed: %s", e)
