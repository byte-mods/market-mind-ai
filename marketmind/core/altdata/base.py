"""
Alt-data primitives: ``AltSignal`` record, ``AltDataSource`` ABC, ``safe_fetch``
decorator. Every alt-data fetcher in this package implements ``AltDataSource``
and decorates its ``fetch`` with ``@safe_fetch`` so a single misbehaving source
can never crash the dashboard.

Design choices (load-bearing — read before changing):

- ``AltSignal`` is a frozen ``@dataclass`` so signals are hashable, JSON-friendly,
  and impossible to mutate after the source emits them. Aggregator persists
  them verbatim.
- ``confidence`` is mandatory (0.0..1.0): each source self-rates how trustworthy
  the value is right now. Maintained-table sources lower their confidence as
  the table goes stale.
- ``safe_fetch`` returns ``[]`` on any exception and logs at WARNING. Returning
  ``[]`` (not ``None``) lets callers iterate without nil-checks. The aggregator
  treats an empty result as "source unavailable", not as a successful zero.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AltSignal:
    """One alt-data observation.

    Fields:
        source: short slug, e.g. ``"reddit"``, ``"siam"``, ``"gst"``.
        key:    sub-identifier within the source, e.g. ``"sentiment_tilt"``,
                ``"passenger_vehicles_yoy"``, ``"top_thread_TATAMOTORS"``.
        value:  float for numeric signals, str for categorical.
        unit:   free-text unit, e.g. ``"pct"``, ``"count"``, ``"inr_cr"``,
                ``"label"``. Empty string for unitless.
        as_of:  observation timestamp (UTC, tz-aware).
        confidence: 0.0..1.0 self-rated confidence. Sources should drop this
                    below 0.5 when serving stale fallback data.
        raw:    source-specific debug blob; aggregator persists it for audit.
    """

    source: str
    key: str
    value: Any
    unit: str
    as_of: _dt.datetime
    confidence: float
    raw: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable projection."""
        d = dataclasses.asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


class AltDataSource(ABC):
    """ABC every alt-data source implements.

    Subclasses define:
        SLUG: short stable identifier used as ``AltSignal.source`` and in
              the aggregator's flat result dict.
        CACHE_TTL_S: in-memory cache TTL. Most alt-data is weekly/monthly,
              so a 6h floor is fine.
        fetch(): returns a list of ``AltSignal``. Decorated with ``@safe_fetch``
              by subclasses so failures degrade to ``[]`` rather than raise.
    """

    SLUG: str = ""
    CACHE_TTL_S: int = 6 * 3600

    @abstractmethod
    def fetch(self) -> List[AltSignal]:  # pragma: no cover - abstract
        """Fetch the latest signals. Must never raise — decorate with safe_fetch."""


def safe_fetch(fn: Callable[..., List[AltSignal]]) -> Callable[..., List[AltSignal]]:
    """Decorator: swallow exceptions, log once, return ``[]``.

    Used on every concrete ``AltDataSource.fetch`` implementation so one
    flaky source (Reddit 429, ValuePickr 503, etc.) cannot take down the
    aggregator or the API route.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> List[AltSignal]:
        try:
            result = fn(*args, **kwargs)
            return result if result is not None else []
        except Exception as e:  # noqa: BLE001 — broad by design
            self_repr = type(args[0]).__name__ if args else fn.__qualname__
            logger.warning(
                "alt-data fetch failed in %s.%s: %s", self_repr, fn.__name__, e
            )
            return []

    return wrapper
