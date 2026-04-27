"""
Forecasting primitives: ``ForecastResult`` record + ``Forecaster`` protocol.

Schema notes (load-bearing — read before changing):

- ``ForecastResult`` is a frozen dataclass so a fitted model can hand the
  caller an immutable result and not worry about downstream mutation.
- The ``regime_conditional`` field is ``Optional`` so individual components
  (GARCH, NeuralProphet) can return ``None`` and the ensemble fills it in.
- ``components`` carries per-sub-forecaster JSON-friendly snapshots so the
  API can render a transparency panel without re-running anything.
- ``calibration`` carries the OOS PI-coverage metrics produced by
  ``evaluator.py``. Populated by the ensemble at request time from a
  cached calibration record; never invented at the component level.
- All datetimes are tz-aware UTC. The API formats to IST at the edge.

Why a Protocol rather than an ABC: the three concrete forecasters live in
unrelated module trees (arch, NeuralProphet, in-house torch) and have
incompatible base classes upstream. A structural Protocol lets us keep the
contract loose and avoid forcing a multi-inheritance dance.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclasses.dataclass(frozen=True)
class _Band:
    """Point + 80/95 prediction interval. All fields in price units."""

    point: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ForecastResult:
    """One forecast record.

    Fields:
        symbol:            "RELIANCE"
        horizon_days:      forecast horizon in trading days
        as_of:             tz-aware UTC observation timestamp
        point:             central estimate at t+horizon
        lower_80/upper_80: 80% prediction interval at t+horizon
        lower_95/upper_95: 95% prediction interval at t+horizon
        regime_conditional: optional ``{"bull": _Band, "bear": _Band}`` —
                            ensemble fills this; components leave it None.
        components:        per-sub-forecaster snapshots, keyed by short slug
                            (``"patchtst"``, ``"neuralprophet"``, ``"garch_vol_1d"``).
        calibration:       OOS quality stats (``pi80_oos_coverage``, etc.).
        model:             "ensemble" | "patchtst" | "neuralprophet" | "garch"
    """

    symbol: str
    horizon_days: int
    as_of: _dt.datetime
    point: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    model: str
    regime_conditional: Optional[Dict[str, _Band]] = None
    components: Dict[str, Any] = dataclasses.field(default_factory=dict)
    calibration: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "symbol": self.symbol,
            "horizon_days": self.horizon_days,
            "as_of": self.as_of.isoformat(),
            "point": self.point,
            "lower_80": self.lower_80,
            "upper_80": self.upper_80,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "model": self.model,
            "components": dict(self.components),
            "calibration": dict(self.calibration),
        }
        if self.regime_conditional:
            d["regime_conditional"] = {
                k: (v.to_dict() if isinstance(v, _Band) else v)
                for k, v in self.regime_conditional.items()
            }
        else:
            d["regime_conditional"] = None
        return d

    # ── Internal helper used by ensemble + tests ─────────────────────────
    @staticmethod
    def make_band(
        point: float, lower_80: float, upper_80: float,
        lower_95: float, upper_95: float,
    ) -> _Band:
        return _Band(
            point=float(point),
            lower_80=float(lower_80), upper_80=float(upper_80),
            lower_95=float(lower_95), upper_95=float(upper_95),
        )


# Re-export the band type so ensemble.py can build regime_conditional.
Band = _Band


@runtime_checkable
class Forecaster(Protocol):
    """Structural protocol every concrete forecaster honours.

    Lifecycle:
        fit(df)        ingest historical OHLCV (and optional features)
        predict(h)     emit a ``ForecastResult`` at horizon h trading days

    Implementations must be safe to call ``predict`` multiple times after one
    ``fit``; they must not retrain implicitly on ``predict``.
    """

    name: str  # short slug used in components dict — e.g. "patchtst"

    def fit(self, df: pd.DataFrame) -> None:  # pragma: no cover - protocol
        ...

    def predict(self, horizon: int) -> ForecastResult:  # pragma: no cover - protocol
        ...
