"""
MarketMind AI - Forecasting models layer (W3.1)

Three forecasters share one ``Forecaster`` protocol and emit
``ForecastResult`` records. The ensemble combines them into a single
calibrated forecast with regime-conditional bull/bear branches.

Components:
    GARCH(1,1)        next-day conditional volatility (drives PI bands)
    NeuralProphet     medium-horizon trend (point + 80/95 PI)
    PatchTST          multivariate transformer (point + bootstrap CI)
    Ensemble          weighted blend, regime-aware

Public surface:
    ForecastResult    immutable record; ``.to_dict()`` for JSON
    Forecaster        protocol — ``fit(df)``, ``predict(horizon) -> ForecastResult``
    forecast(symbol, horizon, model)  high-level façade used by the API route
"""
from marketmind.ml.forecast.base import ForecastResult, Forecaster

__all__ = ["ForecastResult", "Forecaster"]
