"""
MarketMind AI - Alt-Data sources (W2.3)

India-flavoured alternative data signals: Reddit retail sentiment,
ValuePickr forum activity, SIAM auto sales, GST collections, IIP/CPI,
and Google Trends ticker interest. All sources implement the
``AltDataSource`` protocol and emit ``AltSignal`` records.

Exports:
    AltSignal       - immutable signal record
    AltDataSource   - ABC every source implements
    safe_fetch      - decorator that swallows exceptions and returns []
"""
from marketmind.core.altdata.base import AltDataSource, AltSignal, safe_fetch

__all__ = ["AltDataSource", "AltSignal", "safe_fetch"]
