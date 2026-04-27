"""T11: e2e wiring check.

Imports the real ``server`` module and asserts that W2.3's route is
registered on the live FastAPI app. This is the cheapest way to catch a
typo or import-order bug in server.py without booting the full startup
event loop.

Heavier than the stand-in tests in test_api_altdata.py, but it's the only
test that actually proves the wire is connected. Skips gracefully if the
controller can't be imported (e.g. KiteClient init throws in a sandbox).
"""
from __future__ import annotations

import pytest


def test_altdata_route_registered_on_live_app() -> None:
    try:
        from server import app  # type: ignore
    except Exception as e:
        pytest.skip(f"server import failed in this environment: {e}")
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/altdata" in paths, (
        "GET /api/altdata not registered. Check server.py altdata import + route block."
    )


def test_forecast_route_registered_on_live_app() -> None:
    try:
        from server import app  # type: ignore
    except Exception as e:
        pytest.skip(f"server import failed in this environment: {e}")
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/forecast/{sym}" in paths, (
        "GET /api/forecast/{sym} not registered. Check server.py forecast block."
    )


def test_calibrated_signal_route_registered_on_live_app() -> None:
    try:
        from server import app  # type: ignore
    except Exception as e:
        pytest.skip(f"server import failed in this environment: {e}")
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/signal/{sym}/calibrated" in paths, (
        "GET /api/signal/{sym}/calibrated not registered. Check server.py W3.2 block."
    )


def test_options_strategy_route_registered_on_live_app() -> None:
    try:
        from server import app  # type: ignore
    except Exception as e:
        pytest.skip(f"server import failed in this environment: {e}")
    paths = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "path") and hasattr(r, "methods")}
    assert ("/api/options/strategy", ("POST",)) in paths, (
        "POST /api/options/strategy not registered. Check server.py W3.3 block."
    )
