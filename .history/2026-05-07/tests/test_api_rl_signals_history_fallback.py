"""Verify /api/rl/signals falls back to persisted history with correct field names.

The frontend `renderSignalHistory` expects `entry_price`, `exit_price`,
`stop_loss`, and `rationale`.  When the trained-model path returns empty,
the route MUST serve historical SQLite rows — not the multiframe scanner
output which uses `entry`, `target`, `sl`, and `reason`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse


class _StubDB:
    def __init__(self, signals: List[Dict[str, Any]]) -> None:
        self._signals = signals

    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._signals[:limit]


class _StubController:
    def __init__(self, live_signals: List[Dict[str, Any]] = None,
                 history_signals: List[Dict[str, Any]] = None) -> None:
        self.live = live_signals or []
        self.db = _StubDB(history_signals or [])

    def get_rl_signals(self) -> List[Dict[str, Any]]:
        return self.live


def test_rl_signals_fallback_uses_db_history_field_names() -> None:
    """When live signals empty, route returns historical rows with UI-matching keys."""
    from server import app, controller

    history = [
        {
            "timestamp": "2026-05-07T10:00:00",
            "symbol": "RELIANCE",
            "action": "BUY",
            "confidence": 0.72,
            "entry_price": 2845.5,
            "exit_price": 2888.0,
            "stop_loss": 2820.0,
            "rationale": "RSI oversold bounce",
        }
    ]
    stub = _StubController(live_signals=[], history_signals=history)

    # patch module-level controller binding (looked up at call time)
    import server
    orig = server.controller
    server.controller = stub  # type: ignore[misc]
    try:
        client = TestClient(app)
        resp = client.get("/api/rl/signals")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        row = body[0]
        assert row["symbol"] == "RELIANCE"
        assert row["entry_price"] == 2845.5
        assert row["exit_price"] == 2888.0
        assert row["stop_loss"] == 2820.0
        assert row["rationale"] == "RSI oversold bounce"
        # ensure multiframe keys are NOT present
        assert "entry" not in row
        assert "target" not in row
        assert "sl" not in row
        assert "reason" not in row
    finally:
        server.controller = orig


def test_rl_signals_live_wins_when_present() -> None:
    """When live signals exist, they are returned directly."""
    from server import app

    live = [
        {
            "timestamp": "2026-05-07T11:00:00",
            "symbol": "TCS",
            "action": "SELL",
            "confidence": 0.65,
            "entry_price": 3920.0,
            "exit_price": 3870.0,
            "stop_loss": 3950.0,
            "rationale": "MACD bearish",
        }
    ]
    stub = _StubController(live_signals=live, history_signals=[])

    import server
    orig = server.controller
    server.controller = stub  # type: ignore[misc]
    try:
        client = TestClient(app)
        resp = client.get("/api/rl/signals")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["symbol"] == "TCS"
    finally:
        server.controller = orig
