"""Real-server schema-drift lock for /api/causal/* routes."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

server = pytest.importorskip("server")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


class _StubEngine:
    def __init__(self) -> None:
        self.last_whatif: tuple = ()

    def ensure_trained(self) -> None:
        pass

    def get_nodes(self) -> List[Dict[str, Any]]:
        return [{"id": "repo_rate", "label": "Repo", "parents": []}]

    def get_network_summary(self) -> Dict[str, Any]:
        return {"node_count": 1, "edge_count": 0, "edges": []}

    def whatif(self, intervention: Dict[str, float], target: str) -> Dict[str, Any]:
        self.last_whatif = (intervention, target)
        return {
            "target": target,
            "target_estimate": 100.0,
            "target_current": 90.0,
            "delta": 10.0,
            "delta_pct": 11.1,
            "confidence": 0.8,
            "intervention": intervention,
            "paths": [],
            "dag_edges": [],
        }


def test_real_server_causal_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubEngine()
    monkeypatch.setattr(
        server, "get_causal_inference_engine",
        lambda: stub,
        raising=False,
    )
    client = TestClient(server.app)
    resp = client.get("/api/causal/nodes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes"][0]["id"] == "repo_rate"
    assert "summary" in body


def test_real_server_causal_whatif(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubEngine()
    monkeypatch.setattr(
        server, "get_causal_inference_engine",
        lambda: stub,
        raising=False,
    )
    client = TestClient(server.app)
    resp = client.post("/api/causal/whatif", json={
        "intervention": {"repo_rate": 5.5},
        "target": "banknifty",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "banknifty"
    assert stub.last_whatif == ({"repo_rate": 5.5}, "banknifty")
