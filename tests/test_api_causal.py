"""Route-mirror tests for /api/causal/* endpoints."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


class _CausalWhatIfRequest(BaseModel):
    intervention: Dict[str, float]
    target: str


def _make_app() -> FastAPI:
    """Minimal FastAPI app mirroring the causal routes."""
    app = FastAPI()

    @app.get("/api/causal/nodes")
    async def causal_nodes():
        return JSONResponse({
            "nodes": [
                {"id": "repo_rate", "label": "Repo", "parents": []},
                {"id": "banknifty", "label": "Bank Nifty", "parents": ["repo_rate"]},
            ],
            "summary": {"node_count": 2, "edge_count": 1, "edges": [["repo_rate", "banknifty"]]},
        })

    @app.post("/api/causal/whatif")
    async def causal_whatif(req: _CausalWhatIfRequest):
        return JSONResponse({
            "target": req.target,
            "target_estimate": 45000.0,
            "target_current": 44600.0,
            "delta": 400.0,
            "delta_pct": 0.9,
            "confidence": 0.72,
            "intervention": req.intervention,
            "paths": [],
            "dag_edges": [["repo_rate", "banknifty"]],
        })

    return app


def test_causal_nodes_returns_shape() -> None:
    client = TestClient(_make_app())
    resp = client.get("/api/causal/nodes")
    assert resp.status_code == 200
    body = resp.json()
    assert "nodes" in body
    assert "summary" in body
    assert body["summary"]["node_count"] == 2
    assert body["nodes"][1]["parents"] == ["repo_rate"]


def test_causal_whatif_returns_contract() -> None:
    client = TestClient(_make_app())
    resp = client.post("/api/causal/whatif", json={
        "intervention": {"repo_rate": 5.5},
        "target": "banknifty",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "banknifty"
    assert body["target_estimate"] == 45000.0
    assert body["delta"] == 400.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert "intervention" in body
    assert "dag_edges" in body


def test_causal_whatif_bad_body_422() -> None:
    client = TestClient(_make_app())
    resp = client.post("/api/causal/whatif", json={"target": "banknifty"})
    assert resp.status_code == 422
