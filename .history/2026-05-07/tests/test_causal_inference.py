"""Unit tests for CausalInferenceEngine validation and whatif formatting."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from marketmind.ml.causal.inference import CausalInferenceEngine
from marketmind.ml.causal.bayes_net import CausalBayesNet
from marketmind.ml.causal.data_layer import CausalDataCollector


def test_whatif_invalid_intervention_raises() -> None:
    """Intervening on a non-intervention node must raise ValueError."""
    collector = CausalDataCollector()
    net = CausalBayesNet()
    engine = CausalInferenceEngine(collector=collector, net=net)
    engine.ensure_trained()
    with pytest.raises(ValueError, match="Invalid intervention"):
        engine.whatif({"banknifty": 40000}, "nifty500")


def test_whatif_invalid_target_raises() -> None:
    """Targeting a non-target node must raise ValueError."""
    collector = CausalDataCollector()
    net = CausalBayesNet()
    engine = CausalInferenceEngine(collector=collector, net=net)
    engine.ensure_trained()
    with pytest.raises(ValueError, match="Invalid target"):
        engine.whatif({"repo_rate": 5.5}, "repo_rate")


def test_whatif_result_has_required_keys() -> None:
    """A valid whatif must return the contract keys."""
    collector = CausalDataCollector()
    net = CausalBayesNet()
    engine = CausalInferenceEngine(collector=collector, net=net)
    engine.ensure_trained()
    result = engine.whatif({"repo_rate": 5.5}, "banknifty")
    required = {
        "target", "target_estimate", "target_current", "delta",
        "delta_pct", "confidence", "intervention", "paths", "dag_edges",
    }
    assert required.issubset(result.keys())
    assert isinstance(result["confidence"], float)
    assert 0.3 <= result["confidence"] <= 0.95


def test_get_nodes_includes_parents() -> None:
    """get_nodes should append 'parents' to each node entry."""
    collector = CausalDataCollector()
    net = CausalBayesNet()
    engine = CausalInferenceEngine(collector=collector, net=net)
    engine.ensure_trained()
    nodes = engine.get_nodes()
    assert len(nodes) > 0
    for n in nodes:
        assert "parents" in n
        assert isinstance(n["parents"], list)


def test_network_summary_has_counts() -> None:
    """Summary must contain node/edge counts."""
    collector = CausalDataCollector()
    net = CausalBayesNet()
    engine = CausalInferenceEngine(collector=collector, net=net)
    engine.ensure_trained()
    summary = engine.get_network_summary()
    assert "node_count" in summary
    assert "edge_count" in summary
    assert summary["node_count"] > 0
