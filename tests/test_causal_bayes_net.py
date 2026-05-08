"""Unit tests for CausalBayesNet structure learning + counterfactuals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketmind.ml.causal.bayes_net import CausalBayesNet


def _synthetic_panel(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a simple chain: a → b → c with noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = 0.5 * a + rng.normal(0, 0.5, n)
    c = 0.3 * b + rng.normal(0, 0.5, n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_learn_structure_produces_dag() -> None:
    """PC must return a DAG (no cycles) even on noisy data."""
    panel = _synthetic_panel(n=200)
    net = CausalBayesNet()
    net.learn_structure(panel, significance_level=0.10)
    assert net.dag is not None
    import networkx as nx
    assert nx.is_directed_acyclic_graph(net.dag)
    # Should have at least some edges on correlated data
    assert net.dag.number_of_edges() >= 1


def test_learn_structure_breaks_cycles() -> None:
    """If PC creates a cycle on noisy data, we break it and end up with a DAG."""
    rng = np.random.default_rng(7)
    # Strongly correlated variables can confuse PC into cycles
    x = rng.normal(0, 1, 120)
    y = 0.8 * x + rng.normal(0, 0.3, 120)
    z = 0.8 * y + 0.3 * x + rng.normal(0, 0.3, 120)
    panel = pd.DataFrame({"x": x, "y": y, "z": z})
    net = CausalBayesNet()
    net.learn_structure(panel, significance_level=0.10)
    import networkx as nx
    assert nx.is_directed_acyclic_graph(net.dag)


def test_fit_parameters_produces_r2() -> None:
    """After fitting, each node should have an R² between 0 and 1."""
    panel = _synthetic_panel(n=500)
    net = CausalBayesNet()
    # Manually set the DAG so this test is deterministic (tests fitting, not PC)
    import networkx as nx
    net.dag = nx.DiGraph()
    net.dag.add_edges_from([("a", "b"), ("b", "c")])
    net._var_order = list(nx.topological_sort(net.dag))
    net._panel = panel.copy()
    net.fit_parameters()
    for node in ["a", "b", "c"]:
        r2 = net.get_node_r2(node)
        assert 0.0 <= r2 <= 1.0
    # b should have decent R² because it's driven by a
    assert net.get_node_r2("b") > 0.3


def test_counterfactual_propagates() -> None:
    """do(a=10) should increase c because a→b→c is positive."""
    panel = _synthetic_panel(n=1000)
    net = CausalBayesNet()
    net.learn_structure(panel, significance_level=0.01)
    net.fit_parameters()
    # If PC failed to orient a→b→c, force it so the counterfactual test is
    # deterministic — we are testing propagation, not structure learning.
    if ("a", "b") not in net.get_edges() or ("b", "c") not in net.get_edges():
        import networkx as nx
        net.dag = nx.DiGraph()
        net.dag.add_edges_from([("a", "b"), ("b", "c")])
        net._var_order = list(nx.topological_sort(net.dag))
        net.fit_parameters()
    base = net.counterfactual({}, "c")["expected_value"]
    changed = net.counterfactual({"a": 10.0}, "c")["expected_value"]
    assert changed > base


def test_get_paths_finds_directed_paths() -> None:
    """Paths from a to c should include a→b→c when edges exist."""
    import networkx as nx
    net = CausalBayesNet()
    net.dag = nx.DiGraph()
    net.dag.add_edges_from([("a", "b"), ("b", "c")])
    net._var_order = list(nx.topological_sort(net.dag))
    paths = net.get_paths("a", "c")
    assert any(p == ["a", "b", "c"] for p in paths)


def test_counterfactual_no_structure_raises() -> None:
    """Calling counterfactual before learn_structure must raise."""
    net = CausalBayesNet()
    with pytest.raises(RuntimeError):
        net.counterfactual({"a": 1.0}, "c")
