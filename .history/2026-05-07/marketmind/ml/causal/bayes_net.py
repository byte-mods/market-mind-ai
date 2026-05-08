"""Causal Bayesian Network — structure learning + Gaussian parameter fitting.

Learns a DAG from 10 years of monthly macro/sector returns via the PC algorithm
(PC-stable, Pearson correlation test), then fits each node as a linear Gaussian
conditional on its parents.  Counterfactuals propagate deterministically through
the learned coefficients.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports — pgmpy is heavy; only touch it when needed
_PCGMPY: Any = None
_NX: Any = None


def _pgmpy() -> Any:
    global _PCGMPY
    if _PCGMPY is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from pgmpy.estimators import PC
        _PCGMPY = PC
    return _PCGMPY


def _nx() -> Any:
    global _NX
    if _NX is None:
        import networkx as nx
        _NX = nx
    return _NX


class CausalBayesNet:
    """DAG + linear-Gaussian parameters learned from a returns panel."""

    def __init__(self) -> None:
        self.dag: Optional[Any] = None  # networkx.DiGraph
        self._var_order: List[str] = []
        self._params: Dict[str, Dict[str, Any]] = {}
        self._panel: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Structure learning
    # ------------------------------------------------------------------

    def learn_structure(self, panel: pd.DataFrame,
                        significance_level: float = 0.05) -> None:
        """Run PC-stable on the returns panel and store the resulting DAG.

        The PC algorithm is conservative here: we use Pearson correlation tests
        on ~120 monthly observations.  The significance level is relaxed to 0.10
        by default because macro time-series are noisy and we prefer recall over
        precision for explanatory paths (false edges are harmless; missing edges
        hide causal channels).

        Post-processing breaks any cycles created by conflicting orientation
        rules (common on small samples) by removing the weakest edge measured
        by absolute Pearson correlation.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            PC = _pgmpy()
            pc = PC(data=panel)
            model = pc.estimate(
                variant="stable",
                ci_test="pearsonr",
                significance_level=significance_level,
                show_progress=False,
            )
        nx = _nx()
        dag = nx.DiGraph()
        dag.add_nodes_from(model.nodes())
        dag.add_edges_from(model.edges())

        # Break cycles by removing weakest edge (by |correlation|)
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle_edges = nx.find_cycle(dag, orientation="original")
            except nx.NetworkXNoCycle:
                break
            # Find weakest edge in the cycle
            weakest = None
            weakest_corr = float("inf")
            for u, v, _ in cycle_edges:
                corr = abs(panel[u].corr(panel[v]))
                if corr < weakest_corr:
                    weakest_corr = corr
                    weakest = (u, v)
            if weakest:
                dag.remove_edge(*weakest)
                logger.debug("CausalBayesNet: broke cycle by removing %s → %s", *weakest)
            else:
                break

        self.dag = dag
        self._var_order = list(nx.topological_sort(self.dag))
        self._panel = panel.copy()
        logger.info(
            "CausalBayesNet: learned DAG with %d nodes, %d edges",
            self.dag.number_of_nodes(),
            self.dag.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Parameter fitting (linear Gaussian)
    # ------------------------------------------------------------------

    def fit_parameters(self) -> None:
        """Fit OLS coefficients for each node conditional on its parents."""
        if self.dag is None:
            raise RuntimeError("Call learn_structure() first")
        if self._panel is None:
            raise RuntimeError("No panel available")

        panel = self._panel
        self._params = {}
        for node in self._var_order:
            parents = list(self.dag.predecessors(node))
            if not parents:
                # Root node — model as constant mean + variance
                self._params[node] = {
                    "intercept": float(panel[node].mean()),
                    "coefs": {},
                    "sigma": float(panel[node].std()),
                    "r2": 0.0,
                }
                continue

            X = panel[parents].values
            y = panel[node].values
            # OLS: β = (X'X)^{-1} X'y  (with intercept)
            X_design = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            intercept = float(beta[0])
            coefs = {p: float(b) for p, b in zip(parents, beta[1:])}
            y_pred = X_design @ beta
            residuals = y - y_pred
            sigma = float(np.std(residuals))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

            self._params[node] = {
                "intercept": intercept,
                "coefs": coefs,
                "sigma": sigma,
                "r2": max(0.0, min(1.0, r2)),
            }
            logger.debug(
                "CausalBayesNet: %s ← %s  R²=%.3f σ=%.4f",
                node, parents, r2, sigma,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_edges(self) -> List[Tuple[str, str]]:
        """Return list of (parent, child) edges."""
        if self.dag is None:
            return []
        return list(self.dag.edges())

    def get_parents(self, node: str) -> List[str]:
        """Return parents of a node."""
        if self.dag is None:
            return []
        return list(self.dag.predecessors(node))

    def get_paths(self, source: str, target: str) -> List[List[str]]:
        """All simple directed paths from source to target."""
        if self.dag is None:
            return []
        nx = _nx()
        try:
            return list(nx.all_simple_paths(self.dag, source, target))
        except nx.NetworkXNoPath:
            return []

    def get_node_r2(self, node: str) -> float:
        """R² of the local regression for this node."""
        return self._params.get(node, {}).get("r2", 0.0)

    # ------------------------------------------------------------------
    # Counterfactual inference
    # ------------------------------------------------------------------

    def counterfactual(self, intervention: Dict[str, float],
                       target: str) -> Dict[str, Any]:
        """Propagate a do-intervention through the linear-Gaussian network.

        Returns dict with:
            - expected_value: float — predicted target under intervention
            - propagation: dict[node] = value — full node state after do()
            - path_r2s: list[float] — R² along each directed path from any
              intervention node to the target
        """
        if self.dag is None or not self._params:
            raise RuntimeError("Structure not learned or parameters not fitted")

        # Start from current sample means (panel means represent E[·])
        state = {n: float(self._panel[n].mean()) for n in self._var_order}

        # Apply do() — override intervention nodes
        for node, val in intervention.items():
            state[node] = float(val)

        # Propagate topologically
        for node in self._var_order:
            if node in intervention:
                continue  # do() fixed this node
            p = self._params[node]
            pred = p["intercept"]
            for parent, coef in p["coefs"].items():
                pred += coef * state[parent]
            state[node] = pred

        # Collect R² along every path from any intervention node to target
        path_r2s: List[float] = []
        for iv_node in intervention:
            paths = self.get_paths(iv_node, target)
            for path in paths:
                r2s = [self.get_node_r2(n) for n in path[1:]]  # skip source
                # Aggregate as geometric mean (all steps must be decent)
                if r2s:
                    path_r2s.append(float(np.prod(r2s) ** (1.0 / len(r2s))))
                else:
                    path_r2s.append(1.0)  # direct edge

        avg_path_r2 = float(np.mean(path_r2s)) if path_r2s else 0.0
        return {
            "expected_value": state.get(target, 0.0),
            "propagation": state,
            "path_r2s": path_r2s,
            "avg_path_r2": avg_path_r2,
        }


_bayes_net: Optional[CausalBayesNet] = None


def get_causal_bayes_net() -> CausalBayesNet:
    global _bayes_net
    if _bayes_net is None:
        _bayes_net = CausalBayesNet()
    return _bayes_net
