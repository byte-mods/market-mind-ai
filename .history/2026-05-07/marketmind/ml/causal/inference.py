"""Causal inference engine — counterfactual API over the learned Bayes net.

Provides human-readable path explanations and calibrated confidence bands
for ``what-if`` queries on macro/sector nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from marketmind.ml.causal.data_layer import (
    CausalDataCollector,
    get_causal_data_collector,
    INTERVENTION_NODES,
    TARGET_NODES,
)
from marketmind.ml.causal.bayes_net import CausalBayesNet, get_causal_bayes_net

logger = logging.getLogger(__name__)


class CausalInferenceEngine:
    """High-level wrapper: data → learned net → counterfactual answers."""

    def __init__(
        self,
        collector: Optional[CausalDataCollector] = None,
        net: Optional[CausalBayesNet] = None,
    ) -> None:
        self.collector = collector or get_causal_data_collector()
        self.net = net or get_causal_bayes_net()
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_trained(self, significance_level: float = 0.10) -> None:
        """Idempotent: learn structure + fit parameters if not already done."""
        if self._ready and self.net.dag is not None:
            return
        panel = self.collector.get_returns_panel()
        self.net.learn_structure(panel, significance_level=significance_level)
        self.net.fit_parameters()
        self._ready = True
        logger.info("CausalInferenceEngine: trained on %d rows × %d cols",
                    len(panel), len(panel.columns))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def whatif(self, intervention: Dict[str, float],
               target: str) -> Dict[str, Any]:
        """Answer a counterfactual query.

        Args:
            intervention: mapping ``{node_id: new_value}`` — the do() operation.
            target: node_id to predict under the intervention.

        Returns:
            dict with ``target_estimate``, ``target_current``, ``delta``,
            ``confidence`` (0–1), ``paths`` (human-readable), and ``dag_edges``.
        """
        self.ensure_trained()

        # Validate
        invalid = [n for n in intervention if n not in INTERVENTION_NODES]
        if invalid:
            raise ValueError(
                f"Invalid intervention node(s): {invalid}. "
                f"Allowed: {sorted(INTERVENTION_NODES)}"
            )
        if target not in TARGET_NODES:
            raise ValueError(
                f"Invalid target: {target}. Allowed: {sorted(TARGET_NODES)}"
            )

        current = self.collector.get_current_values()

        # Run counterfactual on the returns net — but we want absolute-level
        # answers, not return answers.  Strategy:
        #   1. Convert absolute intervention to a return-space deviation
        #      (intervention_return = log(new / current))
        #   2. Propagate through the returns network
        #   3. Convert target result back to absolute: new = current * exp(delta)
        #
        # This preserves the correlation structure learned on returns while
        # producing interpretable index-level numbers.

        iv_returns: Dict[str, float] = {}
        for node, new_val in intervention.items():
            cur = current.get(node, new_val)
            if cur and cur != 0:
                iv_returns[node] = float(np.log(new_val / cur))
            else:
                iv_returns[node] = 0.0

        result = self.net.counterfactual(iv_returns, target)
        target_delta_return = result["expected_value"]
        target_current = current.get(target, 0.0)
        if target_current and target_current != 0:
            target_estimate = float(target_current * np.exp(target_delta_return))
        else:
            target_estimate = target_current

        delta = target_estimate - target_current

        # Confidence = average path R², clamped to [0.3, 0.95]
        # Low R² means the causal path is weak / noisy
        raw_conf = result.get("avg_path_r2", 0.0)
        confidence = float(np.clip(raw_conf if raw_conf > 0 else 0.5, 0.3, 0.95))

        # Path explanations
        paths = self._explain_paths(intervention, target, current)

        return {
            "target": target,
            "target_estimate": round(target_estimate, 2),
            "target_current": round(target_current, 2),
            "delta": round(delta, 2),
            "delta_pct": round((delta / target_current * 100), 2) if target_current else 0.0,
            "confidence": round(confidence, 2),
            "intervention": intervention,
            "paths": paths,
            "dag_edges": self.net.get_edges(),
        }

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Return all node metadata + current values + parents in DAG."""
        self.ensure_trained()
        nodes = self.collector.get_node_info()
        for n in nodes:
            n["parents"] = self.net.get_parents(n["id"])
        return nodes

    def get_network_summary(self) -> Dict[str, Any]:
        """High-level stats about the learned network."""
        self.ensure_trained()
        edges = self.net.get_edges()
        return {
            "node_count": self.net.dag.number_of_nodes(),
            "edge_count": self.net.dag.number_of_edges(),
            "edges": edges,
            "intervention_nodes": sorted(INTERVENTION_NODES),
            "target_nodes": sorted(TARGET_NODES),
        }

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    def _explain_paths(self, intervention: Dict[str, float],
                       target: str, current: Dict[str, float]) -> List[Dict[str, Any]]:
        """Build human-readable path descriptions."""
        from marketmind.ml.causal.data_layer import NODE_META
        paths_out: List[Dict[str, Any]] = []
        for iv_node in intervention:
            raw_paths = self.net.get_paths(iv_node, target)
            for p in raw_paths:
                labels = [NODE_META.get(n, {}).get("label", n) for n in p]
                r2s = [self.net.get_node_r2(n) for n in p[1:]]
                strength = float(np.mean(r2s)) if r2s else 1.0
                # Classify path strength
                if strength >= 0.6:
                    strength_label = "strong"
                elif strength >= 0.3:
                    strength_label = "moderate"
                else:
                    strength_label = "weak"
                paths_out.append({
                    "from": iv_node,
                    "to": target,
                    "nodes": p,
                    "labels": labels,
                    "strength": round(strength, 2),
                    "strength_label": strength_label,
                })
        return paths_out


_engine: Optional[CausalInferenceEngine] = None


def get_causal_inference_engine() -> CausalInferenceEngine:
    global _engine
    if _engine is None:
        _engine = CausalInferenceEngine()
    return _engine
