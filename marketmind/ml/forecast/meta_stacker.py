"""
Meta-stacker: feature dict → calibrated softmax(P_buy, P_sell, P_hold).

Inputs (per symbol/timestamp):
    forecast_return     (point / last_close - 1)        — expected return at horizon
    forecast_vol        (PI80 half-width / last_close)   — uncertainty proxy
    rl_signal_score     ∈ [-1, +1] from existing RL trader
    regime_*            one-hot of 5 regime states
    sentiment_tilt      ∈ [-1, +1] from sector sentiment

Output:
    {p_buy, p_sell, p_hold}     summing to 1
    expected_return, return_95ci

Implementation: ``sklearn.linear_model.LogisticRegression`` with
multinomial softmax. Weights are bootstrapped at module-import time from a
deterministic synthetic generator (fixed seed) so the API works out of the
box. Operators retrain on real labelled history via ``fit_from_history()``,
which overwrites the in-memory weights.

The synthetic generator embeds the rule:
    if forecast_return > +0.02 and rl > 0.3 → BUY
    if forecast_return < -0.02 or  rl <-0.3 → SELL
    otherwise                                 HOLD

This rule is *intentionally simple* — we want the test harness to verify
the meta-stacker can recover an embedded signal, not to declare we've
discovered alpha.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


REGIME_STATES: Tuple[str, ...] = (
    "trending_bull", "range", "volatile", "crash", "recovery",
)
FEATURE_DIM: int = 1 + 1 + 1 + len(REGIME_STATES) + 1  # 9
LABELS: Tuple[str, ...] = ("BUY", "SELL", "HOLD")
LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}


def feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """Map the feature dict into a fixed-shape numpy vector.

    Missing keys → 0.0. Unknown regime → distributed across all 5 regime dims (0.2 each).
    """
    v = np.zeros(FEATURE_DIM, dtype=np.float64)
    v[0] = float(features.get("forecast_return", 0.0))
    v[1] = float(features.get("forecast_vol", 0.0))
    v[2] = float(features.get("rl_signal_score", 0.0))
    regime = (features.get("regime_state") or "").lower().replace(" ", "_")
    if regime in REGIME_STATES:
        v[3 + REGIME_STATES.index(regime)] = 1.0
    elif regime:
        # Unknown regime — stay neutral
        v[3:3 + len(REGIME_STATES)] = 1.0 / len(REGIME_STATES)
    v[3 + len(REGIME_STATES)] = float(features.get("sentiment_tilt", 0.0))
    return v


class MetaStacker:
    """3-class logistic-regression head over the feature vector."""

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        # sklearn ≥1.7 dropped `multi_class`; lbfgs auto-handles multinomial.
        self._model = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0, random_state=0,
        )
        self._fitted = False

    # ── Training ─────────────────────────────────────────────────────────
    def fit_from_history(self, X: np.ndarray, y: Sequence[str]) -> Dict[str, float]:
        """Train on (X, y). y is a sequence of "BUY" / "SELL" / "HOLD" strings.

        Returns {"train_accuracy": float} — operators decide whether to
        accept the new weights based on the score.
        """
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(f"X must have {FEATURE_DIM} features, got {X.shape[1]}")
        y_arr = np.asarray([LABEL_TO_IDX[s] for s in y], dtype=int)
        self._model.fit(X, y_arr)
        self._fitted = True
        acc = float(self._model.score(X, y_arr))
        return {"train_accuracy": round(acc, 4)}

    # ── Inference ────────────────────────────────────────────────────────
    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Return ``{"p_buy": ..., "p_sell": ..., "p_hold": ...}`` summing to 1."""
        if not self._fitted:
            raise RuntimeError("MetaStacker.predict_proba called before fit")
        x = feature_vector(features).reshape(1, -1)
        probs = self._model.predict_proba(x)[0]
        # sklearn's classes_ may not match our LABELS order — remap explicitly.
        out = {f"p_{lbl.lower()}": 0.0 for lbl in LABELS}
        for class_idx, p in zip(self._model.classes_, probs):
            label = LABELS[int(class_idx)]
            out[f"p_{label.lower()}"] = float(p)
        return out


# ── Synthetic bootstrap (run at import) ──────────────────────────────────

def _synthetic_dataset(n: int = 2000, seed: int = 0) -> Tuple[np.ndarray, list]:
    """Deterministic feature/label corpus used to bootstrap default weights.

    Embedded rule (symmetric in BUY/SELL with balanced HOLD):
        score = 5·forecast_return + rl_signal_score + 0.3·sentiment_tilt
        BUY   iff score >  +0.4
        SELL  iff score <  -0.4
        HOLD  otherwise

    The continuous score keeps BUY/SELL probabilities symmetric and ensures
    HOLD is the modal class (~50%) — matching real-world equity-signal
    distributions where HOLD is the safest default.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float64)
    y: list[str] = []
    for i in range(n):
        fret = rng.normal(0.0, 0.03)
        fvol = abs(rng.normal(0.02, 0.01))
        rl = rng.uniform(-1.0, 1.0)
        regime_idx = rng.integers(0, len(REGIME_STATES))
        sent = rng.uniform(-1.0, 1.0)
        X[i, 0] = fret
        X[i, 1] = fvol
        X[i, 2] = rl
        X[i, 3 + regime_idx] = 1.0
        X[i, 3 + len(REGIME_STATES)] = sent
        score = 5.0 * fret + rl + 0.3 * sent
        if score > 0.4:
            y.append("BUY")
        elif score < -0.4:
            y.append("SELL")
        else:
            y.append("HOLD")
    return X, y


_default_stacker: Optional[MetaStacker] = None


def get_meta_stacker() -> MetaStacker:
    """Return the singleton meta-stacker. Bootstrapped on first call from
    the deterministic synthetic dataset."""
    global _default_stacker
    if _default_stacker is None:
        s = MetaStacker()
        X, y = _synthetic_dataset()
        s.fit_from_history(X, y)
        _default_stacker = s
    return _default_stacker


def reset_meta_stacker_for_tests() -> None:
    """Test hook — wipes the singleton so each test gets a fresh fit if needed."""
    global _default_stacker
    _default_stacker = None
