"""Tests for the PPO-override branch in /api/rl/predict.

Regression: when a saved PPO model fires with confidence >= 0.55 it
overrides the action coming out of get_combined_signal but used to leave
``target``/``sl`` untouched. So a BUY override on a SELL-shaped signal
ended up reporting ``BUY @1435, target 1320, SL 1471`` — target below
entry, SL above. We verify the override now recomputes the levels.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def _ppo_override_action(ql_sig: dict, ppo_sig: dict) -> dict:
    """Inline copy of the override logic from server.predict_rl_signal so
    the test exercises the algorithm directly without spinning up FastAPI.

    Keep this in sync with `server.py` when the override is changed."""
    if ppo_sig.get('source') == 'ppo_model':
        ql_sig['ppo_action']     = ppo_sig['action']
        ql_sig['ppo_confidence'] = ppo_sig['confidence']
        if ppo_sig['confidence'] >= 0.55:
            prev_action          = ql_sig.get('action')
            ql_sig['action']     = ppo_sig['action']
            ql_sig['confidence'] = ppo_sig['confidence']
            ql_sig['method']     = 'PPO'
            if prev_action != ql_sig['action']:
                rl_data  = ql_sig.get('rl_signal') or {}
                rp       = (rl_data.get('risk_params') or {}) if isinstance(rl_data, dict) else {}
                sl_pct   = float(rp.get('stop_loss_pct', 2.5))
                tp_pct   = float(rp.get('take_profit_pct', 8.0))
                entry    = float(ql_sig.get('entry_price') or 0)
                if entry > 0:
                    if ql_sig['action'] == 'BUY':
                        ql_sig['target'] = round(entry * (1 + tp_pct / 100), 2)
                        ql_sig['sl']     = round(entry * (1 - sl_pct / 100), 2)
                    elif ql_sig['action'] == 'SELL':
                        ql_sig['target'] = round(entry * (1 - tp_pct / 100), 2)
                        ql_sig['sl']     = round(entry * (1 + sl_pct / 100), 2)
                    else:
                        ql_sig['target'] = entry
                        ql_sig['sl']     = entry
        else:
            ql_sig['method']     = 'RL+ML'
    return ql_sig


def _sell_shaped_signal():
    """get_combined_signal output that decided SELL — target below entry,
    SL above entry. This is the shape that used to leak through to BUY."""
    return {
        'symbol': 'RELIANCE', 'action': 'SELL', 'confidence': 0.6,
        'entry_price': 1435.2,
        'target': 1320.38,  # entry * 0.92
        'sl': 1471.08,      # entry * 1.025
        'rl_signal': {'risk_params': {'stop_loss_pct': 2.5, 'take_profit_pct': 8.0}},
    }


def test_ppo_override_to_buy_recomputes_target_above_entry():
    ql = _sell_shaped_signal()
    ppo = {'source': 'ppo_model', 'action': 'BUY', 'confidence': 0.92}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'BUY'
    assert out['method'] == 'PPO'
    # BUY: target above entry, SL below entry — the bug was the inverse.
    assert out['target'] > out['entry_price']
    assert out['sl']     < out['entry_price']
    assert out['target'] == round(1435.2 * 1.08, 2)
    assert out['sl']     == round(1435.2 * 0.975, 2)


def test_ppo_override_to_sell_keeps_correct_levels():
    """Already-SELL signal that PPO confirms should not be re-touched."""
    ql = _sell_shaped_signal()
    ppo = {'source': 'ppo_model', 'action': 'SELL', 'confidence': 0.8}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'SELL'
    # No flip → no recompute → original SELL-shaped levels stay.
    assert out['target'] == 1320.38
    assert out['sl']     == 1471.08


def test_ppo_low_confidence_does_not_override():
    """Below 0.55 PPO should leave ql_sig['action'] alone but still expose
    PPO's action/confidence as auxiliary fields for the UI."""
    ql = _sell_shaped_signal()
    ppo = {'source': 'ppo_model', 'action': 'BUY', 'confidence': 0.4}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'SELL'  # unchanged
    assert out['method'] == 'RL+ML'
    assert out['ppo_action'] == 'BUY'
    assert out['ppo_confidence'] == 0.4


def test_ppo_override_to_hold_collapses_levels_to_entry():
    ql = _sell_shaped_signal()
    ppo = {'source': 'ppo_model', 'action': 'HOLD', 'confidence': 0.9}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'HOLD'
    assert out['target'] == out['entry_price']
    assert out['sl']     == out['entry_price']


def test_ppo_override_uses_default_pcts_when_risk_params_missing():
    """Robustness: if rl_signal is missing or risk_params absent, fall back
    to the same 2.5% / 8.0% defaults get_combined_signal uses."""
    ql = {
        'symbol': 'X', 'action': 'SELL', 'confidence': 0.5,
        'entry_price': 1000.0, 'target': 920.0, 'sl': 1025.0,
        'rl_signal': None,  # gap intentionally
    }
    ppo = {'source': 'ppo_model', 'action': 'BUY', 'confidence': 0.8}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'BUY'
    assert out['target'] == 1080.0   # default 8% above
    assert out['sl']     == 975.0    # default 2.5% below


def test_ppo_override_no_ppo_model_leaves_signal_untouched():
    """If PPO source isn't 'ppo_model' (e.g. fallback or no saved model),
    nothing should change."""
    ql = _sell_shaped_signal()
    ppo = {'source': 'fallback', 'action': 'BUY', 'confidence': 1.0}
    out = _ppo_override_action(ql, ppo)
    assert out['action'] == 'SELL'
    assert 'ppo_action' not in out
