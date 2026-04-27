"""
MarketMind AI - PPO (Proximal Policy Optimization) Trainer
============================================================
Deep RL agent for stock trading using PyTorch.

Architecture:
  - Shared body: 3-layer MLP with LayerNorm + ReLU
  - Actor head:  policy logits → softmax → action probabilities
  - Critic head: scalar state-value

Training:
  - Collect N_STEPS transitions per rollout
  - Compute GAE advantages
  - Run K epochs of mini-batch PPO updates with clipped surrogate loss
  - Repeat for num_updates iterations

Pattern diagnostics:
  - After training, analyse which market conditions lead to losses
  - Return top-5 "mistake patterns" in human-readable text
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'marketmind', 'models'
)
os.makedirs(_MODELS_DIR, exist_ok=True)


# ── PyTorch imports (optional graceful fallback) ──────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    _TORCH_AVAILABLE = True
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    _TORCH_AVAILABLE = False
    _DEVICE = 'cpu'


# ── PPO Hyperparameters ───────────────────────────────────────────────────────

PPO_CONFIG = dict(
    hidden_dim      = 256,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_epsilon    = 0.2,
    entropy_coef    = 0.05,     # higher entropy → more exploration, fewer HOLD collapses
    value_coef      = 0.5,
    max_grad_norm   = 0.5,
    n_steps         = 256,      # steps per rollout
    ppo_epochs      = 8,        # PPO update epochs per rollout
    mini_batch_size = 64,
    target_kl       = 0.05,     # more tolerant early stop
)


# ── Actor-Critic Network ──────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class ActorCriticNet(nn.Module):
        def __init__(self, obs_size: int, n_actions: int = 3, hidden: int = 256):
            super().__init__()
            # Shared body with LayerNorm for stable training
            self.body = nn.Sequential(
                nn.Linear(obs_size, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
            )
            # Actor head: produces policy logits
            self.actor  = nn.Linear(hidden // 2, n_actions)
            # Critic head: produces V(s)
            self.critic = nn.Linear(hidden // 2, 1)

            # Orthogonal init for better convergence
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.zeros_(m.bias)
            # Smaller init for actor output
            nn.init.orthogonal_(self.actor.weight, gain=0.01)

        def forward(self, x: 'torch.Tensor'):
            feat   = self.body(x)
            logits = self.actor(feat)
            value  = self.critic(feat).squeeze(-1)
            return logits, value

        def act(self, obs: np.ndarray, deterministic: bool = False):
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
                logits, value = self(t)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob   = dist.log_prob(action)
                confidence = torch.softmax(logits, dim=-1).max().item()
            return int(action.item()), float(log_prob.item()), float(value.item()), confidence

        def evaluate(self, obs_t, actions_t):
            logits, values = self(obs_t)
            dist    = Categorical(logits=logits)
            lp      = dist.log_prob(actions_t)
            entropy = dist.entropy()
            return lp, values, entropy


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def clear(self):
        self.__init__()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        n          = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_adv   = 0.0
        vals       = self.values + [last_value]
        for t in reversed(range(n)):
            nxt_val  = vals[t + 1] * (1.0 - float(self.dones[t]))
            delta    = self.rewards[t] + gamma * nxt_val - vals[t]
            advantages[t] = delta + gamma * lam * (1.0 - float(self.dones[t])) * last_adv
            last_adv = advantages[t]
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


# ── PPO Update Step ───────────────────────────────────────────────────────────

def _ppo_update(
    net: 'ActorCriticNet',
    optimizer: 'torch.optim.Adam',
    buffer: RolloutBuffer,
    last_value: float,
    cfg: Dict,
) -> Dict:
    advantages, returns = buffer.compute_gae(last_value, cfg['gamma'], cfg['gae_lambda'])

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t    = torch.FloatTensor(np.array(buffer.obs)).to(_DEVICE)
    act_t    = torch.LongTensor(buffer.actions).to(_DEVICE)
    old_lp_t = torch.FloatTensor(buffer.log_probs).to(_DEVICE)
    adv_t    = torch.FloatTensor(advantages).to(_DEVICE)
    ret_t    = torch.FloatTensor(returns).to(_DEVICE)

    n = len(buffer.obs)
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'kl': 0.0}

    for epoch_i in range(cfg['ppo_epochs']):
        indices  = np.random.permutation(n)
        approx_kl = 0.0
        n_batches = 0

        for start in range(0, n, cfg['mini_batch_size']):
            idx = indices[start: start + cfg['mini_batch_size']]
            if len(idx) < 4:
                continue

            lp_new, val_new, ent = net.evaluate(obs_t[idx], act_t[idx])

            # PPO clipped ratio
            ratio = torch.exp(lp_new - old_lp_t[idx])
            clip  = torch.clamp(ratio, 1 - cfg['clip_epsilon'], 1 + cfg['clip_epsilon'])

            policy_loss = -torch.min(ratio * adv_t[idx], clip * adv_t[idx]).mean()
            value_loss  = F.mse_loss(val_new, ret_t[idx])
            entropy_loss= -ent.mean()

            loss = (policy_loss
                    + cfg['value_coef']   * value_loss
                    + cfg['entropy_coef'] * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg['max_grad_norm'])
            optimizer.step()

            approx_kl += float((old_lp_t[idx] - lp_new).detach().mean().abs())
            n_batches += 1

            stats['policy_loss'] += float(policy_loss.detach())
            stats['value_loss']  += float(value_loss.detach())
            stats['entropy']     += float((-entropy_loss).detach())

        # Early stop if KL too large
        avg_kl = approx_kl / max(n_batches, 1)
        if avg_kl > cfg['target_kl']:
            break
        stats['kl'] = avg_kl

    nb = max(cfg['ppo_epochs'] * max(1, n // cfg['mini_batch_size']), 1)
    return {k: round(v / nb, 5) for k, v in stats.items()}


# ── Training Orchestrator ─────────────────────────────────────────────────────

def train_ppo_agent(
    df: pd.DataFrame,
    symbol: str,
    epochs: int = 200,
    initial_capital: float = 100_000.0,
    stop_loss_pct: float = 2.5,
    take_profit_pct: float = 8.0,
    trailing_sl_pct: float = 2.0,
) -> Dict:
    """
    Train a PPO agent on historical stock data.
    Returns training metrics, backtest simulation, pattern mistakes, and saves the model.
    """
    from .rl_trainer import compute_features
    from .stock_env import StockTradingEnv, analyze_mistakes, OBS_COLS

    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — falling back to Q-Learning")
        from .rl_trainer import train_rl_agent
        return train_rl_agent(df, symbol, epochs=epochs,
                              initial_capital=initial_capital,
                              stop_loss_pct=stop_loss_pct,
                              take_profit_pct=take_profit_pct,
                              trailing_sl_pct=trailing_sl_pct)

    feat = compute_features(df)
    feat = feat.iloc[50:].reset_index(drop=True)
    if len(feat) < 80:
        return {'error': 'Need at least 130 bars. Increase days parameter.'}

    split      = int(len(feat) * 0.75)
    train_feat = feat.iloc[:split].reset_index(drop=True)
    test_feat  = feat.iloc[split:].reset_index(drop=True)

    env = StockTradingEnv(
        train_feat, initial_capital=initial_capital,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        trailing_sl_pct=trailing_sl_pct,
    )

    net       = ActorCriticNet(env.obs_size, n_actions=3,
                               hidden=PPO_CONFIG['hidden_dim']).to(_DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=PPO_CONFIG['lr'],
                                  eps=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    cfg            = PPO_CONFIG.copy()
    cfg['n_steps'] = min(cfg['n_steps'], len(train_feat) - env.warmup - 1)

    epoch_returns:  List[float] = []
    best_test_return = -float('inf')
    best_state_dict  = None

    # ── Early stopping config ────────────────────────────────────────────────
    # Patience: how many validation checks (every val_interval epochs) without
    # improvement before we stop early.
    val_interval      = max(10, min(100, epochs // 100))  # check every 1% of total epochs
    patience          = max(20, epochs // 20)             # 5% of epochs without improvement
    no_improve_checks = 0
    stopped_at        = epochs

    logger.info(f"PPO training {symbol}: {epochs} epochs, val_interval={val_interval}, patience={patience}")

    for ep in range(epochs):
        # --- Rollout phase ---
        obs     = env.reset(start=np.random.randint(env.warmup, max(env.warmup + 1, len(train_feat) // 3)))
        buffer  = RolloutBuffer()
        done    = False
        steps   = 0

        while not done and steps < cfg['n_steps']:
            action, log_prob, value, _ = net.act(obs)
            next_obs, reward, done, _  = env.step(action)
            buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs
            steps += 1

        # Estimate last value for GAE
        last_value = 0.0
        if not done:
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0).to(_DEVICE)
                _, lv = net(t)
                last_value = float(lv.item())

        if len(buffer.obs) > 1:
            _ppo_update(net, optimizer, buffer, last_value, cfg)

        ep_ret = env.metrics()['total_return_pct']
        epoch_returns.append(ep_ret)

        # Validate on test set at val_interval cadence
        if (ep + 1) % val_interval == 0:
            tr = _test_return(net, test_feat, initial_capital,
                              stop_loss_pct, take_profit_pct, trailing_sl_pct)
            if tr > best_test_return + 0.1:   # must improve by at least 0.1%
                best_test_return = tr
                best_state_dict  = {k: v.cpu().clone()
                                    for k, v in net.state_dict().items()}
                no_improve_checks = 0
            else:
                no_improve_checks += 1
                if no_improve_checks >= patience:
                    stopped_at = ep + 1
                    logger.info(f"PPO early stop at epoch {stopped_at}/{epochs} "
                                f"(no improvement for {patience} checks of {val_interval})")
                    break

        scheduler.step()

    logger.info(f"PPO {symbol}: completed {stopped_at} epochs (early_stop={'yes' if stopped_at<epochs else 'no'})")

    # Load best checkpoint
    if best_state_dict:
        net.load_state_dict(best_state_dict)
    net.eval()

    # Full test-set simulation (deterministic)
    sim       = _simulate_deterministic(net, test_feat, initial_capital,
                                        stop_loss_pct, take_profit_pct, trailing_sl_pct)
    # Full dataset simulation for equity curve
    full_sim  = _simulate_deterministic(net, feat, initial_capital,
                                        stop_loss_pct, take_profit_pct, trailing_sl_pct)

    # Pattern mistake analysis
    all_trades = full_sim.get('_trades_raw', [])
    obs_cols   = [c for c in OBS_COLS if c in feat.columns]
    mistakes   = analyze_mistakes(all_trades, obs_cols)

    # Save model
    model_path = os.path.join(_MODELS_DIR, f'{symbol}_ppo.pt')
    torch.save({
        'net_state':       net.state_dict(),
        'obs_size':        env.obs_size,
        'obs_cols':        obs_cols,
        'symbol':          symbol,
        'trained_at':      pd.Timestamp.now().isoformat(),
        'epochs':          epochs,
        'stop_loss_pct':   stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'trailing_sl_pct': trailing_sl_pct,
        'train_bars':      len(train_feat),
        'test_bars':       len(test_feat),
        # OOS performance stats (persisted for model listing)
        'val_return_pct':   sim.get('total_return_pct', 0),
        'bah_return_pct':   sim.get('bah_return_pct', 0),
        'alpha':            sim.get('alpha', 0),
        'win_rate':         sim.get('win_rate', 0),
        'total_trades':     sim.get('total_trades', 0),
        'profit_factor':    sim.get('profit_factor', 0),
        'losing_trades':    sim.get('losing_trades', 0),
        'max_drawdown_pct': sim.get('max_drawdown_pct', 0),
        'sharpe_ratio':     sim.get('sharpe_ratio', 0),
    }, model_path)

    logger.info(
        f"PPO trained {symbol}: return={sim['total_return_pct']:.1f}% "
        f"BAH={sim['bah_return_pct']:.1f}% WR={sim['win_rate']}% "
        f"Trades={sim['total_trades']}"
    )

    # Remove internal _trades_raw before returning
    sim.pop('_trades_raw', None)
    full_sim.pop('_trades_raw', None)

    return {
        'symbol':             symbol,
        'method':             'PPO',
        'epochs':             epochs,
        'epochs_run':         stopped_at,
        'early_stopped':      stopped_at < epochs,
        'train_bars':         len(train_feat),
        'test_bars':          len(test_feat),
        'epoch_returns':      [round(r, 2) for r in epoch_returns[-30:]],
        'simulation':         sim,
        'full_equity_curve':  full_sim['equity_curve'],
        'mistakes':           mistakes,
        'model_path':         model_path,
        'risk_params': {
            'stop_loss_pct':   stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_sl_pct': trailing_sl_pct,
        },
    }


def _test_return(net, feat, capital, sl, tp, tsl) -> float:
    """Quick test-set return for model selection."""
    sim = _simulate_deterministic(net, feat, capital, sl, tp, tsl)
    return sim.get('total_return_pct', -999.0)


def _simulate_deterministic(
    net: 'ActorCriticNet',
    feat: pd.DataFrame,
    initial_capital: float = 100_000.0,
    stop_loss_pct: float = 2.5,
    take_profit_pct: float = 8.0,
    trailing_sl_pct: float = 2.0,
) -> Dict:
    """
    Run trained PPO policy deterministically on a feature DataFrame.
    Returns full metrics + equity curve + trade log.
    Only trades when agent confidence ≥ 0.5 (selective/high-conviction).
    """
    from .stock_env import StockTradingEnv, ACTION_HOLD

    # Use at most 25% of available bars as warmup so we always have ≥75% to trade
    warmup = min(30, max(10, len(feat) // 4))
    env  = StockTradingEnv(feat, initial_capital=initial_capital,
                           stop_loss_pct=stop_loss_pct,
                           take_profit_pct=take_profit_pct,
                           trailing_sl_pct=trailing_sl_pct,
                           warmup=warmup)
    obs  = env.reset()
    done = False

    while not done:
        action, _, _, conf = net.act(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    m = env.metrics()

    # Equity curve (downsample to 200 pts)
    eq  = env.equity
    step = max(1, len(eq) // 200)
    curve = [{'bar': i, 'value': v} for i, v in enumerate(eq[::step])]

    # Trade log (last 100 for display)
    trades_display = [{k: v for k, v in t.items()
                       if k not in ('entry_obs', 'entry_raw')}
                      for t in env.trades[-100:]]

    m['equity_curve']  = curve
    m['trades']        = trades_display
    m['_trades_raw']   = env.trades   # kept for mistake analysis
    return m


def predict_ppo(symbol: str, feat_row: pd.Series, obs_cols: List[str]) -> Dict:
    """
    Load saved PPO model and predict action + confidence for a single observation row.
    """
    model_path = os.path.join(_MODELS_DIR, f'{symbol}_ppo.pt')
    if not os.path.exists(model_path) or not _TORCH_AVAILABLE:
        return {'action': 'HOLD', 'confidence': 0.5, 'source': 'no_model'}

    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        net  = ActorCriticNet(ckpt['obs_size']).to(_DEVICE)
        net.load_state_dict(ckpt['net_state'])
        net.eval()

        cols_in_model = ckpt.get('obs_cols', obs_cols)
        raw = np.array([feat_row.get(c, 0.0) for c in cols_in_model], dtype=np.float32)

        # Normalize (rough percentile norm unavailable without full data, use tanh scaling)
        raw = np.tanh(raw)

        # Append position features: not holding, 0 unrealized, 0 age
        obs = np.concatenate([raw, [0.0, 0.0, 0.0]]).astype(np.float32)
        obs = obs[:ckpt['obs_size']]  # truncate/pad to model's obs_size
        if len(obs) < ckpt['obs_size']:
            obs = np.pad(obs, (0, ckpt['obs_size'] - len(obs)))

        action, lp, value, conf = net.act(obs, deterministic=True)
        from .stock_env import ACTION_NAMES
        return {
            'action':     ACTION_NAMES[action],
            'confidence': round(conf, 3),
            'value':      round(value, 4),
            'source':     'ppo_model',
        }
    except Exception as e:
        logger.error(f"PPO predict error for {symbol}: {e}")
        return {'action': 'HOLD', 'confidence': 0.5, 'source': 'error'}


def list_ppo_models() -> List[Dict]:
    """List all saved PPO models with metadata."""
    models = []
    if not _TORCH_AVAILABLE:
        return models
    for fname in os.listdir(_MODELS_DIR):
        if fname.endswith('_ppo.pt'):
            path = os.path.join(_MODELS_DIR, fname)
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                models.append({
                    'symbol':         ckpt.get('symbol', fname.replace('_ppo.pt', '')),
                    'trained_at':     ckpt.get('trained_at', ''),
                    'epochs':         ckpt.get('epochs', 0),
                    'method':         'PPO (PyTorch)',
                    'path':           path,
                    'train_bars':     ckpt.get('train_bars', 0),
                    'test_bars':      ckpt.get('test_bars', 0),
                    'val_return_pct': ckpt.get('val_return_pct', None),
                    'bah_return_pct': ckpt.get('bah_return_pct', None),
                    'alpha':          ckpt.get('alpha', None),
                    'win_rate':       ckpt.get('win_rate', None),
                    'total_trades':   ckpt.get('total_trades', None),
                    'losing_trades':  ckpt.get('losing_trades', None),
                    'profit_factor':  ckpt.get('profit_factor', None),
                    'max_drawdown_pct': ckpt.get('max_drawdown_pct', None),
                    'sharpe_ratio':   ckpt.get('sharpe_ratio', None),
                    'stop_loss_pct':  ckpt.get('stop_loss_pct', None),
                    'take_profit_pct': ckpt.get('take_profit_pct', None),
                })
            except Exception:
                pass
    return models


# Re-export for env access
from .stock_env import ACTION_HOLD, ACTION_BUY, ACTION_SELL, ACTION_NAMES
