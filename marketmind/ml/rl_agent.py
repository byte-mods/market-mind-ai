"""
MarketMind AI - RL Agent Module
DQN, PPO, and A3C agents for trading
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import json


class DQNTradingAgent:
    """
    DQN Agent for discrete trading actions
    Actions: BUY (0), HOLD (1), SELL (2), HEDGE (3), REBALANCE (4)
    """

    def __init__(self, state_size: int = 50, action_size: int = 5,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        """
        Initialize DQN Agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.update_target_freq = 1000

        # Simple Q-table approach for desktop app (no TensorFlow dependency)
        # In production, would use actual neural network
        self.q_table = {}
        self.steps = 0

        # Action names
        self.action_names = ['BUY', 'HOLD', 'SELL', 'HEDGE', 'REBALANCE']

    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key for Q-table"""
        # Quantize state to reduce space
        quantized = (state * 10).astype(int)
        return tuple(quantized.tolist())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        """
        state_key = self._get_state_key(state)

        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: best known action
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            return int(np.argmax(q_values))
        else:
            return 1  # Default to HOLD

    def remember(self, state: np.ndarray, action: int, reward: float,
                  next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32) -> float:
        """
        Learn from experience replay
        """
        if len(self.memory) < batch_size:
            return 0.0

        batch = random.sample(self.memory, batch_size)

        total_error = 0

        for state, action, reward, next_state, done in batch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)

            # Initialize Q-table entries if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)

            # Current Q value
            current_q = self.q_table[state_key][action]

            # Target Q value
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_state_key])

            # Update Q value (simple gradient descent approximation)
            error = target_q - current_q
            self.q_table[state_key][action] += self.learning_rate * error
            total_error += abs(error)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1

        return total_error / batch_size

    def train(self, environment, episodes: int = 100) -> Dict:
        """
        Train DQN agent in simulated environment
        """
        training_history = {
            'episode': [],
            'total_reward': [],
            'final_value': [],
            'epsilon': []
        }

        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0

            for step in range(252):  # 252 trading days
                # Choose and perform action
                action = self.act(state)
                next_state, reward, done = environment.step(action)

                # Store experience
                self.remember(state, action, reward, next_state, done)

                # Train
                loss = self.replay(batch_size=32)

                total_reward += reward
                state = next_state

                if done:
                    break

            # Track progress
            metrics = environment.get_performance_metrics()

            if (episode + 1) % 10 == 0:
                training_history['episode'].append(episode + 1)
                training_history['total_reward'].append(total_reward)
                training_history['final_value'].append(metrics.get('final_value', 0))
                training_history['epsilon'].append(self.epsilon)

                print(f"Episode {episode + 1}/{episodes} | "
                      f"Return: {metrics.get('total_return', 0)*100:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")

        return training_history

    def get_action_name(self, action: int) -> str:
        """Get name of action"""
        return self.action_names[action] if 0 <= action < len(self.action_names) else 'UNKNOWN'


class PPOTradingAgent:
    """
    PPO Agent for continuous position sizing and portfolio allocation
    Actions: Position size (0.0 to 1.0), Hedge ratio, Sector allocation
    """

    def __init__(self, state_size: int = 50, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.learning_rate = 0.0003

        # Simple policy table (would be neural network in production)
        self.policy = {}
        self.value_table = {}

        self.action_names = ['POSITION_SIZE', 'HEDGE_RATIO', 'SECTOR_ALLOC']

    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key"""
        quantized = (state * 10).astype(int)
        return tuple(quantized.tolist())

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get continuous actions
        Returns: array of action values
        """
        state_key = self._get_state_key(state)

        if state_key in self.policy:
            return np.array(self.policy[state_key])
        else:
            # Default to neutral position
            return np.array([0.1, 0.0, 0.5])

    def evaluate(self, state: np.ndarray) -> float:
        """Estimate state value"""
        state_key = self._get_state_key(state)

        if state_key in self.value_table:
            return self.value_table[state_key]
        return 0.0

    def update(self, states: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, dones: np.ndarray):
        """Update policy and value function"""
        # Simplified PPO update
        for i in range(len(states)):
            state_key = self._get_state_key(states[i])

            # Initialize if needed
            if state_key not in self.policy:
                self.policy[state_key] = np.random.randn(self.action_size) * 0.1
            if state_key not in self.value_table:
                self.value_table[state_key] = 0.0

            # Simple policy gradient update
            advantage = rewards[i] - self.value_table[state_key]
            self.policy[state_key] += self.learning_rate * advantage * actions[i]
            self.policy[state_key] = np.clip(self.policy[state_key], 0, 1)

            # Value update
            self.value_table[state_key] += self.learning_rate * advantage


class A3CTradingAgent:
    """
    A3C-like agent for multi-agent learning
    Uses multiple "worker" perspectives to learn
    """

    def __init__(self, state_size: int = 50, action_size: int = 5, num_workers: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.num_workers = num_workers

        # Global agent
        self.global_agent = DQNTradingAgent(state_size, action_size)

        # Worker agents (share experience)
        self.workers = [DQNTradingAgent(state_size, action_size)
                        for _ in range(num_workers)]

        self.gamma = 0.99
        self.entropy_coef = 0.01

    def get_global_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action from global agent"""
        return self.global_agent.act(state, training)

    def sync_worker(self, worker_id: int):
        """Sync worker with global agent"""
        if 0 <= worker_id < self.num_workers:
            self.workers[worker_id].q_table = self.global_agent.q_table.copy()
            self.workers[worker_id].epsilon = self.global_agent.epsilon

    def aggregate_experience(self):
        """Aggregate experience from workers to global agent"""
        # Simple aggregation - take average Q-values
        if not self.workers:
            return

        all_keys = set()
        for worker in self.workers:
            all_keys.update(worker.q_table.keys())

        for key in all_keys:
            q_values = []
            for worker in self.workers:
                if key in worker.q_table:
                    q_values.append(worker.q_table[key])

            if q_values:
                self.global_agent.q_table[key] = np.mean(q_values, axis=0)


class RLDecisionEngine:
    """
    Combined RL decision engine that uses multiple agents
    """

    def __init__(self):
        self.dqn_agent = DQNTradingAgent(state_size=50, action_size=5)
        self.ppo_agent = PPOTradingAgent(state_size=50, action_size=3)
        self.a3c_agent = A3CTradingAgent(state_size=50, action_size=5, num_workers=4)

        self.initialized = False

    def initialize(self, historical_data):
        """Initialize with historical data"""
        from .trading_env import TradingEnvironment

        env = TradingEnvironment(historical_data)

        print("Training DQN Agent...")
        self.dqn_agent.train(env, episodes=50)

        self.initialized = True

    def make_decision(self, state: np.ndarray) -> Dict:
        """
        Make investment decision combining all RL agents
        """
        if not self.initialized:
            return {
                'action': 'HOLD',
                'position_size': 0,
                'confidence': 0,
                'reasoning': ['RL model not initialized']
            }

        # Get DQN action (discrete)
        dqn_action = self.dqn_agent.act(state, training=False)
        action_name = self.dqn_agent.get_action_name(dqn_action)

        # Get PPO action (continuous)
        ppo_action = self.ppo_agent.act(state, training=False)
        position_size = float(ppo_action[0])

        # Calculate confidence based on Q-value spread
        state_key = self.dqn_agent._get_state_key(state)
        if state_key in self.dqn_agent.q_table:
            q_values = self.dqn_agent.q_table[state_key]
            q_spread = np.max(q_values) - np.min(q_values)
            confidence = min(1.0, float(q_spread) / 2.0)
        else:
            confidence = 0.3

        # Generate reasoning
        reasoning = []
        if dqn_action == 0:  # BUY
            reasoning.append("DQN recommends BUY based on learned pattern")
        elif dqn_action == 1:  # HOLD
            reasoning.append("DQN recommends HOLD - market conditions unclear")
        elif dqn_action == 2:  # SELL
            reasoning.append("DQN recommends SELL to reduce risk")
        elif dqn_action == 3:  # HEDGE
            reasoning.append("DQN recommends HEDGE - moderate risk detected")
        elif dqn_action == 4:  # REBALANCE
            reasoning.append("DQN recommends REBALANCE to optimize allocation")

        reasoning.append(f"Position sizing: {position_size*100:.1f}% of capital")
        reasoning.append(f"Confidence: {confidence*100:.1f}%")

        return {
            'action': action_name,
            'position_size': position_size,
            'confidence': confidence,
            'dqn_action': dqn_action,
            'ppo_position': float(ppo_action[0]),
            'reasoning': reasoning
        }

    def save_model(self, path: str):
        """Save model state"""
        model_state = {
            'dqn_q_table': {str(k): v.tolist() for k, v in self.dqn_agent.q_table.items()},
            'dqn_epsilon': self.dqn_agent.epsilon,
            'ppo_policy': {str(k): v.tolist() for k, v in self.ppo_agent.policy.items()},
            'initialized': self.initialized
        }
        with open(path, 'w') as f:
            json.dump(model_state, f)

    def load_model(self, path: str):
        """Load model state"""
        try:
            with open(path, 'r') as f:
                model_state = json.load(f)

            self.dqn_agent.q_table = {
                tuple(k): np.array(v) for k, v in model_state['dqn_q_table'].items()
            }
            self.dqn_agent.epsilon = model_state['dqn_epsilon']
            self.ppo_agent.policy = {
                tuple(k): np.array(v) for k, v in model_state['ppo_policy'].items()
            }
            self.initialized = model_state['initialized']
        except Exception as e:
            print(f"Error loading model: {e}")


# Global instance
_rl_engine = None


def get_rl_engine() -> RLDecisionEngine:
    """Get or create global RL engine instance"""
    global _rl_engine
    if _rl_engine is None:
        _rl_engine = RLDecisionEngine()
    return _rl_engine
