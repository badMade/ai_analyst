"""Learning Agent implementation.

An agent that learns from experience using various machine learning
techniques including reinforcement learning and supervised learning.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging
import numpy as np

from .base_agent import BaseAgent, AgentConfig, AgentAction

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig(AgentConfig):
    """Configuration for learning agents."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "mse"
    discount_factor: float = 0.99
    target_update_frequency: int = 100


@dataclass
class Experience:
    """A single experience tuple for replay."""
    state: Any
    action: AgentAction
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """Experience replay buffer for learning."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.position = 0

    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences."""
```suggestion
        """Sample a batch of experiences."""
        batch_size = min(batch_size, len(self.buffer)) # Ensure batch_size is not larger than buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class LearningAgent(BaseAgent):
    """Agent that learns from experience.

    Implements experience replay and can be extended with various
    learning algorithms (DQN, PPO, etc.).
    """

    def __init__(
        self,
        name: str | None = None,
        config: LearningConfig | None = None,
        model: Any = None,
    ):
        """Initialize the learning agent.

        Args:
            name: Optional name for the agent.
            config: Configuration settings.
            model: Optional pre-trained model.
        """
        super().__init__(name, config or LearningConfig())
        self.config: LearningConfig = self.config
        self.model = model
        self.target_model = None
        self.replay_buffer = ReplayBuffer()
        self.training_step = 0
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0
        self._loss_history: list[float] = []

    def perceive(self, observation: Any) -> dict[str, Any]:
        """Process observation into state representation.

        Args:
            observation: Raw observation from environment.

        Returns:
            Processed state representation.
        """
        state = self._preprocess_observation(observation)
        return {
            "state": state,
            "features": self._extract_features(state),
            "normalized": self._normalize_state(state),
        }

    def decide(self, perception: dict[str, Any]) -> AgentAction:
        """Select action using the learned policy.

        Args:
            perception: Processed perception data.

        Returns:
            Selected action.
        """
        state = perception.get("normalized", perception.get("state"))

        if self.model is not None:
            action_values = self._predict(state)
            action_idx = int(np.argmax(action_values))
        else:
            # Random action if no model
            action_idx = np.random.randint(0, 4)
            action_idx = np.random.randint(0, self.config.action_space_size)
        return AgentAction(
            action_type="learned_action",
            parameters={"action_index": action_idx, "state": state}
        )

    def act(self, action: AgentAction) -> Any:
        """Execute action and collect experience.

        Args:
            action: Action to execute.

        Returns:
            Action result including reward.
        """
        result = self._execute_action(action)
        return result

    def learn(
        self,
        state: Any,
        action: AgentAction,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float | None:
        """Learn from a single experience.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode ended.

        Returns:
            Training loss if learning occurred.
        """
        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

        # Update episode reward
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Train if enough experiences
        if len(self.replay_buffer) >= self.config.batch_size:
            loss = self._train_step()
            return loss

        return None

    def train_on_batch(self, batch: list[Experience]) -> float:
        """Train the model on a batch of experiences.

        Args:
            batch: List of experience tuples.

        Returns:
            Training loss.
        """
        if self.model is None:
            logger.warning(f"{self.name}: No model to train")
            return 0.0

        # Extract batch components
        states = [e.state for e in batch]
        actions = [e.action.parameters.get("action_index", 0) for e in batch]
        rewards = [e.reward for e in batch]
        next_states = [e.next_state for e in batch]
        dones = [e.done for e in batch]

        # Compute targets (simplified DQN-style)
        loss = self._compute_loss(states, actions, rewards, next_states, dones)
        self._loss_history.append(loss)

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.config.target_update_frequency == 0:
            self._update_target_network()

        return loss

    def save_model(self, path: str) -> None:
        """Save the learned model.

        Args:
            path: Path to save model.
        """
        logger.info(f"{self.name}: Saving model to {path}")
        # Placeholder - implement actual model saving

    def load_model(self, path: str) -> None:
        """Load a pre-trained model.

        Args:
            path: Path to model file.
        """
        logger.info(f"{self.name}: Loading model from {path}")
        # Placeholder - implement actual model loading

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training metrics.
        """
        return {
            "training_steps": self.training_step,
            "episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_loss": np.mean(self._loss_history[-100:]) if self._loss_history else 0.0,
            "buffer_size": len(self.replay_buffer),
        }

    def _preprocess_observation(self, observation: Any) -> Any:
        """Preprocess raw observation."""
        if isinstance(observation, np.ndarray):
            return observation.flatten()
        if isinstance(observation, (list, tuple)):
            return np.array(observation)
        return observation

    def _extract_features(self, state: Any) -> np.ndarray:
        """Extract features from state."""
        if isinstance(state, np.ndarray):
            return state
        return np.array([state])

    def _normalize_state(self, state: Any) -> Any:
        """Normalize state values."""
        if isinstance(state, np.ndarray):
            mean = np.mean(state)
            std = np.std(state) + 1e-8
            return (state - mean) / std
        return state

    def _predict(self, state: Any) -> np.ndarray:
        """Get action values from model."""
        # Placeholder - implement actual model prediction
        return np.random.rand(4)

    def _execute_action(self, action: AgentAction) -> dict[str, Any]:
        """Execute action in environment."""
        return {
            "success": True,
            "action_index": action.parameters.get("action_index"),
        }

    def _train_step(self) -> float:
        """Perform one training step."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        return self.train_on_batch(batch)

    def _compute_loss(
        self,
        states: list,
        actions: list,
        rewards: list,
        next_states: list,
        dones: list,
    ) -> float:
        """Compute training loss."""
        # Placeholder - implement actual loss computation
        # Placeholder - implement actual loss computation
        # Example (conceptual, requires actual model implementation):
        # current_q_values = self.model.predict(states)
        # next_q_values = self.target_model.predict(next_states)
        # target_q_values = rewards + self.config.discount_factor * np.max(next_q_values, axis=1) * (1 - np.array(dones))
        # loss = np.mean(np.square(target_q_values - current_q_values[np.arange(len(actions)), actions]))
        # return loss
        return np.random.rand() * 0.1

    def _update_target_network(self) -> None:
        """Update target network weights."""
        if self.target_model is not None:
            logger.debug(f"{self.name}: Updating target network")
            # Placeholder - implement actual weight copy
