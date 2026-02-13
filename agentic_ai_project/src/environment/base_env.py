"""Base Environment for Agentic AI.

Provides the abstract base class for all environment implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class EnvironmentConfig:
    """Configuration for environments."""
    max_steps: int = 1000
    render_mode: str | None = None
    seed: int | None = None
    reward_scale: float = 1.0
    discount_factor: float = 0.99


@dataclass
class StepResult(Generic[ObsType]):
    """Result of an environment step."""
    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC, Generic[ObsType, ActType]):
    """Abstract base class for environments.

    Follows OpenAI Gym-style interface for compatibility.
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        """Initialize the environment.

        Args:
            config: Environment configuration.
        """
        self.config = config or EnvironmentConfig()
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self._is_initialized = False

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Initial observation and info dict.
        """
        pass

    @abstractmethod
    def step(self, action: ActType) -> StepResult[ObsType]:
        """Execute one step in the environment.

        Args:
            action: The action to execute.

        Returns:
            Step result containing observation, reward, done flags, and info.
        """
        pass

    @abstractmethod
    def get_observation(self) -> ObsType:
        """Get current observation.

        Returns:
            Current observation.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> list[ActType]:
        """Get the action space.

        Returns:
            List of valid actions.
        """
        pass

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Rendered output (depends on render_mode).
        """
        # Default: no rendering
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        self._is_initialized = False

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed.
        """
        self.config.seed = seed

    def get_state(self) -> dict[str, Any]:
        """Get environment state for serialization.

        Returns:
            Serializable state dictionary.
        """
        return {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "config": self.config.__dict__,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore environment state.

        Args:
            state: State dictionary from get_state().
        """
        self.step_count = state.get("step_count", 0)
        self.episode_count = state.get("episode_count", 0)
        self.total_reward = state.get("total_reward", 0.0)

    def is_done(self) -> bool:
        """Check if environment episode is done.

        Returns:
            True if episode is complete.
        """
        return self.step_count >= self.config.max_steps

    def get_metrics(self) -> dict[str, float]:
        """Get environment metrics.

        Returns:
            Dictionary of metrics.
        """
        return {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "avg_reward_per_step": (
                self.total_reward / self.step_count
                if self.step_count > 0 else 0.0
            ),
        }

    def __enter__(self) -> "BaseEnvironment":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
