"""Base Agent implementation.

Provides the foundation class for all agent types in the Agentic AI Project.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    max_steps: int = 100
    timeout: float = 300.0
    retry_attempts: int = 3
    verbose: bool = True
    log_actions: bool = True


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    result: Any = None


class BaseAgent(ABC):
    """Base class for all agents.

    Provides common functionality for agent lifecycle management,
    action execution, and state tracking.
    """

    def __init__(
        self,
        name: str | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize the base agent.

        Args:
            name: Optional name for the agent. Auto-generated if not provided.
            config: Configuration settings for the agent.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"Agent-{self.id[:8]}"
        self.config = config or AgentConfig()
        self.state = AgentState.IDLE
        self.step_count = 0
        self.action_history: list[AgentAction] = []
        self._memory: dict[str, Any] = {}

        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    @abstractmethod
    def perceive(self, observation: Any) -> dict[str, Any]:
        """Process an observation from the environment.

        Args:
            observation: Raw observation data from the environment.

        Returns:
            Processed perception data.
        """
        pass

    @abstractmethod
    def decide(self, perception: dict[str, Any]) -> AgentAction:
        """Decide on an action based on current perception.

        Args:
            perception: Processed perception data.

        Returns:
            The action to take.
        """
        pass

    @abstractmethod
    def act(self, action: AgentAction) -> Any:
        """Execute an action in the environment.

        Args:
            action: The action to execute.

        Returns:
            The result of the action.
        """
        pass

    def step(self, observation: Any) -> Any:
        """Execute one step of the agent loop.

        Args:
            observation: Current observation from environment.

        Returns:
            Result of the action taken.
        """
        if self.state != AgentState.RUNNING:
            self.state = AgentState.RUNNING

        self.step_count += 1

        if self.step_count > self.config.max_steps:
            logger.warning(f"{self.name}: Max steps reached")
            self.state = AgentState.COMPLETED
            return None

        # Perception -> Decision -> Action loop
        perception = self.perceive(observation)
        action = self.decide(perception)
        result = self.act(action)

        # Record action
        action.result = result
        self.action_history.append(action)

        if self.config.log_actions:
            logger.debug(f"{self.name} Step {self.step_count}: {action.action_type}")

        return result

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.state = AgentState.IDLE
        self.step_count = 0
        self.action_history.clear()
        self._memory.clear()
        logger.info(f"{self.name}: Reset")

    def pause(self) -> None:
        """Pause the agent."""
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            logger.info(f"{self.name}: Paused")

    def resume(self) -> None:
        """Resume a paused agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            logger.info(f"{self.name}: Resumed")

    def get_memory(self, key: str) -> Any:
        """Retrieve a value from agent memory.

        Args:
            key: The memory key.

        Returns:
            The stored value, or None if not found.
        """
        return self._memory.get(key)

    def set_memory(self, key: str, value: Any) -> None:
        """Store a value in agent memory.

        Args:
            key: The memory key.
            value: The value to store.
        """
        self._memory[key] = value

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the agent's current state.

        Returns:
            Dictionary containing state information.
        """
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "step_count": self.step_count,
            "action_count": len(self.action_history),
            "memory_keys": list(self._memory.keys()),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, state={self.state.value})"
