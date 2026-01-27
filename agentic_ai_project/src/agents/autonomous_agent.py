"""Autonomous Agent implementation.

An agent that operates independently with goal-seeking behavior,
exploration strategies, and adaptive learning.
"""

from dataclasses import dataclass
from typing import Any
import random
import logging

from .base_agent import BaseAgent, AgentConfig, AgentAction, AgentState

logger = logging.getLogger(__name__)


@dataclass
class AutonomousConfig(AgentConfig):
    """Configuration for autonomous agents."""
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    memory_size: int = 10000
    goal_threshold: float = 0.95


class AutonomousAgent(BaseAgent):
    """Autonomous agent with goal-seeking behavior.

    This agent can operate independently, setting and pursuing goals,
    exploring its environment, and adapting its behavior based on
    feedback.
    """

    def __init__(
        self,
        name: str | None = None,
        config: AutonomousConfig | None = None,
        goals: list[str] | None = None,
    ):
        """Initialize the autonomous agent.

        Args:
            name: Optional name for the agent.
            config: Configuration settings.
            goals: List of goals for the agent to pursue.
        """
        super().__init__(name, config or AutonomousConfig())
        self.config: AutonomousConfig = self.config
        self.goals = goals or []
        self.current_goal: str | None = None
        self.goal_progress: dict[str, float] = {}
        self.exploration_memory: list[dict[str, Any]] = []
        self.q_values: dict[str, dict[str, float]] = {}

    def perceive(self, observation: Any) -> dict[str, Any]:
        """Process environment observation.

        Args:
            observation: Raw observation from environment.

        Returns:
            Processed perception with state analysis.
        """
        perception = {
            "raw": observation,
            "state": self._extract_state(observation),
            "opportunities": self._identify_opportunities(observation),
            "threats": self._identify_threats(observation),
            "goal_relevance": self._assess_goal_relevance(observation),
        }

        # Store in exploration memory
        if len(self.exploration_memory) >= self.config.memory_size:
            self.exploration_memory.pop(0)
        self.exploration_memory.append(perception)

        return perception

    def decide(self, perception: dict[str, Any]) -> AgentAction:
        """Decide on action using epsilon-greedy strategy.

        Args:
            perception: Processed perception data.

        Returns:
            Selected action.
        """
        state = perception.get("state", "unknown")

        # Exploration vs exploitation
        if random.random() < self.config.exploration_rate:
            action = self._explore(perception)
            logger.debug(f"{self.name}: Exploring - {action.action_type}")
        else:
            action = self._exploit(perception, state)
            logger.debug(f"{self.name}: Exploiting - {action.action_type}")

        return action

    def act(self, action: AgentAction) -> Any:
        """Execute the decided action.

        Args:
            action: Action to execute.

        Returns:
            Action result.
        """
        result = self._execute_action(action)
        self._update_q_values(action, result)
        return result

    def set_goal(self, goal: str) -> None:
        """Set a new goal for the agent.

        Args:
            goal: The goal to pursue.
        """
        if goal not in self.goals:
            self.goals.append(goal)
        self.current_goal = goal
        self.goal_progress[goal] = 0.0
        logger.info(f"{self.name}: New goal set - {goal}")

    def update_goal_progress(self, progress: float) -> None:
        """Update progress on current goal.

        Args:
            progress: Progress value (0.0 to 1.0).
        """
        if self.current_goal:
            self.goal_progress[self.current_goal] = min(1.0, progress)
            if progress >= self.config.goal_threshold:
                logger.info(f"{self.name}: Goal achieved - {self.current_goal}")
                self._complete_goal()

    def _extract_state(self, observation: Any) -> str:
        """Extract state representation from observation."""
        if isinstance(observation, dict):
            return str(hash(frozenset(observation.items())))
        return str(hash(str(observation)))

    def _identify_opportunities(self, observation: Any) -> list[str]:
        """Identify opportunities in the observation."""
        # Placeholder - override in subclass for domain-specific logic
        return []

    def _identify_threats(self, observation: Any) -> list[str]:
        """Identify potential threats or obstacles."""
        # Placeholder - override in subclass for domain-specific logic
        return []

    def _assess_goal_relevance(self, observation: Any) -> float:
        """Assess how relevant observation is to current goal."""
        if not self.current_goal:
            return 0.0
        # Placeholder - override for domain-specific relevance scoring
        return 0.5

    def _explore(self, perception: dict[str, Any]) -> AgentAction:
        """Generate an exploratory action."""
        opportunities = perception.get("opportunities", [])
        if opportunities:
            target = random.choice(opportunities)
            return AgentAction(
                action_type="explore",
                parameters={"target": target}
            )
        return AgentAction(action_type="random_move", parameters={})

    def _exploit(self, perception: dict[str, Any], state: str) -> AgentAction:
        """Select best known action for state."""
        if state in self.q_values and self.q_values[state]:
            best_action = max(self.q_values[state], key=self.q_values[state].get)
            return AgentAction(
                action_type=best_action,
                parameters={"state": state}
            )
        return self._explore(perception)

    def _execute_action(self, action: AgentAction) -> Any:
        """Execute the action and return result."""
        # Placeholder - override for actual action execution
        return {"success": True, "action": action.action_type}

    def _update_q_values(self, action: AgentAction, result: Any) -> None:
        """Update Q-values based on action result."""
        state = action.parameters.get("state", "unknown")
        action_type = action.action_type

        if state not in self.q_values:
            self.q_values[state] = {}

        current_q = self.q_values[state].get(action_type, 0.0)
        reward = self._calculate_reward(result)

        # Q-learning update
        new_q = current_q + self.config.learning_rate * (reward - current_q)
        self.q_values[state][action_type] = new_q

    def _calculate_reward(self, result: Any) -> float:
        """Calculate reward from action result."""
        if isinstance(result, dict) and result.get("success"):
            return 1.0
        return -0.1

    def _complete_goal(self) -> None:
        """Handle goal completion."""
        if self.current_goal:
            completed = self.current_goal
            self.goals.remove(completed)
            if self.goals:
                self.current_goal = self.goals[0]
            else:
                self.current_goal = None
                self.state = AgentState.COMPLETED
