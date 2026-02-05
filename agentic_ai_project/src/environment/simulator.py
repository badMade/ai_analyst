"""Simulator Environment for Agentic AI.

Provides a simulation environment for testing and training agents.
"""

from dataclasses import dataclass, field
from typing import Any
import random
import logging
import numpy as np

from .base_env import BaseEnvironment, EnvironmentConfig, StepResult

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """State of the simulation."""
    position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    resources: dict[str, float] = field(default_factory=dict)
    time: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1],
            self.orientation,
            self.time,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SimulationState":
        """Create state from numpy array."""
        return cls(
            position=(float(arr[0]), float(arr[1])),
            velocity=(float(arr[2]), float(arr[3])),
            orientation=float(arr[4]),
            time=float(arr[5]),
        )


class Simulator(BaseEnvironment[SimulationState, str]):
    """Simulation environment for agent testing and training.

    Provides a customizable simulation with:
    - Multiple agent support
    - Resource management
    - Goal-based rewards
    - Configurable physics
    """

    # Available actions
    ACTIONS = ["move_up", "move_down", "move_left", "move_right",
               "stay", "collect", "interact"]

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        world_size: tuple[float, float] = (100.0, 100.0),
        num_resources: int = 10,
    ):
        """Initialize the simulator.

        Args:
            config: Environment configuration.
            world_size: Size of the simulation world.
            num_resources: Number of resources to spawn.
        """
        super().__init__(config)
        self.world_size = world_size
        self.num_resources = num_resources
        self.state = SimulationState()
        self.goal_position: tuple[float, float] | None = None
        self.resources: list[dict[str, Any]] = []
        self._rng = random.Random(self.config.seed)

    def reset(self, seed: int | None = None) -> tuple[SimulationState, dict[str, Any]]:
        """Reset the simulation to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Initial state and info.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        # Reset counters
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_count += 1

        # Initialize state
        self.state = SimulationState(
            position=(
                self._rng.uniform(0, self.world_size[0]),
                self._rng.uniform(0, self.world_size[1]),
            ),
            velocity=(0.0, 0.0),
            orientation=self._rng.uniform(0, 360),
            resources={"energy": 100.0, "health": 100.0},
            time=0.0,
        )

        # Spawn resources
        self._spawn_resources()

        # Set random goal
        self.goal_position = (
            self._rng.uniform(0, self.world_size[0]),
            self._rng.uniform(0, self.world_size[1]),
        )

        self._is_initialized = True
        logger.info(f"Simulator reset - Episode {self.episode_count}")

        return self.state, {"goal": self.goal_position}

    def step(self, action: str) -> StepResult[SimulationState]:
        """Execute one simulation step.

        Args:
            action: Action to execute.

        Returns:
            Step result.
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.step_count += 1
        self.state.time += 0.1  # Time step

        # Execute action
        reward = self._execute_action(action)

        # Update physics
        self._update_physics()

        # Check termination
        terminated = self._check_goal_reached()
        truncated = self.step_count >= self.config.max_steps

        # Apply reward scaling
        reward *= self.config.reward_scale
        self.total_reward += reward

        info = {
            "distance_to_goal": self._distance_to_goal(),
            "resources_collected": len([r for r in self.resources if r.get("collected")]),
            "energy": self.state.resources.get("energy", 0),
        }

        return StepResult(
            observation=self.state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_observation(self) -> SimulationState:
        """Get current simulation state."""
        return self.state

    def get_action_space(self) -> list[str]:
        """Get available actions."""
        return self.ACTIONS.copy()

    def render(self) -> str | None:
        """Render simulation state as text.

        Returns:
            Text representation of state.
        """
        if self.config.render_mode != "text":
            return None

        lines = [
            f"=== Simulation Step {self.step_count} ===",
            f"Position: ({self.state.position[0]:.2f}, {self.state.position[1]:.2f})",
            f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})" if self.goal_position else "Goal: None",
            f"Distance to goal: {self._distance_to_goal():.2f}",
            f"Energy: {self.state.resources.get('energy', 0):.1f}",
            f"Resources available: {len([r for r in self.resources if not r.get('collected')])}",
        ]
        return "\n".join(lines)

    def add_entity(
        self,
        entity_type: str,
        position: tuple[float, float],
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add an entity to the simulation.

        Args:
            entity_type: Type of entity.
            position: Entity position.
            properties: Optional properties.

        Returns:
            Entity ID.
        """
        entity_id = f"{entity_type}_{len(self.state.entities)}"
        entity = {
            "id": entity_id,
            "type": entity_type,
            "position": position,
            "properties": properties or {},
        }
        self.state.entities.append(entity)
        return entity_id

    def get_nearby_entities(
        self,
        position: tuple[float, float],
        radius: float,
    ) -> list[dict[str, Any]]:
        """Get entities within radius of position.

        Args:
            position: Center position.
            radius: Search radius.

        Returns:
            List of nearby entities.
        """
        nearby = []
        for entity in self.state.entities:
            dist = self._calculate_distance(position, entity["position"])
            if dist <= radius:
                nearby.append(entity)
        return nearby

    def _spawn_resources(self) -> None:
        """Spawn resources in the world."""
        self.resources = []
        for i in range(self.num_resources):
            resource = {
                "id": f"resource_{i}",
                "position": (
                    self._rng.uniform(0, self.world_size[0]),
                    self._rng.uniform(0, self.world_size[1]),
                ),
                "value": self._rng.uniform(5, 20),
                "collected": False,
            }
            self.resources.append(resource)

    def _execute_action(self, action: str) -> float:
        """Execute an action and return reward.

        Args:
            action: Action to execute.

        Returns:
            Reward for the action.
        """
        reward = -0.01  # Small negative reward for each step (encourages efficiency)
        move_speed = 1.0

        x, y = self.state.position

        if action == "move_up":
            y = min(self.world_size[1], y + move_speed)
        elif action == "move_down":
            y = max(0, y - move_speed)
        elif action == "move_left":
            x = max(0, x - move_speed)
        elif action == "move_right":
            x = min(self.world_size[0], x + move_speed)
        elif action == "collect":
            reward += self._collect_nearby_resources()
        elif action == "interact":
            reward += self._interact_with_entities()
        # "stay" does nothing

        self.state.position = (x, y)

        # Energy cost for movement
        if action.startswith("move"):
            self.state.resources["energy"] = max(
                0,
                self.state.resources.get("energy", 0) - 0.5
            )

        return reward

    def _collect_nearby_resources(self) -> float:
        """Collect resources near current position.

        Returns:
            Reward from collected resources.
        """
        reward = 0.0
        collection_radius = 2.0

        for resource in self.resources:
            if resource["collected"]:
                continue

            dist = self._calculate_distance(self.state.position, resource["position"])
            if dist <= collection_radius:
                resource["collected"] = True
                reward += resource["value"]
                self.state.resources["energy"] = min(
                    100,
                    self.state.resources.get("energy", 0) + resource["value"]
                )
                logger.debug(f"Collected resource worth {resource['value']:.1f}")

        return reward

    def _interact_with_entities(self) -> float:
        """Interact with nearby entities.

        Returns:
            Reward from interactions.
        """
        nearby = self.get_nearby_entities(self.state.position, 3.0)
        return len(nearby) * 0.5

    def _update_physics(self) -> None:
        """Update physics simulation."""
        # Apply velocity
        vx, vy = self.state.velocity
        x, y = self.state.position

        x = max(0, min(self.world_size[0], x + vx * 0.1))
        y = max(0, min(self.world_size[1], y + vy * 0.1))

        self.state.position = (x, y)

        # Apply drag
        self.state.velocity = (vx * 0.95, vy * 0.95)

    def _check_goal_reached(self) -> bool:
        """Check if goal has been reached."""
        if self.goal_position is None:
            return False

        distance = self._distance_to_goal()
        return distance < 2.0

    def _distance_to_goal(self) -> float:
        """Calculate distance to goal."""
        if self.goal_position is None:
            return float("inf")
        return self._calculate_distance(self.state.position, self.goal_position)

    def _calculate_distance(
        self,
        pos1: tuple[float, float],
        pos2: tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
