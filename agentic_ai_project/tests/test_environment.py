"""Tests for environment module."""

import pytest
from src.environment import (
    BaseEnvironment,
    EnvironmentConfig,
    Simulator,
    SimulationState,
)
from src.environment.base_env import StepResult


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnvironmentConfig()
        assert config.max_steps == 1000
        assert config.reward_scale == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnvironmentConfig(
            max_steps=500,
            seed=42,
            reward_scale=2.0,
        )
        assert config.max_steps == 500
        assert config.seed == 42


class TestSimulationState:
    """Tests for SimulationState."""

    def test_default_state(self):
        """Test default state values."""
        state = SimulationState()
        assert state.position == (0.0, 0.0)
        assert state.velocity == (0.0, 0.0)

    def test_to_array(self):
        """Test conversion to array."""
        state = SimulationState(
            position=(1.0, 2.0),
            velocity=(0.5, 0.5),
            orientation=90.0,
            time=10.0,
        )
        arr = state.to_array()
        assert len(arr) == 6
        assert arr[0] == 1.0
        assert arr[1] == 2.0

    def test_from_array(self):
        """Test creation from array."""
        import numpy as np
        arr = np.array([1.0, 2.0, 0.5, 0.5, 90.0, 10.0])
        state = SimulationState.from_array(arr)
        assert state.position == (1.0, 2.0)


class TestSimulator:
    """Tests for Simulator."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = Simulator()
        assert sim.world_size == (100.0, 100.0)
        assert not sim._is_initialized

    def test_reset(self):
        """Test environment reset."""
        sim = Simulator()
        state, info = sim.reset(seed=42)

        assert sim._is_initialized
        assert sim.step_count == 0
        assert "goal" in info
        assert state.position[0] >= 0
        assert state.position[0] <= sim.world_size[0]

    def test_step(self):
        """Test environment step."""
        sim = Simulator()
        sim.reset(seed=42)

        initial_pos = sim.state.position
        result = sim.step("move_right")

        assert isinstance(result, StepResult)
        assert sim.step_count == 1
        assert result.observation.position[0] >= initial_pos[0]

    def test_action_space(self):
        """Test action space."""
        sim = Simulator()
        actions = sim.get_action_space()

        assert "move_up" in actions
        assert "move_down" in actions
        assert "collect" in actions

    def test_movement_bounds(self):
        """Test movement stays within bounds."""
        sim = Simulator(world_size=(10.0, 10.0))
        sim.reset(seed=42)

        # Force position to edge
        sim.state.position = (0.0, 0.0)

        # Try to move out of bounds
        sim.step("move_left")
        sim.step("move_down")

        assert sim.state.position[0] >= 0
        assert sim.state.position[1] >= 0

    def test_resource_collection(self):
        """Test resource collection."""
        sim = Simulator(num_resources=5)
        sim.reset(seed=42)

        # Place resource at agent position
        sim.resources[0]["position"] = sim.state.position
        sim.resources[0]["collected"] = False

        initial_energy = sim.state.resources.get("energy", 0)
        result = sim.step("collect")

        assert sim.resources[0]["collected"]
        assert result.reward > 0

    def test_goal_detection(self):
        """Test goal reached detection."""
        sim = Simulator()
        sim.reset(seed=42)

        # Move agent to goal
        sim.state.position = sim.goal_position
        result = sim.step("stay")

        assert result.terminated

    def test_max_steps_truncation(self):
        """Test truncation at max steps."""
        config = EnvironmentConfig(max_steps=5)
        sim = Simulator(config=config)
        sim.reset()

        for _ in range(10):
            result = sim.step("stay")

        assert result.truncated
        assert sim.step_count == config.max_steps

    def test_render(self):
        """Test text rendering."""
        config = EnvironmentConfig(render_mode="text")
        sim = Simulator(config=config)
        sim.reset(seed=42)

        output = sim.render()
        assert output is not None
        assert "Position" in output

    def test_entity_management(self):
        """Test entity addition and retrieval."""
        sim = Simulator()
        sim.reset()

        entity_id = sim.add_entity(
            "obstacle",
            position=(50.0, 50.0),
            properties={"blocking": True},
        )

        assert "obstacle" in entity_id
        nearby = sim.get_nearby_entities((50.0, 50.0), radius=5.0)
        assert len(nearby) == 1

    def test_context_manager(self):
        """Test context manager usage."""
        with Simulator() as sim:
            sim.reset()
            sim.step("move_up")
            assert sim.step_count == 1

        assert not sim._is_initialized

    def test_state_serialization(self):
        """Test state get/set for checkpointing."""
        sim = Simulator()
        sim.reset(seed=42)
        sim.step("move_up")

        state = sim.get_state()
        assert state["step_count"] == 1

        # Create new simulator and restore state
        sim2 = Simulator()
        sim2.reset()
        sim2.set_state(state)

        assert sim2.step_count == 1

    def test_metrics(self):
        """Test environment metrics."""
        sim = Simulator()
        sim.reset()

        for _ in range(5):
            sim.step("move_up")

        metrics = sim.get_metrics()
        assert metrics["step_count"] == 5
        assert "total_reward" in metrics
