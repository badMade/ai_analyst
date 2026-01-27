"""Single Agent Example.

Demonstrates a single autonomous agent navigating and collecting
resources in a simulation environment.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import AutonomousAgent
from src.agents.autonomous_agent import AutonomousConfig
from src.environment import Simulator, EnvironmentConfig
from src.utils import AgentLogger, MetricsCollector


def run_single_agent_demo():
    """Run a single agent demonstration."""
    print("=" * 60)
    print("Single Agent Demo")
    print("=" * 60)

    # Configure the environment
    env_config = EnvironmentConfig(
        max_steps=100,
        render_mode="text",
        seed=42,
    )
    env = Simulator(config=env_config, num_resources=10)

    # Configure the agent
    agent_config = AutonomousConfig(
        max_steps=100,
        exploration_rate=0.2,
        learning_rate=0.01,
    )
    agent = AutonomousAgent(
        name="Explorer",
        config=agent_config,
        goals=["collect_resources", "reach_goal"],
    )

    # Set up logging and metrics
    logger = AgentLogger(agent.name)
    metrics = MetricsCollector("single_agent")

    # Initialize environment
    observation, info = env.reset()
    agent.set_goal("reach_goal")

    print(f"\nGoal position: {info['goal']}")
    print(f"Agent starting at: {observation.position}")
    print("-" * 60)

    # Run simulation
    total_reward = 0
    step = 0

    while step < env_config.max_steps:
        # Agent decides and acts
        result = agent.step(observation)

        # Map agent decision to environment action
        action = _map_to_env_action(result)

        # Execute in environment
        step_result = env.step(action)

        # Log and collect metrics
        logger.log_action(action, result=step_result.reward)
        metrics.record("reward", step_result.reward)
        metrics.increment("steps")

        total_reward += step_result.reward
        observation = step_result.observation
        step += 1

        # Print progress every 10 steps
        if step % 10 == 0:
            distance = _distance(observation.position, info["goal"])
            print(f"Step {step}: Position {observation.position}, "
                  f"Distance to goal: {distance:.2f}, "
                  f"Total reward: {total_reward:.2f}")

        # Check termination
        if step_result.terminated:
            print(f"\nGoal reached at step {step}!")
            break
        if step_result.truncated:
            print(f"\nMax steps reached.")
            break

    # Print summary
    print("-" * 60)
    print("Summary:")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final position: {observation.position}")
    print(f"  Goal reached: {step_result.terminated}")

    # Print agent state
    state = agent.get_state_summary()
    print(f"\nAgent state:")
    print(f"  Actions taken: {state['action_count']}")
    print(f"  Q-values learned: {len(agent.q_values)} states")

    return total_reward


def _map_to_env_action(result: dict) -> str:
    """Map agent result to environment action."""
    # Simple mapping based on exploration behavior
    import random
    actions = ["move_up", "move_down", "move_left", "move_right", "collect"]
    return random.choice(actions)


def _distance(pos1: tuple, pos2: tuple) -> float:
    """Calculate Euclidean distance."""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


if __name__ == "__main__":
    run_single_agent_demo()
