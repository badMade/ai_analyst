"""Multi-Agent Example.

Demonstrates multiple agents working in parallel in a shared
environment with different strategies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import AutonomousAgent, ReasoningAgent
from src.agents.autonomous_agent import AutonomousConfig
from src.agents.reasoning_agent import ReasoningConfig
from src.environment import Simulator, EnvironmentConfig
from src.utils import MetricsCollector, Visualizer


def run_multi_agent_demo():
    """Run a multi-agent demonstration."""
    print("=" * 60)
    print("Multi-Agent Demo")
    print("=" * 60)

    # Configure environment
    env_config = EnvironmentConfig(
        max_steps=50,
        seed=42,
    )
    env = Simulator(config=env_config, num_resources=20)

    # Create different types of agents
    agents = [
        AutonomousAgent(
            name="Explorer-1",
            config=AutonomousConfig(exploration_rate=0.3),
            goals=["explore"],
        ),
        AutonomousAgent(
            name="Collector-1",
            config=AutonomousConfig(exploration_rate=0.1),
            goals=["collect"],
        ),
        ReasoningAgent(
            name="Strategist-1",
            config=ReasoningConfig(reasoning_depth=3),
        ),
    ]

    # Initialize metrics for each agent
    metrics = {agent.name: MetricsCollector(agent.name) for agent in agents}
    visualizer = Visualizer()

    # Initialize environment
    observation, info = env.reset()

    print(f"\nNumber of agents: {len(agents)}")
    print(f"Resources available: {len(env.resources)}")
    print("-" * 60)

    # Track agent positions (simulated - each agent has virtual position)
    import random
    agent_positions = {
        agent.name: (
            random.uniform(0, env.world_size[0]),
            random.uniform(0, env.world_size[1]),
        )
        for agent in agents
    }

    agent_rewards = {agent.name: 0.0 for agent in agents}

    # Run simulation
    for step in range(env_config.max_steps):
        print(f"\n--- Step {step + 1} ---")

        for agent in agents:
            # Each agent perceives and decides
            perception = {
                "observation": observation,
                "position": agent_positions[agent.name],
                "resources": [r for r in env.resources if not r["collected"]],
            }

            # Agent step
            result = agent.step(perception)

            # Simulate agent-specific behavior
            action = _get_agent_action(agent, perception)

            # Simple reward simulation
            reward = random.uniform(-0.1, 0.5)
            agent_rewards[agent.name] += reward

            metrics[agent.name].record("reward", reward)
            metrics[agent.name].increment("actions")

            # Update simulated position
            dx, dy = _get_movement(action)
            x, y = agent_positions[agent.name]
            agent_positions[agent.name] = (
                max(0, min(env.world_size[0], x + dx)),
                max(0, min(env.world_size[1], y + dy)),
            )

            print(f"  {agent.name}: action={action}, "
                  f"reward={reward:.2f}, "
                  f"total={agent_rewards[agent.name]:.2f}")

    # Print summary
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    # Create comparison chart
    chart = visualizer.bar_chart(
        agent_rewards,
        config=visualizer.config,
    )
    print("\nAgent Performance:")
    print(chart)

    # Print detailed stats
    print("\nDetailed Statistics:")
    for agent_name, collector in metrics.items():
        stats = collector.get_all_stats()
        print(f"\n{agent_name}:")
        if "reward" in stats:
            reward_stats = stats["reward"]
            print(f"  Total reward: {agent_rewards[agent_name]:.2f}")
            if reward_stats:
                print(f"  Mean reward: {reward_stats.get('mean', 0):.3f}")
                print(f"  Min reward: {reward_stats.get('min', 0):.3f}")
                print(f"  Max reward: {reward_stats.get('max', 0):.3f}")

    return agent_rewards


def _get_agent_action(agent, perception) -> str:
    """Get action based on agent type."""
    import random

    if "Explorer" in agent.name:
        # Explorers move randomly
        return random.choice(["move_up", "move_down", "move_left", "move_right"])
    elif "Collector" in agent.name:
        # Collectors focus on collection
        return random.choice(["collect", "move_up", "move_right"])
    else:
        # Strategists balance actions
        return random.choice(["move_up", "collect", "stay"])


def _get_movement(action: str) -> tuple:
    """Get movement delta for action."""
    movements = {
        "move_up": (0, 1),
        "move_down": (0, -1),
        "move_left": (-1, 0),
        "move_right": (1, 0),
        "collect": (0, 0),
        "stay": (0, 0),
    }
    return movements.get(action, (0, 0))


if __name__ == "__main__":
    run_multi_agent_demo()
