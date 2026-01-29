"""Reinforcement Learning Example.

Demonstrates a learning agent training through interaction
with the environment using experience replay.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import LearningAgent
from src.agents.learning_agent import LearningConfig
from src.agents.base_agent import AgentAction
from src.environment import Simulator, EnvironmentConfig
from src.utils import MetricsCollector, Visualizer


def run_rl_demo():
    """Run a reinforcement learning demonstration."""
    print("=" * 60)
    print("Reinforcement Learning Demo")
    print("=" * 60)

    # Configure for training
    env_config = EnvironmentConfig(
        max_steps=200,
        seed=None,  # Random seed for diverse training
    )
    env = Simulator(config=env_config, num_resources=15)

    # Configure learning agent
    agent_config = LearningConfig(
        learning_rate=0.001,
        batch_size=16,
        discount_factor=0.99,
    )
    agent = LearningAgent(name="Learner", config=agent_config)

    # Metrics tracking
    metrics = MetricsCollector("rl_training")
    visualizer = Visualizer()

    # Training parameters
    num_episodes = 10
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.95
    epsilon_min = 0.1

    episode_rewards = []
    episode_lengths = []

    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)

    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        step = 0

        while step < env_config.max_steps:
            # Epsilon-greedy action selection
            import random
            if random.random() < epsilon:
                action_idx = random.randint(0, len(env.ACTIONS) - 1)
            else:
                # Use agent's learned policy
```suggestion
            # Use agent's learned policy
            action = env.ACTIONS[action_idx]
            step_result = agent.step(observation)
            action_idx = env.ACTIONS.index(step_result.action.action_type)

            action = env.ACTIONS[action_idx]

            # Execute action
            step_result = env.step(action)

            # Convert observations for learning
            state = observation.to_array()
            next_state = step_result.observation.to_array()
            reward = step_result.reward
            done = step_result.terminated or step_result.truncated

            # Learn from experience
            loss = agent.learn(
                state=state,
                action=AgentAction(
                    action_type=action,
                    parameters={"action_index": action_idx, "state": state},
                ),
                reward=reward,
                next_state=next_state,
                done=done,
            )

            if loss is not None:
                metrics.record("loss", loss)

            episode_reward += reward
            observation = step_result.observation
            step += 1

            if done:
                break

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        metrics.record("episode_reward", episode_reward)

        print(f"Episode {episode + 1:3d}: "
              f"Reward = {episode_reward:7.2f}, "
              f"Steps = {step:3d}, "
              f"Epsilon = {epsilon:.3f}, "
              f"Buffer = {len(agent.replay_buffer)}")

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    # Show reward progression
    print("\nReward Progression:")
    sparkline = visualizer.sparkline(episode_rewards, width=40)
    print(f"  {sparkline}")
    print(f"  Start: {episode_rewards[0]:.2f} -> End: {episode_rewards[-1]:.2f}")

    # Calculate statistics
    import statistics
    print(f"\nStatistics:")
    print(f"  Mean reward: {statistics.mean(episode_rewards):.2f}")
    print(f"  Max reward: {max(episode_rewards):.2f}")
    print(f"  Min reward: {min(episode_rewards):.2f}")
    print(f"  Std dev: {statistics.stdev(episode_rewards):.2f}")

    # Agent training stats
    training_stats = agent.get_training_stats()
    print(f"\nAgent Training Stats:")
    print(f"  Training steps: {training_stats['training_steps']}")
    print(f"  Episodes completed: {training_stats['episodes']}")
    print(f"  Buffer utilization: {training_stats['buffer_size']}")

    # Show improvement
    first_half = statistics.mean(episode_rewards[:len(episode_rewards)//2])
    second_half = statistics.mean(episode_rewards[len(episode_rewards)//2:])
    improvement = ((second_half - first_half) / abs(first_half)) * 100 if first_half != 0 else 0

    print(f"\nLearning Progress:")
    print(f"  First half avg: {first_half:.2f}")
    print(f"  Second half avg: {second_half:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")

    return episode_rewards


if __name__ == "__main__":
    run_rl_demo()
