from flax import nnx
from rl_blox.blox.function_approximator.mlp import MLP
import jax.numpy as jnp
import gymnasium as gym
import pickle


def evaluate_q_network(env_name, q_net, num_episodes=20):
    """Evaluate a Q-network on the environment for a number of episodes."""
    env = gym.make(env_name)
    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0

        while not done:
            # Greedy action
            action = int(jnp.argmax(q_net([obs])))
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs

        episode_rewards.append(total_reward)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    return avg_reward, episode_rewards

def compute_performance_drop(avg_reward_original, avg_reward_pruned):
    """
    Compute the percentage performance drop in percentage of a pruned network.
    """
    performance_drop = 100.0 * (avg_reward_original - avg_reward_pruned) / max(1e-8, avg_reward_original)
    return performance_drop


if __name__ == "__main__":
    num_episodes = 100

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    seed = 42

    # Recreate MLP with same architecture as training
    hparams_model = dict(
        activation="relu",
        hidden_nodes=[128, 128],
    )

    q = MLP(env.observation_space.shape[0], int(env.action_space.n), rngs=nnx.Rngs(seed), **hparams_model)
    q_pruned = MLP(env.observation_space.shape[0], int(env.action_space.n), rngs=nnx.Rngs(seed + 99), **hparams_model)

    # Load and evaluate original q network
    with open("ddqn_model_ckpt.pkl", "rb") as f:
        mlp_state = pickle.load(f)
    nnx.update(q, mlp_state)
    print("Loaded trained q network successfully")

    avg_reward_q, rewards_q = evaluate_q_network(env_name, q, num_episodes)
    print(f"Original Q-network avg reward over {num_episodes} episodes: {avg_reward_q:.2f}")

    # Load and evaluate the pruned network
    with open("ddqn_subnet_ckpt.pkl", "rb") as f:
        mlp_state = pickle.load(f)
    nnx.update(q, mlp_state)
    print("Loaded pruned Q-network successfully")

    # Evaluate pruned network
    avg_reward_q_pruned, rewards_q_pruned = evaluate_q_network(env_name, q_pruned, num_episodes)
    print(f"Pruned Q-network avg reward over {num_episodes} episodes: {avg_reward_q_pruned:.2f}")

    # Compute performance drop
    performance_drop = compute_performance_drop(avg_reward_q, avg_reward_q_pruned)
    print(f"Performance drop due to pruning: {performance_drop:.2f}%")