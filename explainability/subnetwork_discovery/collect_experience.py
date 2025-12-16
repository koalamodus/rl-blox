from flax import nnx
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer
import jax.numpy as jnp
import gymnasium as gym

# Gym environment to get obs and action dimensions
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42


# Recreate MLP with same architecture as training
hparams_model = dict(
    activation="relu",
    hidden_nodes=[128, 128],
)

q = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    rngs=nnx.Rngs(seed),
    **hparams_model,
)

import pickle

with open("ddqn_model_ckpt.pkl", "rb") as f:
    mlp_state = pickle.load(f)
print("Loaded model checkpoint.")

print(f"mlp_state is {mlp_state}")


# Update the network with the saved parameters
nnx.update(q, mlp_state)
print("Loaded trained MLP parameters successfully")


# Initialise the replay buffer
hparams_algorithm = dict(
    buffer_size=50_000,
    total_timesteps=50_000,
    seed=seed,
)

import warnings
from tqdm import tqdm


if hparams_algorithm["buffer_size"] > hparams_algorithm["total_timesteps"]:
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    warnings.warn(
        f"{YELLOW}ReplayBuffer size ({hparams_algorithm['buffer_size']}) is larger than "
        f"the total experience collected ({hparams_algorithm['total_timesteps']}). "
        f"ReplayBuffer will not be full.{RESET}",
        UserWarning,
    )

rb = ReplayBuffer(hparams_algorithm.pop("buffer_size"), discrete_actions=True)

# Collect experience with trained policy
collect_env = gym.make(env_name)
obs, _ = collect_env.reset()

for _ in tqdm(range(hparams_algorithm.pop("total_timesteps")), desc="Experience collection"):
    action = int(jnp.argmax(q([obs])))
    next_obs, reward, terminated, truncated, info = collect_env.step(action)

    rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=terminated or truncated,
        )
    
    if terminated or truncated:
        obs, _ = collect_env.reset()
    else:
        obs = next_obs

# Save experience in replay buffer
with open("rb_ckpt.pkl", "wb") as f:
    pickle.dump(rb, f)
print("Saved replay buffer checkpoint.")


# Show the final policy
eval_env = gym.make(env_name, render_mode="human")
obs, _ = eval_env.reset()

while True:
    action = int(jnp.argmax(q([obs])))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs
