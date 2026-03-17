from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from minigrid_one_context_env import make_ocean_env
import gymnasium as gym
import jax.numpy as jnp
import optax
from flax import nnx
import pickle

from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import ReplayBuffer
from rl_blox.logging.logger import AIMLogger

seed = 42
# ---------------------------
# Test env
# ---------------------------
print("Test env")
obj_colors = ["red", "green", "blue",] # COLOR_NAMES  #
subtask = None # "red", "green", "blue", "purple", "yellow", "grey" or None
env = make_ocean_env(target_color=subtask, obj_colors=obj_colors, render_mode="human")
num_eps=3

print(f"action space: {env.action_space}")
print(f"observation space: {env.observation_space}")

for _ in range(num_eps):
    obs, _ = env.reset(seed=seed)
    done = False

    # for _ in range(num_steps):
    while not done:    
        action = env.action_space.sample()
        # print(f"action={action}")
        # action = 2
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"obs={obs}")
        # print(f"---")

        done = terminated or truncated
        if done:
            print(f"obs={obs}")
            print(f"{env.current_task_color} reward: {reward}")

env.close()

# ---------------------------
# Train ddqn
# ---------------------------
print("Train ddqn")

# 1. train a network with ddqn example 
# Set up environment

env_name = f"minigrid_random_{subtask}"
env = make_ocean_env(target_color=subtask, obj_colors=obj_colors)
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(seed)

hparams_model = dict(
    activation="relu",
    hidden_nodes=[128, 128],
)
hparams_algorithm = dict(
    batch_size=64,
    buffer_size=50_000,
    total_timesteps=100_000,
    learning_rate=0.002,
    learning_rate_start=2e-3,
    learning_rate_end=1e-4,
    #learning_starts=2_000,
    seed=seed,
)


logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="DDQN",
    hparams=hparams_model | hparams_algorithm,
)
# Initialise the Q-Network
q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    rngs=nnx.Rngs(seed),
    **hparams_model,
)

# Initialise the replay buffer
rb = ReplayBuffer(hparams_algorithm.pop("buffer_size"), discrete_actions=True)

lr_constant = hparams_algorithm.pop("learning_rate")
# Set learning rate scheduler
lr_schedule_fn = optax.linear_schedule(
   init_value=hparams_algorithm.pop("learning_rate_start"), end_value=hparams_algorithm.pop("learning_rate_end"), transition_steps=hparams_algorithm.get("total_timesteps"))

# lr = lr_constant
lr = lr_schedule_fn

# Initialise optimiser
optimizer = nnx.Optimizer(
    q_net, optax.adam(learning_rate=lr), wrt=nnx.Param
)

# Train
result_dict = train_ddqn(
    q_net,
    env,
    rb,
    optimizer,
    **hparams_algorithm,
    logger=logger,
)
q = result_dict.q_net

# print(f"q:{q}")

env.close()

# Save trained policy
mlp_state = nnx.state(q)   # dict of parameters
# print(f"mlp_state is {mlp_state}")

with open("ddqn_minigrid_ckpt.pkl", "wb") as f:
    pickle.dump(mlp_state, f)
print("Saved model checkpoint.")

def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy
policy = get_policy_from_q_net(q)



from eval_helper import eval_policy
def evaluate_policy_on_task(policy, task=None, obj_colors=obj_colors, render_mode="human"):
    if task == None:
        task = obj_colors
    else:
        task = task if isinstance(task, list) else [task]
    
    for target_color in task:
        assert target_color in COLOR_NAMES

        eval_env = make_ocean_env(target_color, obj_colors, render_mode)
        _ = eval_policy(target_color, eval_env, policy, verbose=True, num_episode=20)
        eval_env.close()

print("evaluate original q network")
evaluate_policy_on_task(policy, task=subtask, obj_colors=obj_colors, render_mode="None")

# Show the final policy
eval_env = make_ocean_env(target_color=subtask, obj_colors=obj_colors, render_mode="human")
obs, _ = eval_env.reset(seed=seed+1)
while True:
    action = policy(obs)
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        # print(f"obs={obs}")
        print(f"{eval_env.current_task_color} reward: {reward}")
        obs, _ = eval_env.reset(seed=seed+2)
    else:
        obs = next_obs

