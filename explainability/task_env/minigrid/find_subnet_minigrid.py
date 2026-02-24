import os
from minigrid.core.constants import COLOR_NAMES
from minigrid_envs import make_ocean_env

import jax

seed = 10  # random seed for np and jax
key = jax.random.PRNGKey(seed)
# ---------------------------
# (1) Load trained network
# ---------------------------

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

def get_mlp_with_state(env, state):
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [512, 512],
        "relu",
        nnx.Rngs(0),
    )

    graphdef, abstract_state = nnx.split(abstract_mlp)
    mlp = nnx.merge(graphdef, state)

    return mlp

def get_q_net_from_checkpoint(path, env):
    full_path = os.path.abspath(path)
    print(f"ckpts: {full_path}")
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [512, 512],
        "relu",
        nnx.Rngs(0),
    )

    graphdef, abstract_state = nnx.split(abstract_mlp)
    checkpointer = ocp.StandardCheckpointer()
    restored_model = checkpointer.restore(full_path, abstract_state)

    q = nnx.merge(graphdef, restored_model)

    return q

def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy

# Used to map colors to integers
# COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

env = make_ocean_env(color = "red", render_mode = "human")

benchmark, grid_size = "oceans", "medium"
model_seed = 3
ckpt_name = "minigrid_oceans_medium_DDQN-UTS_1769513294.4912446_q_step_001930001_epoch_502057"
ckpt_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}"
)

print(ckpt_path)

q = get_q_net_from_checkpoint(ckpt_path, env)
policy = get_policy_from_q_net(q)


from eval_helper import eval_policy
def evaluate_policy_on_task(policy, task="all"):
    # COLOR_NAMES = ["blue"]
    if task == "all":
        task = COLOR_NAMES
    else:
        assert color in COLOR_NAMES
        print(f"task:{task}")
    
    for color in task:
        eval_env = make_ocean_env(color = color, render_mode = "human")
        _ = eval_policy(color, eval_env, policy, verbose=True, num_episode=1)

# evaluate_policy_on_task(policy)

# ---------------------------
# (2) Define masked network
# ---------------------------

import jax
import jax.numpy as jnp

def get_kernel_shapes(q):
    state = nnx.state(q)
    shapes = {}

    for layer_name, params in state.items():
        for pname, pvalue in params.items():
            if pname == "kernel":
                shapes[(layer_name, pname)] = pvalue.value.shape

    return shapes


def init_mask_logits(shapes, init_value=0.9):
    mask_logits = {}
    for key, shape in shapes.items():
        mask_logits[key] = jnp.full(shape, init_value)
    return mask_logits


def hard_concrete_sample(logits, tau, key, hard_threshold = 0.5):
    u1, u2 = jax.random.uniform(key, (2,))
    print(f"u1, u2={u1, u2}")
    noise = -jnp.log(jnp.log(u1)/jnp.log(u2))
    # noise = -jnp.log(jnp.log(u1)) - jnp.log(jnp.log(u2))
    s = jax.nn.sigmoid((logits - noise) / tau)
    print(f"noise={noise}, logits={logits}")
    
    hard = (s > hard_threshold).astype(s.dtype)
    
    # Straight-through estimator:
    # forward = hard, backward = gradient of s
    b = s + jax.lax.stop_gradient(hard - s)
    print(f"b={b}, s={s}")

    return b


def apply_mask_to_q(q, mask_logits, tau, key):
    state = nnx.state(q)
    new_state = {}

    for layer_name, params in state.items():
        new_state[layer_name] = {}

        for pname, pvalue in params.items():
            if pname == "kernel":
                key, subkey = jax.random.split(key)
                logits = mask_logits[(layer_name, pname)]
                mask = hard_concrete_sample(logits, tau, subkey)
                new_state[layer_name][pname] = pvalue.value * mask
            else:  # bias
                new_state[layer_name][pname] = pvalue

    masked_q_net = get_mlp_with_state(env, nnx.state(new_state))
    return masked_q_net


# 1. Get shapes
shapes = get_kernel_shapes(q)

# 2. Create trainable mask logits
mask_logits = init_mask_logits(shapes)

# 3. Use mask during forward pass
key, subkey = jax.random.split(key)
masked_q_net = apply_mask_to_q(q, mask_logits, tau=0.5, key=subkey)

# # 4. Masked q net
# # hard masked in forward pass, soft masked in back prop
# masked_policy = get_policy_from_q_net(masked_q_net)
# evaluate_policy_on_task(masked_policy)


# ---------- L0 regularization over mask logits ----------

def l0_regularization(mask_logits, lmbda):
    reg = 0.0
    for logits in mask_logits.values():
        reg += jnp.sum(jax.nn.sigmoid(logits))
    return lmbda * reg

# ---------- Simple rollout + replay buffer utilities ----------

import warnings
from tqdm import tqdm
from rl_blox.blox.replay_buffer import ReplayBuffer

def collect_task_data(rb, num_steps, q, task):
    env = make_ocean_env(color = task, render_mode = "None")
    obs, _ = env.reset()
    
    # TODO: try to sample state directly
    obs, _ = env.reset()
    for _ in tqdm(range(num_steps), desc="Experience collection"):
        #action = env.action_space.sample()
        action = int(jnp.argmax(q([obs])))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=terminated or truncated,
        )
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

# Initialise the replay buffer
hparams_algorithm = dict(
    buffer_size=50_000,
    total_timesteps=50_000,
    seed=seed,
)

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
num_steps = hparams_algorithm.pop("total_timesteps")

# Load existing replay buffer
import pickle

with open("rb_ckpt.pkl", "rb") as f:
    rb = pickle.load(f)
    print("Number of transitions:", len(rb))

if len(rb) < rb.buffer_size :
    # Collect experience with trained policy
    collect_task_data(rb, num_steps, q, task="red")


# # Save experience in replay buffer
save_rb = False
if save_rb:
    import pickle
    with open("rb_ckpt.pkl", "wb") as f:
        pickle.dump(rb, f)
    print("Saved replay buffer checkpoint.")

# # Test sample
# import numpy as np
# rng = np.random.default_rng(seed)

# num_train_steps=10000
# batch_size=64
# for step in range(num_train_steps):
#     batch = rb.sample_batch(batch_size, rng)
#     print(f"batch={batch}")
