import os
from find_many_objects_env import make_ocean_env
from minigrid.core.constants import COLOR_NAMES
import flax.nnx as nnx
import jax.numpy as jnp
import jax.random as jr

seed = 49  # random seed for np and jax
key = jr.PRNGKey(seed)
OBJ_COLORS = COLOR_NAMES

# ---------------------------
# Load trained or pruned network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt from pre-trained full q-net

experiment = "reefshield_random_medium"
randomize_env = True
seed_num = 0
file_name = "_minigrid_reefshield_random_medium_DDQN-UTS_1773787402.9864454_q_step_005000000_epoch_5000000"
model_hidden_nodes = [32, 32]

# experiment = "reefshield_fixed_medium"
# randomize_env = False
# seed_num = 0
# file_name = "_minigrid_reefshield_fixed_medium_DDQN-UTS_1773675569.892775_q_step_001000000_epoch_1000000"
# model_hidden_nodes = [32, 32]

# experiment = "XRL_MINIGRID_POLICY"
# randomize_env = False
# seed_num = 48
# file_name = "minigrid_reefshield_medium_DDQN-UTS_1773420833.4308155_q_step_001000000_epoch_1000000"
# model_hidden_nodes = [128, 128]

# Set up ckpt path
ckpt_full_net = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)

# Set up environment to get input/output shapes
subtask = "purple"
env = make_ocean_env(subtask)

step = int(file_name.split("_")[-3])
ckpt_subnet = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
)

# choose ckpt path
task, ckpt_path = f"Pretrained network", ckpt_full_net
task, ckpt_path = f"Subnetwork {subtask}", ckpt_subnet

plot_info = f"{experiment}, seed {seed_num}, step {step}"

# Recreate MLP with same architecture as training
hparams_model = dict(
    n_features=env.observation_space.shape[0],
    n_outputs=int(env.action_space.n),
    activation="relu",
    hidden_nodes=model_hidden_nodes,
)

# Restore q net from checkpoint
abstract_mlp = MLP(rngs=nnx.Rngs(seed), **hparams_model)
graphdef, abstract_state = nnx.split(abstract_mlp)


checkpointer = ocp.StandardCheckpointer()
restored_model = checkpointer.restore(ckpt_path, abstract_state)
q_net = nnx.merge(graphdef, restored_model)

print("Loaded model checkpoint.")

# ---------------------------
# Evaluate subnetworks
# ---------------------------

eval_num_episode = 100 if randomize_env else 1

from eval_helper import eval_policy
def evaluate_policy_on_task(policy, task=None, obj_colors=OBJ_COLORS, randomize=randomize_env, num_episode=eval_num_episode, render_mode=None, seed=42, verbose=True):
    if task == None:
        task = obj_colors
    else:
        task = task if isinstance(task, list) else [task]
    
    all_task_scores = {}  # store results for all colors

    for target_color in task:
        assert target_color in COLOR_NAMES

        eval_env = make_ocean_env(target_color, randomize, render_mode)
        task_score_info = eval_policy(target_color, eval_env, policy, verbose=verbose, num_episode=num_episode, seed=seed)
        eval_env.close()

        all_task_scores.update(task_score_info)
    return all_task_scores

print("evaluate ckpt network")

def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy


policy = get_policy_from_q_net(q_net)
q_eval_scores = evaluate_policy_on_task(policy, num_episode=eval_num_episode, seed=seed)