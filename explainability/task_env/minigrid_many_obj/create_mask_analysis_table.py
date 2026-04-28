import os
from find_many_objects_env import make_ocean_env
from minigrid.core.constants import COLOR_TO_IDX, COLOR_NAMES
import flax.nnx as nnx
import numpy as np

seed = 49  # random seed for np and jax
plot_mask_only = True

# ---------------------------
# (1) Load trained network
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
step = int(file_name.split("_")[-3])
plot_info = f"{experiment}, seed {seed_num}, step {step}"

# Set up environment to get input/output shapes
subtask = "purple"
env = make_ocean_env(subtask)

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

#--------------

subtasks = ["purple", "blue", "grey", "red"]
# subtasks = COLOR_NAMES

weights_dict = {}

for subtask in subtasks:
    # env = make_ocean_env(subtask)
    ckpt_subnet = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
    )
    restored_model = checkpointer.restore(ckpt_subnet, abstract_state)
    q_net_sub = nnx.merge(graphdef, restored_model)
    print("Loaded subnetwork {subtask} checkpoint.")

    # Collect weight matrices (hidden + output)
    weights = []
    for layer in q_net_sub.hidden_layers:
        weights.append(np.array(layer.kernel.value))
    weights.append(np.array(q_net_sub.output_layer.kernel.value))

    weights_dict[subtask] = weights

print("Weights collected for all subtasks.")


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------
# Inputs
# ---------------------------
subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])


import numpy as np
from collections import defaultdict

num_layers = len(weights_dict[subtask_list[0]])

net_stats = {
    "total": {},
    "zero": {},
    "non_zero": {}
}

for subtask in subtask_list:

    total = 0
    non_zero = 0
    zero = 0

    for layer in weights_dict[subtask]:

        arr = np.array(layer)

        total += arr.size
        non_zero += np.count_nonzero(arr)
        zero += arr.size - np.count_nonzero(arr)

    net_stats["total"][subtask] = total
    net_stats["non_zero"][subtask] = non_zero
    net_stats["zero"][subtask] = zero

print("\n===== WEIGHT COUNTS PER NETWORK =====\n")

for subtask in subtask_list:
    print(f"Network: {subtask}")
    print(f"  Total entries   : {net_stats['total'][subtask]}")
    print(f"  Non-zero weights: {net_stats['non_zero'][subtask]}")
    print(f"  Zero weights    : {net_stats['zero'][subtask]}")
    print()

# ---------------------------
# Global counters
# ---------------------------
global_stats = {
    "total_per_net": defaultdict(int),
    "globally_shared": 0,
    "partially_shared": 0,
    "unused": 0,
    "task_specific": defaultdict(int),
}

# optional: per-layer breakdown
layer_global_stats = []

# ---------------------------
# Compute statistics
# ---------------------------
task_specific_masks_per_layer = []
for i in range(num_layers):
    layer_masks = {} 

    stacked = np.stack(
        [weights_dict[sub][i] for sub in subtask_list],
        axis=0
    )

    # presence mask
    used_any = (stacked != 0).any(axis=0)
    used_all = (stacked != 0).all(axis=0)
    unused = ~used_any

    # shared categories
    globally_shared_mask = used_all
    # TODO: this is wrong!
    partially_shared_mask = used_any & (~used_all)  # TODO: this is partially shared + task-specfic

    # task-specific per subtask
    task_specific_masks = {}

    for s_idx, subtask in enumerate(subtask_list):

        current = stacked[s_idx]
        used_current = (current != 0)

        others = np.delete(stacked, s_idx, axis=0)
        used_others = others.any(axis=0)

        exclusive_mask = used_current & (~used_others)
        task_specific_masks[subtask] = exclusive_mask
        layer_masks[subtask] = exclusive_mask

        global_stats["task_specific"][subtask] += np.sum(exclusive_mask)

        # total weights per net (any nonzero weight position)
        global_stats["total_per_net"][subtask] += np.sum(used_current)

    task_specific_masks_per_layer.append(layer_masks)
    # global aggregations (count unique positions once per layer)
    global_stats["globally_shared"] += np.sum(globally_shared_mask)
    global_stats["partially_shared"] += np.sum(partially_shared_mask)
    global_stats["unused"] += np.sum(unused)

    layer_global_stats.append({
        "globally_shared": np.sum(globally_shared_mask),
        "partially_shared": np.sum(partially_shared_mask),
        "unused": np.sum(unused),
    })

print("\n===== WEIGHT STATISTICS =====\n")

print("Total weights per network:")
for k, v in global_stats["total_per_net"].items():
    print(f"  {k}: {v}")

print("\nGlobally shared weights:")
print(f"  {global_stats['globally_shared']}")

print("\nPartially shared weights:")
print(f"  {global_stats['partially_shared']}")

print("\nTask-specific (exclusive) weights:")
for k, v in global_stats["task_specific"].items():
    print(f"  {k}: {v}")

print("\nUnused weights:")
print(f"  {global_stats['unused']}")

#
# Computation
#
# ----------------------------
# GLOBAL VALUES
# ----------------------------
T = net_stats["total"][subtask_list[0]]  # same for all networks
U = global_stats["unused"]
E = T - U

G = global_stats["globally_shared"]
P = global_stats["partially_shared"]

print("\n===== GLOBAL RATIOS =====\n")

print("Unused / Total")
print(U / T)

print("\nGlobally shared / (Total - unused)")
print(G / E)

print("\nPartially shared / (Total - unused)")
print(P / E)

print("\nTask-specific =  1 - Globally shared - Partially shared")
print(1 - (G+P) / E)


# ----------------------------
# PER-SUBTASK RATIOS
# ----------------------------
print("\n===== PER-SUBTASK RATIOS =====\n")
task_spec_sum = 0
for s in subtask_list:

    zero = net_stats["zero"][s]
    non_zero = net_stats["non_zero"][s]
    task_spec = global_stats["task_specific"][s]
    partial_task_specific = non_zero - G

    ratio_zero = (zero - U) / E
    ratio_partial_task_specific = partial_task_specific / E
    ratio_task_specific = task_spec / E
    ratio_task_specific_in_partial_task_specific = task_spec / partial_task_specific

    print(f"Network: {s}")
    print(f"  zero-adjusted: {ratio_zero}")
    print(f"  partial_task_specific_adjusted: {ratio_partial_task_specific}")
    print(f"  task_specific_adjusted: {ratio_task_specific}")
    print(f"  task-specific in partial_task_specific: {ratio_task_specific_in_partial_task_specific}")

    task_spec_sum += task_spec

# count task-specific weights in rows

rows_to_check = [-1, -3, -4, -6]

first_layer_masks = task_specific_masks_per_layer[0]
# task_specific_masks_per_layer[layer_idx][subtask]

print("\n===== TASK-SPECIFIC WEIGHTS IN ROWS [-1, -3, -4, -6] (FIRST LAYER) =====\n")
count_sum = 0
for subtask in subtask_list:

    mask = first_layer_masks[subtask]   # shape: (in_dim, out_dim)

    # select rows and count
    selected = mask[rows_to_check, :]
    count = np.sum(selected)

    # safe division
    ratio = count / partial_task_specific if partial_task_specific != 0 else 0.0
    
    print(f"{subtask}:")
    # print(f"  count (rows 2–5): {count}")
    # print(f"  partial_task_specific: {partial_task_specific}")
    print(f"  ratio: {ratio}")

    count_sum += count

ratio_context = count_sum / task_spec_sum
print(f"ratio of context variabes in all task specific variables: {ratio_context:.4f}")
print("\nTask-specific ratio = Task-specific / (Total - unused)")
print(task_spec_sum / E)
