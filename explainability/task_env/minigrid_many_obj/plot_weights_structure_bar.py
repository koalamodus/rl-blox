import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import ConnectionPatch

# palette = sns.color_palette("pastel")

FONT_SIZE = 14
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
})

percentage = 0.01

# make figure and assign axis objects
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 6))

# =========================================================
# experiment values
# =========================================================
# experiment = "minigrid"
# unused_weights_ratio = 12.82 * percentage
# globally_shared_weights_ratio = 96.84 * percentage
# task_specific_weights_ratio = 1.58 * percentage
# partially_shared_weights_ratio = 1 - globally_shared_weights_ratio - task_specific_weights_ratio
# context_variable_weights_ratio = 81.25 * percentage

experiment = "holoocean"
unused_weights_ratio = 33.37 * percentage
globally_shared_weights_ratio = 98.23 * percentage
task_specific_weights_ratio = 1.45 * percentage
partially_shared_weights_ratio = 1 - globally_shared_weights_ratio - task_specific_weights_ratio
context_variable_weights_ratio = 85.46 * percentage

# =========================================================
# LEFT: unused weights and others
# =========================================================
bar_ratios_left = [1-unused_weights_ratio, unused_weights_ratio]
bar_labels_left = ['used', 'unused']

bottom = 1
width = .75

# stack from top to bottom (matches original example style)
for j, (height, label) in enumerate(reversed([*zip(bar_ratios_left, bar_labels_left)])):
    bottom -= height

    bc = ax1.bar(
        0,
        height,
        width,
        bottom=bottom,
        label=label,
        # color=palette[j]
        alpha=0.1 + 0.25 * j
    )

    ax1.bar_label(
        bc,
        labels=[f"{height:.2%}"],
        label_type='center'
    )

ax1.set_title('all weights')
ax1.legend()
ax1.axis('off')

# =========================================================
# MIDDLE: 
# =========================================================
bar_ratios_middle = [globally_shared_weights_ratio, partially_shared_weights_ratio, task_specific_weights_ratio]
bar_labels_middle = ['globally shared', 'partially shared', 'task-specific']

bottom = 1

for j, (height, label) in enumerate(reversed([*zip(bar_ratios_middle, bar_labels_middle)])):
    bottom -= height

    bc = ax2.bar(
        0,
        height,
        width,
        bottom=bottom,
        color='C0',
        label=label,
        # color=palette[j]
        alpha=0.1 + 0.25 * j
    )

    ax2.bar_label(
        bc,
        labels=[f"{height:.2%}"] if label != 'partially shared' else [''],
        label_type='center'
    )

ax2.set_title('used weights')
ax2.legend()
ax2.axis('off')

# =========================================================
# RIGHT: task-specific weights
# =========================================================
bar_ratios_right = [context_variable_weights_ratio, 1-context_variable_weights_ratio]
bar_labels_right = ['context var', 'others']

bottom = 1

# stack from top to bottom (matches original example style)
for j, (height, label) in enumerate(reversed([*zip(bar_ratios_right, bar_labels_right)])):
    bottom -= height

    bc = ax3.bar(
        0,
        height,
        width,
        bottom=bottom,
        label=label,
        # color=palette[j]
        alpha=0.1 + 0.25 * j
    )

    ax3.bar_label(
        bc,
        labels=[f"{height:.2%}"],
        label_type='center'
    )

ax3.set_title('task-specific weights')
ax3.legend()
ax3.axis('off')


# =========================================================
# Connection lines
# =========================================================

# keep both axes on the same scale
ax1.set_xlim(-1, 1)
ax2.set_xlim(-1, 1)
ax3.set_xlim(-1, 1)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)

# vertical span of the connected group in the stacked bar
group_bottom_left = 0
group_top_left = bar_ratios_left[0]

# left top connecting line
con_top_left = ConnectionPatch(
    xyA=(-width / 2, 1),
    coordsA=ax2.transData,
    xyB=(width / 2, group_top_left),
    coordsB=ax1.transData,
    color='black',
    linewidth=1
)

# left bottom connecting line
con_bottom_left = ConnectionPatch(
    xyA=(-width / 2, 0),
    coordsA=ax2.transData,
    xyB=(width / 2, group_bottom_left),
    coordsB=ax1.transData,
    color='black',
    linewidth=1
)

fig.add_artist(con_top_left)
fig.add_artist(con_bottom_left)

# vertical span of the connected group in the stacked bar
group_bottom_right = 1-bar_ratios_middle[2]
group_top_right = 1

# right top connecting line
con_top_right = ConnectionPatch(
    xyA=(-width / 2, 1),
    coordsA=ax3.transData,
    xyB=(width / 2, group_top_right),
    coordsB=ax2.transData,
    color='black',
    linewidth=1
)

# right bottom connecting line
con_bottom_right = ConnectionPatch(
    xyA=(-width / 2, 0),
    coordsA=ax3.transData,
    xyB=(width / 2, group_bottom_right),
    coordsB=ax2.transData,
    color='black',
    linewidth=1
)

fig.add_artist(con_top_right)
fig.add_artist(con_bottom_right)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))

plt.subplots_adjust(left=0.05, right= 0.95, bottom=0.25, wspace=.5)
fig.suptitle(f"{experiment} network")
# ---------------------------
# Save
# ---------------------------
save_fig = False

if save_fig:
    import os
    plot_name = "weights_structure_bar"
    path = os.path.expanduser(f'~/Pictures/{experiment}_{plot_name}.pdf')
    plt.savefig(path)

plt.show()