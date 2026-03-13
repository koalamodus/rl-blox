import numpy as np
import random
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from minigrid.core.constants import COLOR_NAMES , COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv


class FindObjectEnv(MiniGridEnv):
    def __init__(self, target_color, obj_colors, **kwargs):
        # Validate obj_colors
        invalid_colors = [c for c in obj_colors if c not in COLOR_NAMES]
        if invalid_colors:
            raise ValueError(f"obj_colors contain invalid colors not in COLOR_NAMES: {invalid_colors}")

        # Validate target_color
        assert target_color in obj_colors or target_color is None, \
            f"target_color must be in obj_colors or None, got {target_color}"

        self.target_color = target_color
        self.obj_colors = obj_colors

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.obj_colors],
        )

        super().__init__(
            mission_space=mission_space,
            width=7,
            height=7,
            max_steps=20,
            agent_view_size=3,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(target_color: str):
        return f"find the {target_color} object"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Fixed agent position
        self.agent_pos = (1, 3)
        self.agent_dir = 0

        if self.target_color is None:
            raise ValueError("target_color must be set in order to assign target_pos and mission.")

        objects = {target_color: Ball(target_color) for target_color in self.obj_colors}

        
        # # Generate random positions inside (1,1) to (width-2, height-2)
        # available_positions = [
        #     (x, y) for x in range(1, width-1) for y in range(1, height-1)
        # ]
        # # Remove agent starting position so objects don’t spawn there
        # available_positions.remove(self.agent_pos)
        # # Randomly pick positions without replacement
        # positions = random.sample(available_positions, len(objects))

        positions = [
            (width - 4, 2),
            (width - 2, 3),
            (width - 4, 4),
            # (width - 2, 7),
        ]
        random.shuffle(positions)

        for ball_obj, pos in zip(objects.values(), positions):
            i, j = pos
            self.put_obj(ball_obj, i, j)
            ball_obj.cur_pos = (i, j)
        
        # Assign target
        self.target_pos = objects[self.target_color].cur_pos
        self.mission = f"find the {self.target_color} object"


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.target_pos is None:
            print("self.target_pos is None")
            return obs, reward, terminated, truncated, info

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        agent_facing_object = (
            (ax == (tx - 1) and ay == ty and self.agent_dir == 0)
            or (ax == (tx + 1) and ay == ty and self.agent_dir == 2)
            or (ax == tx and ay == (ty - 1) and self.agent_dir == 1)
            or (ax == tx and ay == (ty + 1) and self.agent_dir == 3)
        )

        if agent_facing_object:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info


class FlatContextObsWrapper(ObservationWrapper):
    """
    Flatten image observation and append the task context as a single index
    using COLOR_TO_IDX instead of one-hot encoding.
    """

    def __init__(self, env, obj_colors, target_color=None):
        super().__init__(env)

        assert target_color is None or target_color in obj_colors

        self.obj_colors = obj_colors
        self.fixed_task_color = target_color
        self.current_task_color = target_color

        img_height, img_width, num_channels = env.observation_space.spaces["image"].shape
        imgSize = img_height * img_width * (num_channels - 1)

        # Now context is a single uint8 representing the color index
        self.observation_space = spaces.Box(
            low=0,
            high=7,  # max color index in MiniGrid
            shape=(imgSize + 1,),  # append 1 integer instead of one-hot
            dtype="uint8",
        )

        self.action_space = spaces.Discrete(3)

    def reset(self, **kwargs):
        # Choose task color
        if self.fixed_task_color is None:
            self.current_task_color = random.choice(self.obj_colors)
            self.env.target_color = self.current_task_color
        else:
            self.current_task_color = self.fixed_task_color

        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        image = obs["image"][:, :, :2]

        # Replace one-hot with COLOR_TO_IDX
        color_idx = COLOR_TO_IDX[self.current_task_color]

        return np.append(image.flatten(), color_idx)


def make_ocean_env(target_color=None, obj_colors=None, render_mode=None):
    if obj_colors is None:
        obj_colors = list(COLOR_NAMES)

    env = FlatContextObsWrapper(
        FindObjectEnv(
            target_color=target_color if target_color is not None else random.choice(obj_colors),
            obj_colors=obj_colors,
            render_mode=render_mode,
        ),
        obj_colors=obj_colors,
        target_color=target_color,
    )
    return env