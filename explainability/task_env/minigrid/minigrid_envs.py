import operator
import numpy as np
from functools import reduce
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv

# Used to map colors to integers
# COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
# context = COLOR_TO_IDX[color]


class FindObjectEnv(MiniGridEnv):
    def __init__(
        self,
        color,
        **kwargs,
    ):
        assert color in COLOR_NAMES

        self.color = color

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        super().__init__(
            mission_space=mission_space,
            width=13,
            height=13,
            max_steps=500,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"find the {color} object"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        objects = {
            "red": Ball("red"),
            "blue": Ball("blue"),
            "green": Ball("green"),
            "yellow": Ball("yellow"),
            "purple": Ball("purple"),
            "grey": Ball("grey"),
        }

        self.put_obj(objects["red"], width - 2, 1)
        self.put_obj(objects["blue"], width - 2, 3)
        self.put_obj(objects["green"], width - 2, 5)
        self.put_obj(objects["yellow"], width - 2, 7)
        self.put_obj(objects["purple"], width - 2, 9)
        self.put_obj(objects["grey"], width - 2, 11)

        self.agent_pos = (1, 6)
        self.agent_dir = 0

        self.target_pos = objects[self.color].cur_pos
        self.mission = f"find the {self.color} object"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Reward for reaching the target object
        if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
            reward = self._reward()
            terminated = True
        # Reward for performing the done action next to target object and terminate episode
        # if action == self.actions.done:
        #     terminated = True

        return obs, reward, terminated, truncated, info

class FlatContextObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + 6,),
            dtype="uint8",
        )
    
    def observation(self, obs):
        image = obs["image"]

        if "red" in obs["mission"]:
            context = COLOR_TO_IDX["red"]
        elif "blue" in obs["mission"]:
            context = COLOR_TO_IDX["blue"]
        elif "green" in obs["mission"]:
            context = COLOR_TO_IDX["green"]
        elif "yellow" in obs["mission"]:
            context = COLOR_TO_IDX["yellow"]
        elif "purple" in obs["mission"]:
            context = COLOR_TO_IDX["purple"]
        elif "grey" in obs["mission"]:
            context = COLOR_TO_IDX["grey"]
        else:
            raise RuntimeError

        one_hot_context = np.zeros(6)
        one_hot_context[context] = 1

        obs = np.append(image.flatten(), one_hot_context)


        return obs

def make_ocean_env(color: str = "None", render_mode = None):    
    env = FlatContextObsWrapper(
        FindObjectEnv(
            color,
            render_mode=render_mode,
        )
    )
    return env