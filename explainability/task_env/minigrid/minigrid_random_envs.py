import numpy as np
import random
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

# OBJECT_TO_IDX = {
#     "unseen": 0,
#     "empty": 1,
#     "wall": 2,
#     "floor": 3,
#     "door": 4,
#     "key": 5,
#     "ball": 6,
#     "box": 7,
#     "goal": 8,
#     "lava": 9,
#     "agent": 10,
# }

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
            max_steps=100,
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

        # list of the fixed positions you want to choose from
        positions = [
            (width - 2, 1),
            (width - 2, 3),
            (width - 2, 5),
            (width - 2, 7),
            (width - 2, 9),
            (width - 2, 11),
        ]

        # randomly shuffle the list of positions
        random.shuffle(positions)  # rearranges positions in-place

        # place each ball at one shuffled position
        for ball_obj, pos in zip(objects.values(), positions):
            i, j = pos
            self.put_obj(ball_obj, i, j)
            ball_obj.cur_pos = (i, j)  # make sure Ball knows its position
        
        # # random agent pos and dir
        # self.place_agent()

        # fixed agent pos and dir
        self.agent_pos = (1, 6)
        self.agent_dir = 0

        self.target_pos = objects[self.color].cur_pos
        self.mission = f"find the {self.color} object"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # # TODO: get reward only when agent is facing the target
        # Get reward and terminate, if agent is at the above, below, left, right position to target object
        if ((abs(ay - ty) + abs(ax - tx)) == 1):
            reward = self._reward()
            terminated = True

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

        img_height, img_width, num_channels = env.observation_space.spaces["image"].shape
        imgSize = img_height * img_width * (num_channels - 1)

        self.observation_space = spaces.Box(
            low=0,
            high=7, # obj is 0-6, color is 0-5, highest value is not included
            shape=(imgSize + 6,),
            dtype="uint8",
        )

        self.action_space = spaces.Discrete(4)
    
    def observation(self, obs):
        image = obs["image"][:, :, :2] # (OBJECT_IDX, COLOR_IDX, STATE) -> (OBJECT_IDX, COLOR_IDX)

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