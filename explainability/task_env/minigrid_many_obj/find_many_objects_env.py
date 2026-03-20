import operator
from functools import reduce

import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv

class FindManyObjectsEnv(MiniGridEnv):
    def __init__(
        self,
        color,
        random_object_positions=False,
        **kwargs,
    ):
        assert color in COLOR_NAMES

        self.color = color
        self.random_object_positions = random_object_positions

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        super().__init__(
            mission_space=mission_space,
            width=21,
            height=21,
            max_steps=200,
            **kwargs,
        )

        self.action_space = spaces.Discrete(3)

    @staticmethod
    def _gen_mission(color: str):
        return f"find the {color} objects"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = (9, 11)
        self.agent_dir = 0
        self.prev_pos = (9, 11)

        objects = {
            "red": Ball("red"),
            "blue": Ball("blue"),
            "green": Ball("green"),
            "yellow": Ball("yellow"),
            "purple": Ball("purple"),
            "grey": Ball("grey"),
        }

        if not self.random_object_positions:
            self.red_coords = [(3, 1), (2, 1), (1, 2), (5, 4), (6, 11)]
            self.blue_coords = [(11, 1), (11, 2), (12, 1), (13, 6), (11, 10)]
            self.green_coords = [(19, 1), (19, 3), (19, 2), (17, 5), (13, 8)]
            self.yellow_coords = [(1, 19), (2, 19), (4, 19), (6, 16), (5, 15)]
            self.purple_coords = [(13, 18), (12, 19), (11, 18), (10, 19), (9, 18)]
            self.grey_coords = [(18, 19), (19, 19), (19, 18), (12, 12), (15, 16)]
        else:
            has_overlaps = True
            while has_overlaps:
                self.red_coords = np.random.randint([1, 1], [11, 10], [5, 2])
                self.blue_coords = np.random.randint([5, 1], [14, 10], [5, 2])
                self.green_coords = np.random.randint([11, 1], [19, 10], [5, 2])
                self.yellow_coords = np.random.randint([1, 12], [11, 19], [5, 2])
                self.purple_coords = np.random.randint([5, 12], [15, 19], [5, 2])
                self.grey_coords = np.random.randint([11, 12], [19, 19], [5, 2])

                all = np.concatenate(
                    [
                        self.red_coords,
                        self.blue_coords,
                        self.green_coords,
                        self.yellow_coords,
                        self.grey_coords,
                        self.purple_coords,
                        np.array(self.agent_pos)[:, np.newaxis].T,
                    ],
                    axis=0,
                )
                has_overlaps = np.unique(all, axis=0).shape[0] < 31

        for coords in self.red_coords:
            self.put_obj(objects["red"], *coords)

        for coords in self.blue_coords:
            self.put_obj(objects["blue"], *coords)

        for coords in self.green_coords:
            self.put_obj(objects["green"], *coords)

        for coords in self.yellow_coords:
            self.put_obj(objects["yellow"], *coords)

        for coords in self.purple_coords:
            self.put_obj(objects["purple"], *coords)

        for coords in self.grey_coords:
            self.put_obj(objects["grey"], *coords)

        match self.color:
            case "red":
                self.target_coords = self.red_coords
            case "blue":
                self.target_coords = self.blue_coords
            case "green":
                self.target_coords = self.green_coords
            case "yellow":
                self.target_coords = self.yellow_coords
            case "purple":
                self.target_coords = self.purple_coords
            case "grey":
                self.target_coords = self.grey_coords

        self.mission = f"find the {self.color} objects"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos

        if action == self.actions.forward and self.agent_pos == self.prev_pos:
            reward = -1
            return obs, reward, terminated, truncated, info

        self.prev_pos = self.agent_pos

        unfound_targets = []

        for coords in self.target_coords:
            if self.agent_is_next_to_coord(*coords):
                reward = 1
                self.grid.set(coords[0], coords[1], None)
            else:
                unfound_targets.append(coords)

        self.target_coords = unfound_targets
        if len(self.target_coords) == 0:
            reward = 5
            terminated = True

        if reward == 0:
            reward = -0.1

        return obs, reward, terminated, truncated, info

    def agent_is_next_to_coord(self, x, y) -> bool:
        ax, ay = self.agent_pos

        # Reward for reaching a target object
        return (ax == x and abs(ay - y) == 1) or (ay == y and abs(ax - x) == 1)


class FlatContextObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these
    have their own language component.

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

    def __init__(self, env, context_in_obs=True):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        self.context_in_obs = context_in_obs

        if context_in_obs:
            self.observation_space = spaces.Box(
                low=0,
                high=20,
                shape=(int(2 * imgSize / 3) + 12,),
                dtype="uint8",
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=20,
                shape=(int(2 * imgSize / 3) + 6,),
                dtype="uint8",
            )

    def observation(self, obs):
        image = np.asarray(obs["image"][:, :, 0:2].flatten())

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

        direction = obs["direction"]
        one_hot_dir = np.zeros(4)
        one_hot_dir[direction] = 1

        pos = np.zeros(2)
        pos[0] = self.unwrapped.agent_pos[0]
        pos[1] = self.unwrapped.agent_pos[1]

        if self.context_in_obs:
            one_hot_context = np.zeros(6)
            one_hot_context[context] = 1

            obs = np.concatenate([image, pos, one_hot_dir, one_hot_context])
        else:
            obs = np.concatenate([image, pos, one_hot_dir])

        return obs

def make_ocean_env(color: str = "None", randomize = True ,render_mode = None):    
    env = FlatContextObsWrapper(
        FindManyObjectsEnv(
            color,
            random_object_positions=randomize,
            render_mode=render_mode,
        )
    )
    return env