from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper


TASK_COLORS = ["red", "green", "yellow", "purple", "blue", "grey"]

# training env colors: "red", "green", "yellow", "purple"
# test env colors: "blue", "grey"

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
            width=21,
            height=18,
            max_steps=200,
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
        self.put_obj(objects["blue"], width - 2, 4)
        self.put_obj(objects["green"], width - 2, 7)
        self.put_obj(objects["yellow"], width - 2, 10)
        self.put_obj(objects["purple"], width - 2, 13)
        self.put_obj(objects["grey"], width - 2, 16)

        self.agent_pos = (1, 9)
        self.agent_dir = 0

        self.target_pos = objects[self.color].cur_pos
        self.mission = f"find the {self.color} object"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            terminated = True

        # Reward for performing the done action next to target object
        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info

def make_ocean_env(color: str = "None", render_mode = None):    
    env = FlatObsWrapper(
        ImgObsWrapper(
            FindObjectEnv(
                color,
                render_mode=render_mode,
            )
        )
    )
    return env