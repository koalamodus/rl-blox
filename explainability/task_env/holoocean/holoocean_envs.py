from typing import Optional

import gymnasium as gym
import holoocean
import numpy as np

scenario = {
    "name": "test",
    "package_name": "Ocean",
    "world": "OpenWater",
    "main_agent": "auv",
    "agents": [
        {
            "agent_name": "auv",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "LocationSensor",
                    "socket": "COM",
                },
                {
                    "sensor_type": "RotationSensor",
                    "socket": "COM",
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "IMUSocket",
                },
            ],
            "control_scheme": 0,
            "location": [-140, 125, -286],
            "rotation": [0, 0, 0],
        }
    ],
}


class HoloOceanEnv(gym.Env):
    def __init__(self, render_mode=None, context=None, context_in_obs=True):
        self.render_mode = render_mode
        self.sim = holoocean.make(
            scenario_cfg=scenario,
            show_viewport=render_mode == "human",
            ticks_per_sec=20,
        )
        self.context = context
        self.context_in_obs = context_in_obs
        self.action_space = gym.spaces.Discrete(5)
        min_array = np.array(
            [
                -10,
                -10,
                # -300,
                0,
                0,
                0,
                -10,
                -10,
                # -10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        max_array = np.array(
            [
                10,
                10,
                # 0,
                0,
                0,
                0,
                -10,
                -10,
                # -10,
                200,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                1,
                1,
                1,
                1,
            ]
        )

        if self.context_in_obs:
            max_array = np.concatenate([np.ones(6), max_array])
            min_array = np.concatenate([np.array([-1, -1, 0, 0, 0, 0]), min_array])

        self.observation_space = gym.spaces.Box(min_array, max_array)

        self._action_map = {
            0: np.array([0, 0, 0, 0, 50]),
            1: np.array([0, 0, 0, 0, 0]),
            2: np.array([0, 0, 0, 0, -50]),
            3: np.array([0, 45, 0, -45, 50]),
            4: np.array([0, -45, 0, 45, 50]),
            # 3: np.array([45, 0, -45, 0, 20]),
            # 4: np.array([-45, 0, 45, 0, 20]),
            # 5: np.array([0, 0, 0, 0, 0]),
        }

        self.currents = np.array([self.context[0], self.context[1], 0])
        self.oracle_distance = 7.0
        self.organism_depth = -291
        self.max_steps = 500
        self.step_length = 10
        self.start_pos = np.array([-140, 125, -286])

    def _get_obs(self, state):
        location = state["LocationSensor"] - self.start_pos
        rotation = state["RotationSensor"]
        velocity = state["VelocitySensor"]
        time = np.array([state["t"]])
        remaining_organisms = np.array(
            [
                self.remaining_red_organisms.shape[0] / 500.0,
                self.remaining_blue_organisms.shape[0] / 500.0,
                self.remaining_green_organisms.shape[0] / 500.0,
                self.remaining_black_organisms.shape[0] / 500.0,
            ]
        )

        local_organisms = np.array(
            [
                self.local_red,
                self.local_blue,
                self.local_green,
                self.local_black,
            ]
        )
        local_unseen = np.array(
            [
                self.local_red_unseen,
                self.local_blue_unseen,
                self.local_green_unseen,
                self.local_black_unseen,
            ]
        )
        observation = np.concatenate(
            [
                location[:2],
                rotation,
                velocity[:2],
                time,
                local_organisms.flatten(),
                local_unseen.flatten(),
                remaining_organisms,
            ]
        )
        if self.context is not None and self.context_in_obs:
            observation = np.concatenate([self.context, observation])
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.total_steps = 0
        self.currents = np.array([self.context[0], self.context[1], 0])
        state = self.sim.reset()

        rng = np.random.default_rng(seed=seed)
        self.red_organisms = rng.multivariate_normal(
            np.array([-140, 125, self.organism_depth]),
            np.array([[5, 0, 0], [0, 500, 0], [0, 0, 0]]),
            500,
        )
        self.green_organisms = rng.multivariate_normal(
            np.array([-140, 125, self.organism_depth]),
            np.array([[500, 0, 0], [0, 5, 0], [0, 0, 0]]),
            500,
        )
        self.blue_organisms = rng.multivariate_normal(
            np.array([-140, 125, self.organism_depth]),
            np.array([[252.5, -247.5, 0], [-247.5, 252.5, 0], [0, 0, 0]]),
            500,
        )

        self.black_organisms = rng.multivariate_normal(
            np.array([-140, 125, self.organism_depth]),
            np.array([[252.5, 247.5, 0], [247.5, 252.5, 0], [0, 0, 0]]),
            500,
        )

        self.remaining_red_organisms = self.red_organisms.copy()
        self.remaining_blue_organisms = self.blue_organisms.copy()
        self.remaining_green_organisms = self.green_organisms.copy()
        self.remaining_black_organisms = self.black_organisms.copy()

        self.local_black = np.array([0, 0, 0, 0])
        self.local_blue = np.array([0, 0, 0, 0])
        self.local_green = np.array([0, 0, 0, 0])
        self.local_red = np.array([0, 0, 0, 0])

        self.local_black_unseen = np.array([0, 0, 0, 0])
        self.local_blue_unseen = np.array([0, 0, 0, 0])
        self.local_green_unseen = np.array([0, 0, 0, 0])
        self.local_red_unseen = np.array([0, 0, 0, 0])

        if self.render_mode == "human":
            for organism in self.red_organisms:
                self.sim.draw_point(
                    [organism[0], organism[1], self.organism_depth],
                    color=[150, 0, 0],
                    thickness=10,
                    lifetime=0,
                )

            for organism in self.green_organisms:
                self.sim.draw_point(
                    [organism[0], organism[1], self.organism_depth],
                    color=[0, 150, 0],
                    thickness=10,
                    lifetime=0,
                )

            for organism in self.blue_organisms:
                self.sim.draw_point(
                    [organism[0], organism[1], self.organism_depth],
                    color=[0, 0, 150],
                    thickness=10,
                    lifetime=0,
                )

            for organism in self.black_organisms:
                self.sim.draw_point(
                    [organism[0], organism[1], self.organism_depth],
                    color=[50, 50, 50],
                    thickness=10,
                    lifetime=0,
                )

        observation = self._get_obs(state)
        return observation, {}

    def step(self, action):
        self.total_steps += 1

        command = self._action_map[action]

        total_found_red = 0
        total_found_black = 0
        total_found_blue = 0
        total_found_green = 0
        for _ in range(self.step_length):
            state = self.sim.step(command)
            self.sim.set_ocean_currents("auv", self.currents)
            red_dists = np.linalg.norm(
                self.remaining_red_organisms - state["LocationSensor"], axis=1
            )
            blue_dists = np.linalg.norm(
                self.remaining_blue_organisms - state["LocationSensor"], axis=1
            )
            black_dists = np.linalg.norm(
                self.remaining_black_organisms - state["LocationSensor"], axis=1
            )
            green_dists = np.linalg.norm(
                self.remaining_green_organisms - state["LocationSensor"], axis=1
            )

            found_new_red = np.argwhere(red_dists <= self.oracle_distance)
            self.remaining_red_organisms = np.delete(
                self.remaining_red_organisms, found_new_red, axis=0
            )
            total_found_red += len(found_new_red)

            found_new_blue = np.argwhere(blue_dists <= self.oracle_distance)
            self.remaining_blue_organisms = np.delete(
                self.remaining_blue_organisms, found_new_blue, axis=0
            )
            total_found_blue += len(found_new_blue)

            found_new_black = np.argwhere(black_dists <= self.oracle_distance)
            self.remaining_black_organisms = np.delete(
                self.remaining_black_organisms, found_new_black, axis=0
            )
            total_found_black += len(found_new_black)

            found_new_green = np.argwhere(green_dists <= self.oracle_distance)
            self.remaining_green_organisms = np.delete(
                self.remaining_green_organisms, found_new_green, axis=0
            )
            total_found_green += len(found_new_green)

            red_dists = np.linalg.norm(
                self.red_organisms - state["LocationSensor"], axis=1
            )
            blue_dists = np.linalg.norm(
                self.blue_organisms - state["LocationSensor"], axis=1
            )
            black_dists = np.linalg.norm(
                self.black_organisms - state["LocationSensor"], axis=1
            )
            green_dists = np.linalg.norm(
                self.green_organisms - state["LocationSensor"], axis=1
            )

            diffs = (self.red_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            red_q1 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            red_q2 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            red_q3 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            red_q4 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.remaining_red_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            red_q1_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            red_q2_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            red_q3_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            red_q4_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.green_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            green_q1 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            green_q2 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            green_q3 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            green_q4 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.remaining_green_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            green_q1_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            green_q2_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            green_q3_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            green_q4_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.blue_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            blue_q1 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            blue_q2 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            blue_q3 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            blue_q4 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.remaining_blue_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            blue_q1_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            blue_q2_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            blue_q3_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            blue_q4_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.black_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            black_q1 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            black_q2 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            black_q3 = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            black_q4 = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            diffs = (self.remaining_black_organisms - state["LocationSensor"])[:, :2]
            mask = (np.abs(diffs[:, 0]) <= self.oracle_distance) & (
                np.abs(diffs[:, 1]) <= self.oracle_distance
            )
            nearby = diffs[mask]
            black_q1_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] > 0))
            black_q2_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] > 0))
            black_q3_unseen = np.sum((nearby[:, 0] < 0) & (nearby[:, 1] < 0))
            black_q4_unseen = np.sum((nearby[:, 0] > 0) & (nearby[:, 1] < 0))

            self.local_red = np.array([red_q1, red_q2, red_q3, red_q4])
            self.local_red_unseen = np.array(
                [red_q1_unseen, red_q2_unseen, red_q3_unseen, red_q4_unseen]
            )
            self.local_blue = np.array([blue_q1, blue_q2, blue_q3, blue_q4])
            self.local_blue_unseen = np.array(
                [blue_q1_unseen, blue_q2_unseen, blue_q3_unseen, blue_q4_unseen]
            )
            self.local_green = np.array([green_q1, green_q2, green_q3, green_q4])
            self.local_green_unseen = np.array(
                [green_q1_unseen, green_q2_unseen, green_q3_unseen, green_q4_unseen]
            )
            self.local_black = np.array([black_q1, black_q2, black_q3, black_q4])
            self.local_black_unseen = np.array(
                [black_q1_unseen, black_q2_unseen, black_q3_unseen, black_q4_unseen]
            )

            if np.linalg.norm(state["LocationSensor"] - self.start_pos) >= 75:
                return self._get_obs(state), -1000, True, False, {}

        next_obs = self._get_obs(state)
        if np.sum(next_obs[-4:]) == 0:
            reward = 1000
            terminated = True
            truncated = False
        else:
            truncated = self.total_steps >= self.max_steps
            terminated = False
            total_found = np.array(
                [
                    total_found_red,
                    total_found_blue,
                    total_found_green,
                    total_found_black,
                ]
            )
            if self.context is None:
                reward = (
                    total_found_black
                    + total_found_green
                    + total_found_blue
                    + total_found_red
                ) / 4.0 - 0.01
            else:
                reward = (
                    np.dot(self.context[2:], total_found) / np.sum(self.context[2:])
                    - 0.01
                )

        return self._get_obs(state), reward, terminated, truncated, {}
