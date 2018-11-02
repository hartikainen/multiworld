from collections import OrderedDict

import numpy as np
from gym import spaces
from pygame import Color

from multiworld.core.image_env import ImageEnv
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from multiworld.envs.pygame.pygame_viewer import PygameViewer
from multiworld.envs.pygame.walls import VerticalWall, HorizontalWall
from .point_2d_network import (
    grid_2d_graph_with_diagonal_edges,
    remove_walls_from_graph,
    get_shortest_paths,
)


class OptimalPoint2DEnvPolicy(object):
    def __init__(self,
                 goal,
                 graph,
                 all_pairs_shortest_paths):
        self.goal = tuple(goal) if goal is not None else (0.0, 0.0)
        self.graph = graph
        self.all_pairs_shortest_paths = all_pairs_shortest_paths

    def set_goal(self, goal):
        self.goal = tuple(goal)

    def get_action(self, observation):
        observation = np.array(observation)
        goal = self.goal
        if np.all(np.abs(goal - observation) < 1.0):
            action = goal - observation
            return ((action, None, None,), None)

        round_observation = tuple(np.round(observation))
        round_goal = tuple(np.round(goal))
        shortest_path = self.all_pairs_shortest_paths[
            round_observation][round_goal]

        next_step = shortest_path[1]

        action = next_step - observation

        return ((action, None, None,), None)

    def reset(self):
        pass


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            render_onscreen=True,
            render_size=400,
            reward_type='dense',
            target_radius=0.5,
            point_radius=0.0,
            walls=(),
            observation_bounds=((-5, -5), (5, 5)),
            action_bounds=((-1, -1), (1, 1)),
            fixed_goal=None,
            reset_positions=None,
            images_are_rgb=False,  # else black and white
            discretize=False):
        self.quick_init(locals())

        self.render_dt_msec = render_dt_msec
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.target_radius = target_radius
        self.point_radius = point_radius

        self.walls = walls

        dtype = 'int64' if discretize else 'float32'

        self.fixed_goal = (
            np.array(fixed_goal, dtype=dtype)
            if fixed_goal is not None
            else None)

        self.images_are_rgb = images_are_rgb
        self.discretize = discretize

        self._max_episode_steps = 50

        self._current_position = np.zeros(2, dtype=dtype)
        self._previous_action = np.zeros(2, dtype=dtype)

        self._reset_positions = (
            np.reshape(np.array(reset_positions, dtype=dtype), (-1, 2))
            if reset_positions is not None
            else None)
        self._target_position = None

        # lower_bound_offset fixes a problem where gym.spaces.Box.sample()
        # produces incorrect value
        action_bounds = np.array(action_bounds, dtype=dtype)
        self.action_x_bounds = action_bounds[:, 0]
        self.action_y_bounds = action_bounds[:, 1]
        assert np.abs(self.action_x_bounds[0]) == self.action_x_bounds[1]
        assert np.abs(self.action_y_bounds[0]) == self.action_y_bounds[1]
        self.action_space = spaces.Box(
            action_bounds[0, :],
            action_bounds[1, :],
            dtype=dtype)

        observation_bounds = np.array(observation_bounds, dtype=dtype)
        self.observation_x_bounds = observation_bounds[:, 0]
        self.observation_y_bounds = observation_bounds[:, 1]
        observation_box = spaces.Box(
            observation_bounds[0, :],
            observation_bounds[1, :],
            dtype=dtype)

        self.observation_box = observation_box

        self.observation_space = spaces.Dict({
            'observation': observation_box,
            'desired_goal': observation_box,
            'achieved_goal': observation_box,
            'state_observation': observation_box,
            'state_desired_goal': observation_box,
            'state_achieved_goal': observation_box,
        })

        self.drawer = None
        self.initialize_grid_graph()

        self.optimal_policy = OptimalPoint2DEnvPolicy(
            goal=self.fixed_goal,
            graph=self.grid_graph,
            all_pairs_shortest_paths=self.all_pairs_shortest_paths)

    def get_approximate_shortest_paths(self, starts, ends):
        optimal_distances = []
        for start, end in zip(starts, ends):
            start, end = tuple(start), tuple(end)
            optimal_distances.append(
                len(self.all_pairs_shortest_paths[start][end]))
        optimal_distances = np.array(optimal_distances)
        return optimal_distances

    def initialize_grid_graph(self):
        x_low, x_high = self.observation_x_bounds
        dx = self.action_x_bounds[1]
        y_low, y_high = self.observation_y_bounds
        dy = self.action_y_bounds[1]

        graph = grid_2d_graph_with_diagonal_edges(
            np.arange(x_low, x_high+1, dx),
            np.arange(y_low, y_high+1, dy))

        graph = remove_walls_from_graph(graph, self.walls)

        self.grid_graph = graph
        self.all_pairs_shortest_paths = get_shortest_paths(graph)

    def step(self, action):
        action = np.clip(
            action,
            a_min=self.action_space.low,
            a_max=self.action_space.high)
        self._previous_action = action

        new_position = self._current_position + action
        new_position = np.clip(
            new_position,
            a_min=self.observation_box.low,
            a_max=self.observation_box.high)
        self._current_position = self.handle_collision(
            self._current_position, new_position)

        distance_to_target = np.linalg.norm(
            self._current_position - self._target_position, ord=2)

        is_success = distance_to_target < self.target_radius

        observation = self._get_obs()
        reward = self.compute_reward(action, observation)

        if self.discretize:
            try:
                assert issubclass(observation['observation'].dtype.type, np.integer)
                assert issubclass(action.dtype.type, np.integer)
            except Exception as e:
                from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
                pass

        assert not self._position_inside_wall(observation['observation'])

        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': action,
            'speed': np.linalg.norm(action),
            'is_success': is_success,
        }
        done = False
        return observation, reward, done, info

    def handle_collisions(self, previous_positions, new_positions):
        new_positions = np.array([
            self.handle_collision(previous_position, new_position)
            for previous_position, new_position in zip(
                    previous_positions, new_positions)
        ])

        return new_positions

    def handle_collision(self, previous_position, new_position):
        for wall in self.walls:
            new_position = wall.handle_collision(
                previous_position, new_position)

        return new_position

    def get_reset_positions(self, N=1):
        if self._reset_positions is None:
            positions = self._sample_realistic_observations(N)
        else:
            positions = self._reset_positions[np.random.choice(
                self._reset_positions.shape[0], N)]

        positions_inside_walls = self._positions_inside_wall(positions)
        assert np.all(~positions_inside_walls), (positions, self.walls)

        return positions

    def get_reset_position(self):
        position = self.get_reset_positions(N=1)[0, ...]
        return position

    def reset(self):
        self._current_position = self.get_reset_position()
        self._target_position = self.sample_goal()['desired_goal']

        if self.discretize:
            assert issubclass(self._target_position.dtype.type, np.integer)
            assert issubclass(self._current_position.dtype.type, np.integer)

        return self._get_obs()

    def _positions_inside_wall(self, positions):
        inside_walls = [
            wall.contains_point(positions)
            for wall in self.walls
        ]
        inside_any_wall = np.any(inside_walls, axis=0)
        return inside_any_wall

    def _position_inside_wall(self, position):
        return self._positions_inside_wall(position[None])[0]

    def _get_obs(self):
        observation = {
            'observation': self._current_position.copy(),
            'desired_goal': self._target_position.copy(),
            'achieved_goal': self._current_position.copy(),
            'state_observation': self._current_position.copy(),
            'state_desired_goal': self._target_position.copy(),
            'state_achieved_goal': self._current_position.copy(),
        }

        return observation

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_observation']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)

        if self.reward_type == "sparse":
            reward = -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            reward = -d

        return reward

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def _sample_position(self, low, high, realistic=True):
        pos = np.random.uniform(low, high)
        if realistic:
            while self._current_position_inside_wall(pos) is True:
                pos = np.random.uniform(low, high)
        return pos

    def _sample_realistic_observations(self, N=1):
        positions = np.array([
            self.observation_box.sample()
            for _ in range(N)
        ])

        positions_inside_walls = self._positions_inside_wall(positions)

        while np.any(positions_inside_walls):
            # positions_inside_walls_idx = np.where(positions_inside_walls)
            num_positions_inside_walls = np.sum(positions_inside_walls)
            new_positions = np.array([
                self.observation_box.sample()
                for _ in range(num_positions_inside_walls)
            ])
            positions[positions_inside_walls] = new_positions
            positions_inside_walls = self._positions_inside_wall(
                positions)

        return positions

    def sample_goals(self, N=1):
        if self.fixed_goal is None:
            goals = self._sample_realistic_observations(N)
        else:
            goals = np.repeat(self.fixed_goal[None], N, axis=0)

        goals_inside_walls = self._positions_inside_wall(goals)
        assert np.all(~goals_inside_walls), (goals, self.walls)

        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def sample_goal(self):
        samples = self.sample_goals(N=1)
        sample = {key: value[0, ...] for key, value in samples.items()}
        return sample

    def set_position(self, pos):
        self._current_position[0] = pos[0]
        self._current_position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if width is not None:
            if width != height:
                raise NotImplementedError()
            if width != self.render_size:
                self.drawer = PygameViewer(
                    screen_width=width,
                    screen_height=height,
                    x_bounds=(
                        self.observation_box.low[0],
                        self.observation_box.high[0]),
                    y_bounds=(
                        self.observation_box.low[1],
                        self.observation_box.high[1]),
                    render_onscreen=self.render_onscreen,
                )
                self.render_size = width
        self.render()
        image = self.drawer.get_image()

        if self.images_are_rgb:
            image = image.transpose().flatten()
        else:
            r, b = image[:, :, 0], image[:, :, 2]
            image = (-r + b).flatten()

        return image

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._current_position = goal
        self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._current_position = position
        self._target_position = goal
        return self.get_env_state()

    def render(self, mode='human', close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=self.observation_x_bounds,
                y_bounds=self.observation_y_bounds,
                render_onscreen=self.render_onscreen,
            )

        self.drawer.fill(Color('white'))
        self.drawer.draw_solid_circle(
            self._target_position,
            np.max((self.target_radius, 0.05)) * 5,
            Color('green'),
        )
        self.drawer.draw_solid_circle(
            self._current_position,
            np.max((self.point_radius, 0.05)) * 5,
            Color('blue'),
        )

        for wall in self.walls:
            self.drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            self.drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )

        self.drawer.render()
        self.drawer.tick(self.render_dt_msec)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_optimal_paths(self, states1, states2):
        if self.walls:
            raise NotImplementedError()

        # if np.sum(states2) > 0.0:
        #     raise NotImplementedError()

        if (any(self.action_space.high != 1)
            or any(self.action_space.low != -1)):
            raise NotImplementedError()

        num_steps_to_goal = np.ceil(
            np.linalg.norm(states1 - states2, ord=float('inf'), axis=1)
        ).astype(int)

        optimal_paths = [
            np.concatenate([
                np.linspace(
                    states1[i, 0], states2[i, 0], num_steps_to_goal[i]
                )[:, None],
                np.linspace(
                    states1[i, 1], states2[i, 1], num_steps_to_goal[i]
                )[:, None],
            ], axis=1)
            for i in range(states1.shape[0])
        ]

        return optimal_paths

    def initialize_camera(self, init_fctn):
        pass


class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            point_radius=0.0,
            observation_bounds=((-5, -5), (5, 5)),
            wall_shape="zigzag",
            inner_wall_max_dist=2,
            thickness=1.0,
            **kwargs
    ):

        self.quick_init(locals())
        self.point_radius = point_radius
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape

        observation_bounds = np.array(observation_bounds)
        x_low, x_high = observation_bounds[:, 0]
        y_low, y_high = observation_bounds[:, 1]

        if wall_shape == 'zigzag':
            walls = (
                # Top wall
                HorizontalWall(
                    self.point_radius,
                    (2/5) * y_high,
                    x_low * 0.4,
                    # 0.9 below s.t. the wall blocks the edges of the env
                    x_high - thickness * 0.9,
                    thickness=thickness,
                ),
                # Bottom wall
                HorizontalWall(
                    self.point_radius,
                    (2/5) * y_low,
                    # 0.9 below s.t. the wall blocks the edges of the env
                    x_low + thickness * 0.9,
                    x_high * 0.4,
                    thickness=thickness,
                ),
            )

        super().__init__(
            point_radius=point_radius,
            observation_bounds=observation_bounds,
            walls=walls,
            **kwargs)


if __name__ == "__main__":
    # e = Point2DEnv()
    import matplotlib.pyplot as plt

    # e = Point2DWallEnv("-", render_size=84)
    e = ImageEnv(Point2DWallEnv(wall_shape="u", render_size=200))
    for i in range(10):
        e.reset()
        for j in range(50):
            e.step(np.random.rand(2))
            e.render()
            im = e.get_image()
