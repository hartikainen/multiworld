from collections import OrderedDict
import logging

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
    get_shortest_distances,
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

    def action_np(self, observation, noisy=True):
        assert observation.shape == (2,)
        goal = self.goal

        if np.all(np.abs(goal - observation) < 1.0):
            action = goal - observation
            return action[None]

        round_observation = tuple(np.round(observation))
        round_goal = tuple(np.round(goal))
        shortest_path = self.all_pairs_shortest_paths[
            round_observation][round_goal]

        next_step = shortest_path[1]

        action = next_step - observation

        if noisy:
            action += np.random.uniform(-1, 1, 2)

        return action

    def actions_np(self, observations, noisy=True):
        actions = np.array([
            self.action_np(observation, noisy=noisy)
            for observation in observations
        ])

        return actions

    def true_distance(self, observation, goal):

        round_observation = tuple(np.round(observation))
        round_goal = tuple(np.round(goal))

        shortest_path = self.all_pairs_shortest_paths[
            round_observation][round_goal]

        return [len(shortest_path) - 1.0]

    def true_distances(self, observations, goals):
        true_distances = np.array([
            self.true_distance(observation, goal)
            for observation, goal in zip(observations, goals)
        ])

        return true_distances

    def reset(self):
        pass


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    GOAL_INDEX = slice(0, 2)

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=True,
            render_size=400,
            reward_type='dense',
            action_scale=1.0,
            target_radius=0.5,
            ball_radius=0.0,
            walls=None,
            observation_bounds=((-5, -5), (5, 5)),
            action_bounds=((-1, -1), (1, 1)),
            fixed_goal=None,
            reset_positions=None,
            images_are_rgb=False,  # else black and white
            discretize=False,
            terminate_on_success=False,
            show_goal=True,
            **kwargs
    ):
        if walls is None:
            walls = ()
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if len(kwargs) > 0:
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.succeeded_this_episode = False

        self.render_dt_msec = render_dt_msec
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.ball_radius = ball_radius
        self.terminate_on_success = terminate_on_success

        self.walls = walls

        dtype = 'int64' if discretize else 'float32'
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
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

        self.fixed_goal = (
            np.array(fixed_goal, dtype=dtype)
            if fixed_goal is not None
            else None)
        self.set_goal(self.sample_metric_goal(), dtype=dtype)
        self.ultimate_goal = self.fixed_goal

        self.drawer = None
        self.render_drawer = None

        self.initialize_grid_graph()

        self.optimal_policy = OptimalPoint2DEnvPolicy(
            goal=self._current_goal,
            graph=self.grid_graph,
            all_pairs_shortest_paths=self.all_pairs_shortest_paths)

    @property
    def _current_goal(self):
        return self._target_position

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
        self.all_pairs_observations, self.all_pairs_shortest_distances = (
            get_shortest_distances(self.all_pairs_shortest_paths))

    def numeric_observations(self, observations):
        return observations['state_observation']

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
        self.succeeded_this_episode |= is_success

        observation = self._get_obs()
        reward = self.compute_reward(action, observation)

        if self.discretize:
            try:
                assert issubclass(
                    self.numeric_observations(observation).dtype.type,
                    np.integer)
                assert issubclass(action.dtype.type, np.integer)
            except Exception as e:
                from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
                pass

        assert not self._position_inside_wall(
            self.numeric_observations(observation))

        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': action,
            'speed': np.linalg.norm(action),
            'is_success': is_success,
            'succeeded_this_episode': self.succeeded_this_episode,
        }
        done = is_success and self.terminate_on_success
        return observation, np.asscalar(reward), done, info

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
            positions = self._sample_realistic_positions(N)
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
        self.succeeded_this_episode = False
        self._current_position = self.get_reset_position()
        self._target_position = self.sample_goal()['state_observation']

        if self.discretize:
            assert issubclass(self._target_position.dtype.type, np.integer)
            assert issubclass(self._current_position.dtype.type, np.integer)

        return self._get_obs()

    def _positions_inside_wall(self, positions):
        if not self.walls:
            return np.zeros(positions.shape[0], dtype=np.bool)

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
        d = np.linalg.norm(
            achieved_goals - desired_goals,
            axis=-1,
            keepdims=True
        )

        if self.reward_type == "sparse":
            reward = -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            reward = -d
        elif self.reward_type == 'vectorized_dense':
            reward = -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

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

    def _sample_realistic_positions(self, N=1):
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

    def sample_metric_goal(self):
        positions = self._sample_realistic_positions(1)
        return {
            'state_observation': positions[0],
            'state_desired_goal': positions[0],
        }

    def sample_goals(self, N=1):
        if self.fixed_goal is None:
            goals = self._sample_realistic_positions(N)
        else:
            goals = np.repeat(self.fixed_goal[None], N, axis=0)

        goals_inside_walls = self._positions_inside_wall(goals)
        assert np.all(~goals_inside_walls), (goals, self.walls)

        return {'state_observation': goals}

    def sample_goal(self):
        samples = self.sample_goals(N=1)
        sample = {key: value[0, ...] for key, value in samples.items()}
        return sample

    def set_position(self, pos):
        assert self._current_position.shape == pos.shape
        self._current_position[:2] = pos[:2]

    def set_goal(self, goal, dtype=np.float32):
        self.fixed_goal = goal['state_observation']

        if self.fixed_goal is not None and self.fixed_goal.size > 2:
            from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
            pass

        if hasattr(self, 'optimal_policy'):
            self.optimal_policy.set_goal(self.fixed_goal)

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self.drawer is None:
            if width != height:
                raise NotImplementedError()
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
        self.draw(self.drawer, False)
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def position_to_observation(self, positions):
        return {'state_observation': positions}

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

    def draw(self, drawer, tick):
        # if self.drawer is not None:
        #     self.drawer.fill(Color('white'))
        # if self.render_drawer is not None:
        #     self.render_drawer.fill(Color('white'))
        drawer.fill(Color('white'))
        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        drawer.draw_solid_circle(
            self._current_position,
            np.maximum(self.ball_radius, 0.5),
            Color('blue'),
        )

        for wall in self.walls:
            drawer.draw_segment(
                wall.endpoint1,
                wall.endpoint2,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint2,
                wall.endpoint3,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint3,
                wall.endpoint4,
                Color('black'),
            )
            drawer.draw_segment(
                wall.endpoint4,
                wall.endpoint1,
                Color('black'),
            )

        drawer.render()
        if tick:
            drawer.tick(self.render_dt_msec)

    def render(self, mode='human', close=False):
        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                screen_width=self.render_size,
                screen_height=self.render_size,
                x_bounds=(
                    self.observation_box.low[0],
                    self.observation_box.high[0]),
                y_bounds=(
                    self.observation_box.low[1],
                    self.observation_box.high[1]),
                render_onscreen=True,
            )
            # self.render_drawer = PygameViewer(
            #     self.render_size,
            #     self.render_size,
            #     x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
            #     y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
            #     render_onscreen=True,
            # )
        self.draw(self.render_drawer, True)

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
            return self.get_approximate_shortest_paths(
                np.round(states1), np.round(states2))

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
            ball_radius=0.0,
            observation_bounds=((-5, -5), (5, 5)),
            wall_shape="zigzag",
            inner_wall_max_dist=2,
            thickness=1.0,
            wall_thickness=1.0,
            **kwargs,
    ):

        self.quick_init(locals())
        self.ball_radius = ball_radius
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape

        observation_bounds = np.array(observation_bounds)
        x_low, x_high = observation_bounds[:, 0]
        y_low, y_high = observation_bounds[:, 1]

        if wall_shape == 'zigzag':
            walls = (
                # Top wall
                HorizontalWall(
                    self.ball_radius,
                    (2/5) * y_high,
                    # x_low * 0.6,
                    x_low * 0.4,
                    # 0.9 below s.t. the wall blocks the edges of the env
                    x_high - thickness * 0.9,
                    thickness=thickness,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    (2/5) * y_low,
                    # 0.9 below s.t. the wall blocks the edges of the env
                    x_low + thickness * 0.9,
                    x_high * 0.4,
                    # x_high * 0.6,
                    thickness=thickness,
                ),
            )

        super().__init__(
            ball_radius=ball_radius,
            observation_bounds=observation_bounds,
            walls=walls,
            **kwargs)


def Point2DImageWallEnv(imsize=64, *args, **kwargs):
    env = Point2DWallEnv(*args, **kwargs)

    return ImageEnv(
        wrapped_env=env,
        imsize=env.render_size,
        transpose=True,
    )


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    # e = Point2DWallEnv("-", render_size=84)
    e = ImageEnv(Point2DWallEnv(wall_shape="u", render_size=200))
    for i in range(10):
        e.reset()
        for j in range(5):
            e.step(np.random.rand(2))
            e.render()
            im = e.get_image()
