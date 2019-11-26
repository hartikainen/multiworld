from collections import OrderedDict
import glob
import itertools
import os
import re
import logging

import numpy as np
from gym import spaces
from pygame import Color
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

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


def plot_walls(axis, walls):
    wall_rectangles = []

    for wall in walls:
        top_right = wall.endpoint1
        bottom_right = wall.endpoint2
        bottom_left = wall.endpoint3
        top_left = wall.endpoint4

        width = max(top_right[0] - top_left[0], 0.1)
        height = max(top_right[1] - bottom_right[1], 0.1)
        wall_rectangle = mpl.patches.Rectangle(
            bottom_left,
            width,
            height,
            fill=True)

        wall_rectangles.append(wall_rectangle)

    wall_patch_collection = mpl.collections.PatchCollection(
        wall_rectangles,
        facecolor='black',
        edgecolor=None)

    axis.add_collection(wall_patch_collection)

    return wall_patch_collection, wall_rectangles


def plot_waters(axis, waters):
    water_rectangles = []

    for water in waters:
        bottom_left, top_right = water
        # top_right = water.endpoint1
        # bottom_right = water.endpoint2
        # bottom_left = water.endpoint3
        # top_left = water.endpoint4
        # top_left = (bottom_left[0], top_right[1])
        # bottom_right = (top_right[0], bottom_left[1])
        width, height = top_right - bottom_left

        # width = top_right[0] - top_left[0]
        # height = top_right[1] - bottom_right[1]
        water_rectangle = mpl.patches.Rectangle(
            bottom_left,
            width,
            height,
            fill=True)

        water_rectangles.append(water_rectangle)

    water_patch_collection = mpl.collections.PatchCollection(
        water_rectangles,
        facecolor='blue',
        edgecolor=None)

    axis.add_collection(water_patch_collection)

    return water_patch_collection, water_rectangles


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
        self.ultimate_goal = self.fixed_goal
        self.set_goal(self.sample_goal(), dtype=dtype)

        self.drawer = None
        self.render_drawer = None

        # self.initialize_grid_graph()

        # self.optimal_policy = OptimalPoint2DEnvPolicy(
        #     goal=self._current_goal,
        #     graph=self.grid_graph,
        #     all_pairs_shortest_paths=self.all_pairs_shortest_paths)

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

    def get_path_infos(self, paths, evaluation_type='training'):
        infos = {}

        if (getattr(self, 'wall_shape', None) != '-'
            and not isinstance(self, Point2DBridgeEnv)
            and not isinstance(self, Point2DPondEnv)):
            return infos

        if getattr(self, 'wall_shape', None) == '-':
            lefts_success, rights_success = 0, 0
            goals_reached = 0
            for path in paths:
                observations = path['observations']['observation']
                path_length = path['terminals'].size
                succeeded = np.any(path['infos']['is_success'][-path_length//2:])
                crossed_x_indices = (
                    np.flatnonzero(observations[:, 1] > 0))
                did_not_cross_x = crossed_x_indices.size < 1

                if did_not_cross_x:
                    continue

                first_crossed_x_index = crossed_x_indices[0]

                if observations[first_crossed_x_index, 0] <= -self.inner_wall_max_dist + 1.0:
                    lefts_success += int(succeeded)
                elif self.inner_wall_max_dist - 1.0  <= observations[first_crossed_x_index, 0]:
                    rights_success += int(succeeded)
                else:
                    raise ValueError("Should never be here!")

            infos.update({
                'succeeded_from_both_sides': (
                    lefts_success > 0 and rights_success > 0),
                'succeeded_from_left_count': lefts_success,
                'succeeded_from_right_count': rights_success,
            })
        elif isinstance(self, Point2DBridgeRunEnv):
            x, y = np.split(np.concatenate(tuple(itertools.chain(*[
                [
                    path['observations']['observation'],
                    path['next_observations']['observation'][[-1]]
                ]
                for path in paths
            ]))), 2, axis=-1)

            bins_per_unit = 1
            x_bounds = (
                self.observation_x_bounds[0]
                + self.extra_width_before
                + self.wall_length
                + self.bridge_length,
                self.observation_x_bounds[1]
            )
            y_bounds = tuple(self.observation_y_bounds)

            where_past_bridge = np.flatnonzero(np.logical_and.reduce((
                x_bounds[0] <= x,
                x <= x_bounds[1],
                y_bounds[0] <= y,
                y <= y_bounds[1])))

            if 0 < where_past_bridge.size:
                min_x = np.min(x[where_past_bridge])
                max_x = np.max(x[where_past_bridge])
                min_y = np.min(y[where_past_bridge])
                max_y = np.max(y[where_past_bridge])
                ptp_x = max_x - min_x
                ptp_y = max_y - min_y
                rectangle_area = ptp_x * ptp_y
                rectangle_support = rectangle_area / (
                    np.ptp(x_bounds) * np.ptp(y_bounds))
                rectangle_x_support = ptp_x / np.ptp(x_bounds)
                rectangle_y_support = ptp_y / np.ptp(y_bounds)
            else:
                min_x = max_x = min_y = max_y = ptp_x = ptp_y = 0.0
                rectangle_area = rectangle_support = 0.0
                rectangle_x_support = rectangle_y_support = 0.0

            H, xedges, yedges = np.histogram2d(
                np.squeeze(x),
                np.squeeze(y),
                bins=(
                    int(np.ptp(x_bounds) * bins_per_unit),
                    int(np.ptp(y_bounds) * bins_per_unit),
                ),
                range=np.array((x_bounds, y_bounds)),
            )

            histogram_support = np.sum(H > 0) / H.size
            H_x = np.sum(H, axis=1)
            H_y = np.sum(H, axis=0)
            histogram_x_support = np.sum(H_x > 0) / H_x.size
            histogram_y_support = np.sum(H_y > 0) / H_y.size

            infos.update({
                'after-bridge-min_x': min_x,
                'after-bridge-max_x': max_x,
                'after-bridge-min_y': min_y,
                'after-bridge-max_y': max_y,
                'after-bridge-ptp_x': ptp_x,
                'after-bridge-ptp_y': ptp_y,
                'after-bridge-histogram_support': histogram_support,
                'after-bridge-histogram_x_support': histogram_x_support,
                'after-bridge-histogram_y_support': histogram_y_support,
                'after-bridge-rectangle_area': rectangle_area,
                'after-bridge-rectangle_support': rectangle_support,
                'after-bridge-rectangle_x_support': rectangle_x_support,
                'after-bridge-rectangle_y_support': rectangle_y_support,
            })

        elif isinstance(self, Point2DPondEnv):
            x, y = np.split(np.concatenate(tuple(itertools.chain(*[
                [
                    path['observations']['observation'],
                    path['next_observations']['observation'][[-1]]
                ]
                for path in paths
            ]))), 2, axis=-1)

            bins_per_unit = 5
            x_bounds = tuple(self.observation_x_bounds)
            y_bounds = tuple(self.observation_y_bounds)

            H, xedges, yedges = np.histogram2d(
                np.squeeze(x),
                np.squeeze(y),
                bins=(
                    int(np.ptp(x_bounds) * bins_per_unit),
                    int(np.ptp(y_bounds) * bins_per_unit),
                ),
                range=np.array((x_bounds, y_bounds)),
            )

            full_area = (
                (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]))
            water_area = np.pi * self.pond_radius ** 2 / 4
            support_of_total_area = (np.sum(H > 0) / H.size)
            support_of_walkable_area = (
                support_of_total_area * full_area
                / (full_area - water_area))

            infos.update({'support': support_of_walkable_area})

        log_base_dir = os.getcwd()
        heatmap_dir = os.path.join(log_base_dir, 'heatmap')
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

        previous_heatmaps = glob.glob(
            os.path.join(heatmap_dir, f"{evaluation_type}-iteration-*-heatmap.png"))
        heatmap_iterations = [
            int(re.search(f"{evaluation_type}-iteration-(\d+)-heatmap.png", x).group(1))
            for x in previous_heatmaps
        ]
        if not heatmap_iterations:
            iteration = 0
        else:
            iteration = int(max(heatmap_iterations) + 1)

        base_size = 6.4
        x_min, x_max = self.observation_x_bounds
        y_min, y_max = self.observation_y_bounds
        width = x_max - x_min
        height = y_max - y_min

        if width > height:
            figsize = (base_size, base_size * (height / width))
        else:
            figsize = (base_size * (width / height), base_size)

        figure, axis = plt.subplots(1, 1, figsize=figsize)
        axis.set_xlim(self.observation_x_bounds)
        axis.set_ylim(self.observation_y_bounds)

        color_map = plt.cm.get_cmap('PuBuGn', len(paths))
        for i, path in enumerate(paths):
            positions = np.concatenate((
                path['observations']['observation'],
                path['next_observations']['observation'][[-1]],
            ), axis=0)
            color = color_map(i)
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linestyle=':',
                linewidth=1.0,
                label='evaluation_paths' if i == 0 else None,
            )
            axis.scatter(
                *positions[0],
                color=color,
                marker='o',
                s=20.0,
            )
            axis.scatter(
                *positions[-1],
                color=color,
                marker='x',
                s=20.0,
            )

        axis.scatter(
            *self.fixed_goal,
            color='red',
            marker='*',
            s=30.0)

        plot_walls(axis, self.walls)
        if hasattr(self, 'waters'):
            plot_waters(axis, self.waters)
        elif isinstance(self, Point2DPondEnv):
            pond_circle = mpl.patches.Circle(
                (0, 0),
                self.pond_radius,
                facecolor='blue',
                edgecolor='blue',
                fill=True,
            )

            axis.add_patch(pond_circle)

        # nx = ny = 500

        # x = np.linspace(*x_bounds, nx)
        # y = np.linspace(*y_bounds, ny)
        # X, Y = np.meshgrid(x, y)
        # xy = np.stack((X, Y), axis=-1).reshape(-1, 2)

        # axis.scatter(
        #     xy[:, 0],
        #     xy[:, 1],
        #     c=np.where(self.in_waters(xy), 'red', 'green').ravel(),
        #     marker='.',
        #     s=1.0
        # )

        # x, y = np.split(np.concatenate([
        #     path['observations']['observation']
        #     for path in paths
        # ]), 2, axis=-1)

        # x_bounds = tuple(self.observation_x_bounds)
        # y_bounds = tuple(self.observation_y_bounds)

        # bins_per_unit = 5
        # counts, xedges, yedges, image = axis.hist2d(
        #     x.squeeze(),
        #     y.squeeze(),
        #     bins=(
        #         int(np.ptp(x_bounds) * bins_per_unit),
        #         int(np.ptp(y_bounds) * bins_per_unit),
        #     ),
        #     range=(x_bounds, y_bounds),
        #     cmap="PuBuGn",
        #     vmax=10,
        # )

        # plt.colorbar(image, ax=axis)

        heatmap_path = os.path.join(
            heatmap_dir,
            f'{evaluation_type}-iteration-{iteration:05}-heatmap.png')
        plt.savefig(heatmap_path)
        figure.clf()
        plt.close(figure)

        return infos


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

        if d < self.target_radius:
            reward += 200.0

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
                np.maximum(self.target_radius, 0.5),
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

        for water in getattr(self, 'waters', ()):
            lower_left, upper_right = water
            # upper_left = (lower_left[0], upper_right[1])
            # lower_right = (upper_right[0], lower_left[1])
            # endpoints = np.array((lower_left, upper_left, upper_right, lower_right))
            # for point1, point2 in zip(endpoints, np.roll(endpoints, -1, axis=0)):
            #     drawer.draw_segment(point1, point2, Color('blue'))
            # upper_left = None
            # rect_location = (upper_right + lower_left) / 2
            # rect_location = (lower_left[0], upper_right[1])
            rect_location = lower_left
            width, height = (upper_right - lower_left) # + [0, 0.1]
            drawer.draw_rect(rect_location, width, height, (0, 0, 255), thickness=0)

        if isinstance(self, Point2DPondEnv):
            drawer.draw_solid_circle((0, 0), self.pond_radius, (0, 0, 255))

        drawer.render()
        if tick:
            drawer.tick(self.render_dt_msec)

    def render(self, mode='human', close=False, width=None, height=None):
        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            observation_width, observation_height = (
                self.observation_box.high - self.observation_box.low)
            self.render_drawer = PygameViewer(
                screen_width=width or int(
                    self.render_size * observation_width / observation_height),
                screen_height=width or self.render_size,
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
        elif wall_shape == '-':
            walls = (
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
            )

        super().__init__(
            ball_radius=ball_radius,
            observation_bounds=observation_bounds,
            walls=walls,
            **kwargs)


class Point2DBridgeEnv(Point2DEnv):
    def __init__(
            self,
            ball_radius=0.0,
            bridge_width=1.0,
            bridge_length=4.0,
            wall_width=4.0,
            wall_length=0.1,
            wall_thickness=0.1,
            scale=1.0,
            fixed_goal=None,
            target_radius=0.5,
            extra_width_before=2.0,
            extra_width_after=2.0,
            **kwargs,
    ):

        self.quick_init(locals())

        self.ball_radius = ball_radius

        self.bridge_width = bridge_width
        self.bridge_length = bridge_length
        self.wall_width = wall_width
        self.wall_length = wall_length
        self.scale = scale

        # 8.0 = 2.0 behind the wall + 2.0 between wall and bridge
        # + 4.0 after the bridge.
        self.extra_width_before = extra_width_before
        self.extra_width_after = extra_width_after

        assert 2.0 <= extra_width_before
        assert 2.0 <= extra_width_after

        total_length = scale * (
            bridge_length + wall_length * 2 + extra_width_before + extra_width_after)
        fixed_goal_y = fixed_goal[1] if fixed_goal else 0.0

        total_width = scale * (
            2 * extra_width_after
            # extra_width_before
            + max(wall_width, bridge_width, 2 * np.abs(fixed_goal_y))
            # + extra_width_after
        )

        max_x = total_length / 2
        min_x = - max_x

        fixed_goal = fixed_goal or (max_x - extra_width_after / 2, 0)
        # max_y = 2.0 + max(wall_width / 2, bridge_width / 2, fixed_goal[1])
        # min_y = - (2.0 + min(-wall_width / 2, -bridge_width / 2, fixed_goal[1]))
        # total_width = max_y - min_y

        max_y = total_width / 2
        min_y = - max_y

        observation_bounds = np.array(((min_x, min_y), (max_x, max_y)))
        x_low, x_high = observation_bounds[:, 0]
        y_low, y_high = observation_bounds[:, 1]

        if wall_thickness <= 0.0 or wall_width <= 0.0 or wall_length <= 0.0:
            walls = ()
        else:
            walls = (
                VerticalWall(
                    self.ball_radius,
                    min_x + extra_width_before + bridge_length + 1.0,
                    # max_x - extra_width_after - 0.1,
                    # min_x + 1.5 - 0.1,
                    -wall_width / 2,
                    wall_width / 2,
                    thickness=wall_thickness,
                ),
            )

        water_width = (total_width - bridge_width) / 2 # - bridge_width
        water_length = bridge_length
        self.waters = (  # lower-left, upper-right
            (
                np.array((min_x + wall_length + extra_width_before, max_y - water_width)),
                np.array((min_x + wall_length + extra_width_before + water_length, max_y + 0.1)),
            ),
            (
                np.array((min_x + wall_length + extra_width_before, min_y - 0.1)),
                np.array((min_x + wall_length + extra_width_before + water_length, min_y + water_width)),
            ),
            # (
            #     np.array((-extra_width, -4.0)),
            #     np.array((3.0, 4.0))
            # ),
        )

        super().__init__(
            ball_radius=ball_radius,
            observation_bounds=observation_bounds,
            walls=walls,
            reset_positions=((min_x + 1, 0), ),
            fixed_goal=fixed_goal,
            **kwargs)

    def in_water(self, states):
        states = np.atleast_2d(states)

        lower_lefts, upper_rights = np.swapaxes(self.waters, 1, 0)
        in_waters = np.all(
            np.logical_and(lower_lefts <= states, states <= upper_rights),
            axis=1)

        in_water = np.any(in_waters)

        expected_value = False
        for lower_left, upper_right in self.waters:
            in_water_index = np.all(
                np.logical_and(lower_left <= states, states <= upper_right),
                axis=1)
            if np.any(in_water_index):
                expected_value = True

        assert in_water == expected_value, (in_water, expected_value)

        return in_water

    def step(self, action, *args, **kwargs):
        observation = self._get_obs()
        if self.in_water(observation['state_observation']):
            action = np.zeros_like(action)
            observation, reward, done, info = super(Point2DBridgeEnv, self).step(
                action, *args, **kwargs)
            reward = -1.0 * info['distance_to_target'] - 2.0 * np.log(2.0)
            info['in_water'] = True
            return observation, reward, done, info

        observation, reward, done, info = super(Point2DBridgeEnv, self).step(
            action, *args, **kwargs)

        info['in_water'] = False
        if self.in_water(observation['state_observation']):
            reward = -1.0 * info['distance_to_target'] - 2.0 * np.log(2.0)
            info['in_water'] = True
            # done = True

        return observation, reward, done, info

class Point2DBridgeRunEnv(Point2DBridgeEnv):
    def __init__(self,
                 *args,
                 extra_width_before=2.0,
                 extra_width_after=10.0,
                 **kwargs):
        self.quick_init(locals())

        return super(Point2DBridgeRunEnv, self).__init__(
            *args,
            wall_width=0,
            wall_length=0,
            wall_thickness=0,
            extra_width_before=extra_width_before,
            extra_width_after=extra_width_after,
            **kwargs)

    def step(self, action, *args, **kwargs):
        observation0 = self._get_obs()
        observation, reward, done, info = super(Point2DBridgeRunEnv, self).step(
            action, *args, **kwargs)

        before_water = (
            observation['observation'][0]
            <= (
                self.observation_x_bounds[0]
                + self.extra_width_before
                + self.wall_length
            )
        )
        past_water = (
            (
                self.observation_x_bounds[0]
                + self.extra_width_before
                + self.wall_length
                + self.bridge_length
            ) <= observation['observation'][0]
        )
        if before_water:
            reward = -0.15
        elif past_water:
            reward = 3.0
        elif not info['in_water']:
            xy_velocity = observation['observation'] - observation0['observation']
            x_velocity = xy_velocity[0]
            multiplier = 3.0
            reward = multiplier * x_velocity
            try:
                assert -1.0 <= reward / multiplier <= 1.0, reward
            except Exception as e:
                from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
                pass
        elif info['in_water']:
            # reward = -2 * np.log(2)
            reward = -0.1
        else:
            from pprint import pprint; import ipdb; ipdb.set_trace(context=30)
            raise ValueError

        return observation, reward, done, info


class Point2DPondEnv(Point2DEnv):
    def __init__(
            self,
            ball_radius=0.0,
            pond_radius=1.0,
            fixed_goal=None,
            target_radius=0.5,
            **kwargs,
    ):
        self.quick_init(locals())
        self.ball_radius = ball_radius
        self.pond_radius = pond_radius

        total_length = 2 * pond_radius + 44  # 44 = 2 * (max_path_length + 2)
        max_x = total_length / 2
        min_x = -max_x
        min_y, max_y = min_x, max_x
        observation_bounds = np.array(((min_x, min_y), (max_x, max_y)))

        fixed_goal = fixed_goal or (
            0, pond_radius + max(1.0, 0.1 * pond_radius))

        super().__init__(
            ball_radius=ball_radius,
            observation_bounds=observation_bounds,
            walls=(),
            reset_positions=((pond_radius + 0.5, 0), ),
            fixed_goal=fixed_goal,
            **kwargs)

    def step(self, action, *args, **kwargs):
        observation, reward, done, info = super(Point2DPondEnv, self).step(
            action, *args, **kwargs)

        info['distance_from_water'] = self.distance_from_pond_center(
            observation['state_observation']
        ) - self.pond_radius

        if self.in_water(observation['state_observation']):
            reward = -self._max_episode_steps * info['distance_to_target']
            done = True
            info['in_water'] = True

        return observation, reward, done, info

    def distances_from_pond_center(self, states):
        states = np.atleast_2d(states)
        pond_center = np.array((0.0, 0.0))
        distances_from_pond_center = np.linalg.norm(
            states - pond_center, ord=2, keepdims=True, axis=1)
        return distances_from_pond_center

    def distance_from_pond_center(self, state):
        distance_from_pond_center = self.distances_from_pond_center(
            np.atleast_2d(state))[0]
        return distance_from_pond_center

    def in_waters(self, states):
        in_waters = self.distances_from_pond_center(states) < self.pond_radius
        return in_waters

    def in_water(self, state):
        in_water = self.in_waters(np.atleast_2d(state))[0]
        return in_water


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
