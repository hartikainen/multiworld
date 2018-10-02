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


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=True,
            render_size=400,
            reward_type="dense",
            target_radius=0.5,
            boundary_dist=30,
            ball_radius=0.25,
            walls=None,
            fixed_goal=(15.0, -15.0),
            randomize_position_on_reset=False,
            images_are_rgb=False,  # else black and white
            reset_position=(-15.0, -15.0),
            **kwargs
    ):
        if walls is None:
            walls = []
        if len(kwargs) > 0:
            import logging
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = (
            np.array(fixed_goal, dtype=np.float32)
            if fixed_goal is not None
            else None)
        self.randomize_position_on_reset = randomize_position_on_reset
        self.images_are_rgb = images_are_rgb

        self._max_episode_steps = 50
        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = None
        self._position = np.zeros((2))
        self._velocity = np.zeros((2))
        self._reset_position = np.array(reset_position)

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
            ('reacher_observation', spaces.Box(
                np.concatenate((
                    -o, # position
                    -o, # position
                    -o, # target position
                    -u, # velocity
                    -2*o, # position_diff
                    np.zeros(1), # z-axis
                )),
                np.concatenate((
                    o, # position
                    o, # position
                    o, # target position
                    u, # velocity
                    2*o, # position_diff
                    np.zeros(1) # z-axis
                )),
                dtype='float32'))
        ])

        self.drawer = None

    def step(self, velocities):
        velocities = np.clip(velocities, a_min=-1, a_max=1)
        self._velocity = velocities
        new_position = self._position + velocities
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
        }
        done = False
        return ob, reward, done, info

    def _sample_goal(self):
        return np.random.uniform(
            size=2, low=-self.max_target_distance, high=self.max_target_distance
        )

    def get_reset_positions(self, N=1):
        if self.randomize_position_on_reset:
            positions = np.random.uniform(
                size=(N, 2), low=-self.boundary_dist, high=self.boundary_dist)
            positions_inside_walls = self._positions_inside_wall(positions)
            while np.any(positions_inside_walls):
                # positions_inside_walls_idx = np.where(positions_inside_walls)
                num_positions_inside_walls = np.sum(positions_inside_walls)
                positions[positions_inside_walls] = np.random.uniform(
                    size=(num_positions_inside_walls, 2),
                    low=-self.boundary_dist,
                    high=self.boundary_dist)
                positions_inside_walls = self._positions_inside_wall(positions)
        else:
            positions = np.tile(self._reset_position, (N, 1))

        return positions

    def get_reset_position(self):
        position = self.get_reset_positions(N=1)[0, ...]
        return position

    def reset(self):
        # self._target_position = np.random.uniform(
        #     size=2, low=-self.max_target_distance, high=self.max_target_distance
        # )
        if self.fixed_goal is not None:
            self._target_position = np.array(self.fixed_goal)
        else:
            self._target_position = np.random.uniform(
                size=2,
                low=-self.max_target_distance,
                high=self.max_target_distance)

        self._position = self.get_reset_position()

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
        observation_dict =  dict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=self._position.copy(),
            state_desired_goal=self._target_position.copy(),
            state_achieved_goal=self._position.copy(),
            reacher_observation=np.concatenate([
                self._position.copy(),
                # Duplicate position to make consistent with reacher observation
                self._position.copy(),
                self._target_position.copy(),
                self._velocity,
                self._position.copy() - self._target_position.copy(),
                # Add z-axis to make consistent with reacher observation
                np.array([0])
            ])
        )
        return observation_dict

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_observation']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        if self.reward_type == "dense":
            return -d

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def _sample_position(self, low, high, realistic=True):
        pos = np.random.uniform(low, high)
        if realistic:
            while self._position_inside_wall(pos) is True:
                pos = np.random.uniform(low, high)
        return pos

    def sample_goals(self, batch_size):
        if self.fixed_goal is not None:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.zeros((batch_size, self.obs_range.low.size))
            for b in range(batch_size):
                goals[b, :] = self._sample_position(
                    self.obs_range.low, self.obs_range.high,
                    # realistic=self.sample_realistic_goals
                )
            # goals = np.random.uniform(
            #     self.obs_range.low,
            #     self.obs_range.high,
            #     size=(batch_size, self.obs_range.low.size),
            # )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

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
                    x_bounds=(-self.boundary_dist, self.boundary_dist),
                    y_bounds=(-self.boundary_dist, self.boundary_dist),
                    render_onscreen=self.render_onscreen,
                )
                self.render_size = width
        self.render()
        img = self.drawer.get_image()
        if self.images_are_rgb:
            return img.transpose().flatten()
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).flatten()
            return img

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._position = goal
        self._target_position = goal

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._position = position
        self._target_position = goal

    def render(self, mode='human', close=False):
        if close:
            self.drawer = None
            return

        if self.drawer is None or self.drawer.terminated:
            self.drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist, self.boundary_dist),
                y_bounds=(-self.boundary_dist, self.boundary_dist),
                render_onscreen=self.render_onscreen,
            )

        self.drawer.fill(Color('white'))
        self.drawer.draw_solid_circle(
            self._target_position,
            self.target_radius * 5,
            Color('green'),
        )
        self.drawer.draw_solid_circle(
            self._position,
            self.ball_radius * 5,
            Color('blue'),
        )

        # self.drawer.draw_segment(
        #     (0, -self.boundary_dist),
        #     (0, self.boundary_dist),
        #     Color('black')
        # )

        # self.drawer.draw_segment(
        #     (-self.boundary_dist, 0),
        #     (self.boundary_dist, 0),
        #     Color('black')
        # )

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

    # Static visualization/utility methods

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )
        ax.set_xlim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )

    def initialize_camera(self, init_fctn):
        pass


class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            # wall_shape="_|",
            wall_shape="u",
            inner_wall_max_dist=12,
            thickness=3.0,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        if wall_shape == "_|":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.fixed_goal[0] - 10,
                    self.fixed_goal[1] - 10,
                    self.fixed_goal[1] + 5,
                    thickness=thickness,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.fixed_goal[1] + 5,
                    self.fixed_goal[0] - 25,
                    self.fixed_goal[0] - 10,
                    thickness=thickness,
                )
            ]
        elif wall_shape == "u":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                )
            ]
        elif wall_shape == "n":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                )
            ]
        if wall_shape == "-":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                )
            ]
        if wall_shape == "--":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                )
            ]
        elif wall_shape == "|-":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                    thickness=thickness,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist // 2,
                    self.inner_wall_max_dist // 2,
                    thickness=thickness,
                ),
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist // 2,
                    self.inner_wall_max_dist // 2,
                    thickness=thickness,
                ),
            ]



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
