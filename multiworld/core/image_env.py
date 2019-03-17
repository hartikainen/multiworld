import copy
import random

import cv2
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict


class ImageEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            transpose=False,
            grayscale=False,
            normalize=False,
            reward_type='image_distance',
            threshold=10,
            image_length=None,
            presampled_goals=None,
            non_presampled_goal_img_is_garbage=False,
            recompute_reward=True,
            high_res_size=480,
            use_high_res=False,
    ):
        """

        :param wrapped_env:
        :param imsize:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        :param non_presampled_goal_img_is_garbage: Set this option to True if
        you want to allow the code to work without presampled goals,
        but where the underlying env doesn't support set_to_goal. As the name,
        implies this will make it so that the goal image is garbage if you
        don't provide pre-sampled goals. The main use case is if you want to
        use an ImageEnv to pre-sample a bunch of goals.
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = True
        self.imsize = imsize
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
        self.high_res_size = high_res_size

        if image_length is not None:
            self.image_length = image_length
        else:
            if grayscale:
                self.image_length = self.imsize * self.imsize
            else:
                self.image_length = 3 * self.imsize * self.imsize
        self.channels = 1 if grayscale else 3

        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.initialize_camera(init_camera)
            # viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            # init_camera(viewer.cam)
            # sim.add_render_context(viewer)
        self._render_local = False
        img_space = Box(0, 1, (self.image_length,), dtype=np.float32)
        high_res_img_space = Box(0, 1, (high_res_size**2 * 3,), dtype=np.float32)
        self._img_goal = img_space.sample() #has to be done for presampling
        self._high_res_img_goal = high_res_img_space.sample() #has to be done for presampling
        self._high_res_3d_img_goal = high_res_img_space.sample() #has to be done for presampling
        self.use_high_res = use_high_res
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['observation'] = img_space
        spaces['desired_goal'] = img_space
        spaces['achieved_goal'] = img_space
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space
        spaces['high_res_image_desired_goal'] = high_res_img_space
        spaces['high_res_3d_image_desired_goal'] = high_res_img_space

        self.return_image_proprio = False
        if 'proprio_observation' in spaces.keys():
            self.return_image_proprio = True
            spaces['image_proprio_observation'] = concatenate_box_spaces(
                spaces['image_observation'],
                spaces['proprio_observation']
            )
            spaces['image_proprio_desired_goal'] = concatenate_box_spaces(
                spaces['image_desired_goal'],
                spaces['proprio_desired_goal']
            )
            spaces['image_proprio_achieved_goal'] = concatenate_box_spaces(
                spaces['image_achieved_goal'],
                spaces['proprio_achieved_goal']
            )

        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]

        if self.use_high_res:
            self.high_res_cam = cv2.VideoCapture(0)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        achieved_goal = obs['image_achieved_goal']
        desired_goal = self._img_goal
        image_dist = np.linalg.norm(achieved_goal-desired_goal)
        image_success = (image_dist<self.threshold).astype(float)-1
        info['image_dist'] = image_dist
        info['image_success'] = image_success

    def reset(self):
        obs = self.wrapped_env.reset()

        if self.use_high_res:
            self.high_res_cam.release()
            self.high_res_cam = cv2.VideoCapture(0)

        if self.num_goals_presampled > 0:
            goal = self.sample_goal()
            self._img_goal = goal['image_desired_goal']
            if self.use_high_res:
                self._high_res_img_goal = goal['high_res_image_desired_goal']
                self._high_res_3d_img_goal = goal['high_res_3d_image_desired_goal']
            self.temp_img_goal = self._img_goal.copy()
            self.wrapped_env.set_goal(goal)
            for key in goal:
                obs[key] = goal[key]
        elif self.non_presampled_goal_img_is_garbage:
            # This is use mainly for debugging or pre-sampling goals.
            self._img_goal = self._get_flat_img()
            if self.use_high_res:
                self._high_res_img_goal = self._get_flat_img(imsize=(self.high_res_size, self.high_res_size))
                self._high_res_3d_img_goal = self._get_flat_img(imsize=(self.high_res_size, self.high_res_size), use_webcam=True)
        else:
            env_state = self.wrapped_env.get_env_state()
            self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())
            self._img_goal = self._get_flat_img()
            if self.use_high_res:
                self._high_res_img_goal = self._get_flat_img(imsize=(self.high_res_size, self.high_res_size), use_webcam=False)
            self.wrapped_env.set_env_state(env_state)

        return self._update_obs(obs)

    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        img_obs = self._get_flat_img(use_webcam=False)
        obs['image_observation'] = img_obs
        obs['image_desired_goal'] = self._img_goal
        obs['image_achieved_goal'] = img_obs

        if self.use_high_res:
            high_res_img_obs = self._get_flat_img((self.high_res_size, self.high_res_size), use_webcam=False)
            high_res_3d_img_obs = self._get_flat_img((self.high_res_size, self.high_res_size), use_webcam=True)
            obs['high_res_image_observation'] = high_res_img_obs
            obs['high_res_image_desired_goal'] = self._high_res_img_goal

            obs['high_res_3d_image_observation'] = high_res_3d_img_obs
            obs['high_res_3d_image_desired_goal'] = self._high_res_3d_img_goal

        obs['observation'] = img_obs
        obs['desired_goal'] = self._img_goal
        obs['achieved_goal'] = img_obs

        if self.return_image_proprio:
            obs['image_proprio_observation'] = np.concatenate(
                (obs['image_observation'], obs['proprio_observation'])
            )
            obs['image_proprio_desired_goal'] = np.concatenate(
                (obs['image_desired_goal'], obs['proprio_desired_goal'])
            )
            obs['image_proprio_achieved_goal'] = np.concatenate(
                (obs['image_achieved_goal'], obs['proprio_achieved_goal'])
            )

        return obs

    def _get_flat_img(self, imsize=None, use_webcam=False):
        if imsize is None:
            imsize = (self.imsize, self.imsize)
        # returns the image as a torch format np array
        if use_webcam:
            success, img_obs = self.high_res_cam.read()
            if success:
                img_obs = copy.deepcopy(img_obs)
                img_obs = np.array(img_obs).reshape(480, 640, 3)
                img_obs = img_obs[:, 0:(640-160), :]
                img_obs = cv2.resize(img_obs, (0, 0), fx=imsize[0] / img_obs.shape[0], fy=imsize[1] / img_obs.shape[1],
                                     interpolation=cv2.INTER_AREA)
                image_obs = np.asarray(img_obs).reshape(imsize[0], imsize[1], 3)[:, :, [2, 1, 0]]
            else:
                raise EnvironmentError('unable to read video')
        else:
            image_obs = self._wrapped_env.get_image(
                width=imsize[0],
                height=imsize[1],
            )
        if self._render_local:
            cv2.imshow('env', image_obs)
            cv2.waitKey(1)
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        return image_obs.flatten()

    def render(self):
        self.wrapped_env.render()

    def enable_render(self):
        self._render_local = True

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._img_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)

    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")
        img_goals = np.zeros((batch_size, self.image_length))
        goals = self.wrapped_env.sample_goals(batch_size)
        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)
            img_goals[i, :] = self._get_flat_img()
        goals['desired_goal'] = img_goals
        goals['image_desired_goal'] = img_goals
        return goals

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type=='image_distance':
            return -dist
        elif self.reward_type=='image_sparse':
            return -(dist > self.threshold).astype(float)
        elif self.reward_type=='wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
