"""State-based (proprio) Continual environment for SAC.

Supports both dm_control tasks (e.g. 'cheetah_run') and gym Mujoco tasks
(e.g. 'HalfCheetah-v4').  Task type is detected automatically by whether
the name contains '-v' (gym convention).

A fixed observation_space / action_space is shared across all tasks.
Shorter observations are zero-padded; actions from SAC (in [-1, 1]) are
rescaled to the real action range before being passed to the underlying env.
"""

import numpy as np
import gym
from gym import spaces
from dm_control import suite


# Common dm_control proprio observation sizes (flattened).
TASK_OBS_DIMS = {
    'cartpole_balance':    5,
    'cartpole_swingup':    5,
    'reacher_easy':        6,
    'reacher_hard':        6,
    'finger_spin':         9,
    'ball_in_cup_catch':   8,
    'pendulum_swingup':    3,
    'acrobot_swingup':     6,
    'cheetah_run':        17,
    'hopper_hop':         15,
    'hopper_stand':       15,
    'walker_walk':        24,
    'walker_run':         24,
    'walker_stand':       24,
    'humanoid_walk':      67,
    'humanoid_run':       67,
    'dog_walk':          223,
}


def _is_gym_task(name: str) -> bool:
    """Return True if name looks like a gym env id (e.g. 'HalfCheetah-v4')."""
    return '-v' in name


class ContinualDMCEnv(gym.Env):
    """Serial continual-learning wrapper over dm_control or gym Mujoco tasks."""

    metadata = {'render.modes': []}

    def __init__(
        self,
        task_list: list,
        obs_dim: int,
        act_dim: int,
        seed: int = 0,
    ):
        super().__init__()
        assert len(task_list) > 0
        self.task_list = task_list
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self._seed     = seed
        self._task_idx = 0
        self.task_step = 0

        # Fixed spaces shared across all tasks.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        self._env          = None
        self._task_type    = None   # 'dmc' or 'gym'
        self._real_act_dim = None
        self._act_low      = None   # gym task only
        self._act_high     = None   # gym task only
        self._load_task(self.task_list[0])

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _load_task(self, task_name: str):
        """Load a dm_control or gym task."""
        if self._env is not None:
            self._env.close()

        if _is_gym_task(task_name):
            raw = gym.make(task_name)
            self._env          = raw
            self._task_name    = task_name
            self._task_type    = 'gym'
            self._real_act_dim = raw.action_space.shape[0]
            self._act_low      = raw.action_space.low [:self._real_act_dim].astype(np.float32)
            self._act_high     = raw.action_space.high[:self._real_act_dim].astype(np.float32)
        else:
            domain, task = task_name.split('_', 1)
            if domain == 'cup':
                domain = 'ball_in_cup'
            dm_env = suite.load(domain, task,
                                task_kwargs={'random': self._seed})
            self._env          = dm_env
            self._task_name    = task_name
            self._task_type    = 'dmc'
            self._real_act_dim = dm_env.action_spec().shape[0]
            self._act_low      = None
            self._act_high     = None

        print(f'[ContinualDMCEnv] Loaded task: {task_name}  '
              f'(act_dim={self._real_act_dim}, fixed_act_dim={self.act_dim})')

    @property
    def current_task(self) -> str:
        return self._task_name

    def switch_task(self, task_name: str):
        self._load_task(task_name)
        self.task_step = 0
        return self.reset()

    def next_task(self):
        self._task_idx = (self._task_idx + 1) % len(self.task_list)
        return self.switch_task(self.task_list[self._task_idx])

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _pad_obs(self, flat: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat, dtype=np.float32).flatten()
        if flat.shape[0] >= self.obs_dim:
            return flat[:self.obs_dim]
        return np.concatenate([flat,
                                np.zeros(self.obs_dim - flat.shape[0],
                                         dtype=np.float32)])

    def _flatten_dmc_obs(self, time_step) -> np.ndarray:
        parts = [np.asarray(v, dtype=np.float32).flatten()
                 for v in time_step.observation.values()]
        return self._pad_obs(np.concatenate(parts, axis=0))

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.task_step = 0
        if self._task_type == 'gym':
            obs = self._env.reset()
            return self._pad_obs(obs)
        else:
            return self._flatten_dmc_obs(self._env.reset())

    def step(self, action: np.ndarray):
        self.task_step += 1
        if self._task_type == 'gym':
            # Rescale SAC action from [-1, 1] to gym action range.
            a = np.asarray(action[:self._real_act_dim], dtype=np.float32)
            real_action = self._act_low + (a + 1.0) * 0.5 * (self._act_high - self._act_low)
            obs, reward, done, info = self._env.step(real_action)
            return self._pad_obs(obs), float(reward), bool(done), info
        else:
            real_action = np.asarray(action[:self._real_act_dim], dtype=np.float64)
            time_step = self._env.step(real_action)
            obs    = self._flatten_dmc_obs(time_step)
            reward = float(time_step.reward or 0.0)
            done   = time_step.last()
            info   = {}
            if done and time_step.discount == 1.0:
                info['TimeLimit.truncated'] = True
            return obs, reward, done, info

    def seed(self, seed: int = None):
        self._seed = seed or self._seed
        return [self._seed]

    def render(self, mode='rgb_array', height=84, width=84, camera_id=0):
        if self._task_type == 'gym':
            return self._env.render(mode=mode)
        return self._env.physics.render(height=height, width=width,
                                        camera_id=camera_id)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
