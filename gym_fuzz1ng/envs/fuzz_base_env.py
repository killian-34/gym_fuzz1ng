import gym
import numpy as np

from gym import spaces

import gym_fuzz1ng.coverage as coverage


class FuzzBaseEnv(gym.Env):
    def __init__(self):
        # Classes that inherit FuzzBase must define before calling this
        # constructor:
        # - self._input_size
        # - self._dict
        # - self._target_path
        # - self._args
        self.engine = coverage.Afl(self._target_path, args=self._args)

        self.observation_space = spaces.Box(
            0, 255, shape=(
                2, coverage.EDGE_MAP_SIZE, coverage.EDGE_MAP_SIZE
            ), dtype='int32',
        )
        self.action_space = spaces.Box(
            0, self._dict.eof(), shape=(self._input_size,), dtype='int32',
        )
        self.reset()

    def reset(self):
        self.total_coverage = coverage.Coverage()
        return coverage.Coverage().observation()

    def step_raw(self, action):

        action = action.astype(np.int32)
        assert self.action_space.contains(action)

        input_data = b""

        for i in range(self._input_size):
            if int(action[i]) == self._dict.eof():
                break
            input_data += self._dict.bytes(int(action[i]))

        c = self.engine.run(input_data)

        if c.crash_count() > 0:
            print("CRASH {}".format(input_data))

        return {
            "step_coverage": c,
            "input_data": input_data,
        }

    def step(self, action):
        info = self.step_raw(action)

        reward = 0.0
        done = False
        c = info['step_coverage']

        reward = c.transition_count()

        old_path_count = self.total_coverage.skip_path_count()
        self.total_coverage.add(c)
        new_path_count = self.total_coverage.skip_path_count()

        if old_path_count == new_path_count:
            done = True

        info['total_coverage'] = self.total_coverage

        return c.observation(), reward, done, info

    def render(self, mode='human', close=False):
        pass

    def eof(self):
        return self._dict.eof()

    def dict_size(self):
        return self._dict.size()

    def input_size(self):
        return self._input_size
