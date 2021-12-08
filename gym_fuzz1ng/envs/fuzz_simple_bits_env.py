import gym_fuzz1ng
import gym_fuzz1ng.coverage as coverage

from gym_fuzz1ng.envs.fuzz_base_env import FuzzBaseEnv, FuzzBaseEnvBytes


class FuzzSimpleBitsEnv(FuzzBaseEnvBytes):
    def __init__(self):
        self._input_size = 64
        self._target_path = gym_fuzz1ng.simple_bits_target_path()
        self._args = []
        self._dict = coverage.Dictionary({
            'tokens': [],
            'bytes': True,
        })
        super(FuzzSimpleBitsEnv, self).__init__()


class FuzzSimpleBitsEnvSmall(FuzzBaseEnvBytes):
    def __init__(self):
        self._input_size = 16
        self._target_path = gym_fuzz1ng.simple_bits_target_path()
        self._args = []
        self._dict = coverage.Dictionary({
            'tokens': [],
            'bytes': True,
        })
        super(FuzzSimpleBitsEnvSmall, self).__init__()
    
    def getpath(self):
        return self._target_path
