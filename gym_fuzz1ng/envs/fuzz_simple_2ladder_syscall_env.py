import gym_fuzz1ng
import gym_fuzz1ng.coverage as coverage

from gym_fuzz1ng.envs.fuzz_base_env import FuzzBaseEnvBytes


class FuzzSimple2LadderSyscallEnv(FuzzBaseEnvBytes):
    def __init__(self):
        self._input_size = 64
        self._target_path = gym_fuzz1ng.simple_2ladder_syscall_target_path()
        self._args = []
        self._dict = coverage.Dictionary({
            'tokens': [],
            'bytes': True,
        })
        super(FuzzSimple2LadderSyscallEnv, self).__init__()