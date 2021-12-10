import os

from gym.envs.registration import register


def afl_forkserver_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/afl-2.52b-mod/afl-2.52b/afl-forkserver',
    )


def simple_2ladder_syscall_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/simple_2ladder_syscall-mod/simple_2ladder_syscall_afl',
    )

register(
    id='FuzzSimple2LadderSyscall-v0',
    entry_point='gym_fuzz1ng.envs:FuzzSimple2LadderSyscallEnv',
)


def simple_bits_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/simple_bits-mod/simple_bits_afl',
    )


register(
    id='FuzzSimpleBits-v0',
    entry_point='gym_fuzz1ng.envs:FuzzSimpleBitsEnv',
)
