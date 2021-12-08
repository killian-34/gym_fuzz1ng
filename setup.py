import subprocess
import sys

from setuptools import setup
from distutils.command.build import build as DistutilsBuild


class Build(DistutilsBuild):
    def run(self):
        try:
            subprocess.check_call(['make', 'all'], cwd='gym_fuzz1ng/mods')
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not build mods: %s.\n" % e)
            raise
        DistutilsBuild.run(self)


setup(
    name='gym_fuzz1ng',
    version='0.1.0',
    platforms='Posix',
    install_requires=[
        'gym>=0.10.3',
        'xxhash>=1.0.1',
        'posix_ipc>=1.0.3',
    ],
    author='adbq, spolu',
    package_data={
        'gym_fuzz1ng.mods': [
            'afl-2.52b-mod/afl-2.52b/afl-forkserver',
            'libpng-1.6.34-mod/libpng_simple_fopen_afl',
            'simple_bits-mod/simple_bits_afl',
            'simple_loop-mod/simple_loop_afl',
        ],
    },
    packages=[
        'gym_fuzz1ng',
        'gym_fuzz1ng.coverage',
        'gym_fuzz1ng.mods',
        'gym_fuzz1ng.envs',
        'gym_fuzz1ng.utils'
    ],
    cmdclass={'build': Build},
    include_package_data=True
)
