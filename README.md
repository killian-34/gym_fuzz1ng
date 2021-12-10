# RL-based fuzz1ng

OpenAI Gym[0] environment for binary fuzzing of a variety of libraries (libpng
for now), executables, as well as simpler examples.

The environment's engine is based on american fuzzy lop[1] (afl) and capable of
thousands of executions per seconds for moderaltely sized executables.

The action space is the following:
```
Box(low=0, high=DICT_SIZE-1, shape=(INPUT_SIZE,), dtype='int32')
```

`DICT_SIZE` and `INPUT_SIZE` depend on the environnment and the underlying
program to fuzz:
- `DICT_SIZE` is the size of the dictionnary used to fuzz the program. `EOF` is
  represented by `DICT_SIZE-1` and accessible by the `eof()` method on the
  environment.
- `INPUT_SIZE` is the input submitted for fuzzing it is fixed for each
  environment and represents a maximal size for inputs to fuzz; smaller inputs
  can be represented using `EOF`.

The environment simulates the following game:

- each action submits a full input for fuzzing and returns the number of unique
  transitions executed as reward.
- if no new coverage is discovered by an input, the game is ended.

(It is possible to simply call `step` independently of whether the game is done
or not if you're just interested in easily executing binaries and retrieving
the associated coverage from Python. See also `step_raw`[2]).

The observation space is the following:
```
Box(low=0, high=255, shape=(256, 256), dtype='int32')
```

To compute coverage, the underlying excecution engine assigns a random integer
in `[0, 255]` to each simple block in the targeted binary.  The coverage is
then represented by a `256x256` matrix of `int8` representing the number of
time a transition was executed (note that this differs from how afl computes
coverage). Since `int8` are used for efficiency, the number of transitions can
only be within `[0, 255]` and wraps otherwise. This coverage matrix for the
last step execution is exactly what is returned as observation.

- [0] https://gym.openai.com/
- [1] http://lcamtuf.coredump.cx/afl/
- [2] https://github.com/spolu/gym_fuzz1ng/blob/master/gym_fuzz1ng/envs/fuzz_base_env.py

## Installation

```
# Note that running setup.py bdist_wheel takes a bit a time as it builds our
# afl mod as well as the available targets.
pip install .

# You may need to run the following commands as well as superuser.
echo core >/proc/sys/kernel/core_pattern

# You can then test that everything works by running our dummy example.
python dummy_simple_bits.py
```

## Experiments

In order to replicate the experiments run in the paper, just run the following:
`./run_experiments.sh input_size n_trials`
where `input_size` is the size of the input, and `n_trials` is the number of trials to average runs over.

The plots are generated and stored in the `img` directory, while associated run data across
trials is logged in csvs stored in `csv` directory. The trained models are stored in the 
`model` directory