# Reinforcement Learning for Selecting Edit Actions During Fuzzing: An Investigation

Jackson Killian and Susobhan Ghosh

OpenAI Gym[0] environment for Reinforcement Learning based binary fuzzing. 
This repository contains a sample toy example to demonstrate the proof-of-concept
for learning using RL. More examples can be added easily to the 
`gym_fuzz1ng/mods/` folder, and editing the corresponding `Makefile`
instruction. This is part of the class project on Systems Security (CS263)
at Harvard University, taught by Prof. James Mickens.

The fuzzer implements a DQN to fuzz C programs.
State: Underlying system call counts 
Actions: Given an input of size `t` bytes, an action corresponds to choosing one 
byte out of `t` bytes, and randomly flipping one bit in that chosen byte
of the input. 
Reward: Reward of 1 if new execution path is found, otherwise it is decaying, inversely
proportional to the number of times the execution path has been seen before
Transition: The toy example has been set such that each branch triggers a new syscall.
This is a proof-of-concept example in order to overcome the limitation of not having
exact basic block execution information; so this serves as a proxy for the same.
So a new execution path will lead to change in sys call counts, essentially changing the 
state.

The RL-fuzz environment's engine is based on american fuzzy lop[1] (afl) and is 
capable of thousands of executions per seconds for moderaltely sized executables.

To compute code coverage, the underlying excecution engine assigns a random integer
in `[0, 255]` to each simple block in the targeted binary.  The coverage is
then represented by a `256x256` matrix of `int8` representing the number of
time a transition was executed (note that this differs from how afl computes
coverage). Since `int8` are used for efficiency, the number of transitions can
only be within `[0, 255]` and wraps otherwise. This coverage matrix for the
last step execution is exactly what is returned as observation.

- [0] https://gym.openai.com/
- [1] http://lcamtuf.coredump.cx/afl/

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