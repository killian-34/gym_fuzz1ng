import gym
import gym_fuzz1ng.coverage as coverage
import helper
import numpy as np
import time

from collections import deque
from gym_fuzz1ng.envs.fuzz_simple_bits_env import FuzzSimpleBitsEnvSmall
from gym_fuzz1ng.utils import run_strace
from fs.tempfs import TempFS

# RL imports
import my_ppo
import spinup.algos.pytorch.ppo.core as core
import torch
from torch.optim import Adam
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import xxhash

from my_dqn import Agent


# import wandb
# wandb.init(project="CS263", entity="cs263")

h = xxhash.xxh32()

N_EDIT_TYPES = 2
DUMMY = 10
NSTATES = 335

def get_observation(b,a):
    return get_observation_strace(b,a)
    # return get_observation_random(a)
    # return get_observation_determ(a)

def get_observation_random(a):
    return np.random.rand(DUMMY)

def get_observation_determ(a):
    return np.arange(DUMMY)



sno=0
tmp = TempFS(identifier='_fuzz', temp_dir='/tmp/') 
tempdir = str(tmp._temp_dir) + "/"

program_name = "FuzzSimple2LadderSyscall-v0"

def get_observation_strace(path_to_binary, a):
    global sno

    # update the file counter
    sno += 1

    # saves the file to a temporary path
    savepath = tempdir + program_name + \
        str(sno) + "_" + str(time.time())

    # s = time.time()
    with open(savepath, "wb") as binary_file:
        # Write bytes to file
        binary_file.write(a)
    # s1 = time.time()
    # print('write time',s1-s)

    # store strace output and then get the state from it
    # s = time.time()
    strace_out = run_strace.run_strace(path_to_binary, savepath)
    # s1 = time.time()
    
    state = run_strace.strace_state(strace_out)

    # just keep read and write!
    state = state[:2]

    # s2 = time.time()
    # print("next times...")
    # print(s1-s)
    # print(s2-s1)

    return state



# arguments copied from spinup my_dqn
def main(n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):


    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # NSTATES = len(run_strace.syscall_map)
    NSTATES = 2

    # for the simple 2ladder env, just need three bytes
    NACTIONS = 3*8
    
    
    # Prepare for interaction with environment
    start_time = time.time()
    
    ##################################
    # ------------------------------ #
    ##################################


    ####################
    # main fuzzing loop
    ####################

    env = gym.make('FuzzSimple2LadderSyscall-v0')
    # env = FuzzSimpleBitsEnvSmall()
    print("dict_size={} eof={}".format(env.dict_size(), env.eof()))
    env.reset()

    agent = Agent(state_size=NSTATES, action_size=NACTIONS, device=device, lr=LR, 
            buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, update_every=UPDATE_EVERY,
             gamma=GAMMA, tau=TAU, seed=0)

    total_coverage = coverage.Coverage()

    inputs = [
        np.array([0, 0, 0], dtype=np.int8).tobytes(),
        np.array([5, 0, 5], dtype=np.int8).tobytes(),
        np.array([50, 0, 50], dtype=np.int8).tobytes(),
        np.array([122, 0, 122], dtype=np.int8).tobytes(),
        np.array([0, 60, 0], dtype=np.int8).tobytes(),
        np.array([5, 60, 5], dtype=np.int8).tobytes(),
        np.array([50, 60, 50], dtype=np.int8).tobytes(),
        np.array([122, 60, 122], dtype=np.int8).tobytes(),
    ]

    inputs = [
        np.array([0, 0, 0, 0, 0], dtype=np.int8).tobytes(),
        np.array([5, 0, 0, 0, 5], dtype=np.int8).tobytes(),
        np.array([50, 0, 0, 0, 50], dtype=np.int8).tobytes(),
        np.array([122, 0, 0, 0, 122], dtype=np.int8).tobytes(),
        np.array([0, 0, 60, 0, 0], dtype=np.int8).tobytes(),
        np.array([5, 0, 60, 0, 5], dtype=np.int8).tobytes(),
        np.array([50, 0, 60, 0, 50], dtype=np.int8).tobytes(),
        np.array([122, 0, 60, 0, 122], dtype=np.int8).tobytes(),
    ]

    Ntot = 9
    N = (Ntot -3)//2
    inputs = [
        np.array([0]+ [0]*N + [0] + [0]*N + [0], dtype=np.int8).tobytes(),
        np.array([5]+ [0]*N + [0] + [0]*N + [5], dtype=np.int8).tobytes(),
        np.array([50]+ [0]*N + [0] + [0]*N +[50], dtype=np.int8).tobytes(),
        np.array([122]+ [0]*N + [0] + [0]*N + [122], dtype=np.int8).tobytes(),
        np.array([0]+ [0]*N + [60] + [0]*N + [0], dtype=np.int8).tobytes(),
        np.array([5]+ [0]*N + [60] + [0]*N + [5], dtype=np.int8).tobytes(),
        np.array([50]+ [0]*N + [60] + [0]*N + [50], dtype=np.int8).tobytes(),
        np.array([122]+ [0]*N + [60] + [0]*N + [122], dtype=np.int8).tobytes(),
    ]

    di_list = []

    input_queue = deque(inputs)

    global_coverage = coverage.Coverage()

    path_to_binary = env.getpath()

    epoch = 0

    state_count_dict = {}

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    count_edits_were_useful = 0

    were_useful = []

    ind = 0
    while len(input_queue) > 0:
        print('Queue length:', len(input_queue))
        next_input = input_queue.popleft()
    
        print(next_input)#, int(edit_input[0]),int(edit_input[1]))
        _, env_reward, done, info = env.step(next_input)

        total_coverage.add(info['step_coverage'])
        
        edit_was_useful = global_coverage.union(info['step_coverage'])
        
        if edit_was_useful:

            count_edits_were_useful+=1
            were_useful.append(ind)

            print(info['step_coverage'].transitions, next_input[:3])
            print(("STEP: reward={} done={} " +
                "step={}/{}/{}").format(
                    env_reward, done,
                    info['step_coverage'].skip_path_count(),
                    info['step_coverage'].transition_count(),
                    info['step_coverage'].crash_count(),
                ))
        if done:
            env.reset()
            # print("DONE!")
        
        ind += 1

    print("Done")
    print("count useful edits",count_edits_were_useful)
    print("global observation",global_coverage.observation().sum(axis=0).tolist())
    print('global transitions',global_coverage.transitions)

    print([np.frombuffer(inputs[i], dtype=np.uint8) for i in were_useful])

        


        

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
