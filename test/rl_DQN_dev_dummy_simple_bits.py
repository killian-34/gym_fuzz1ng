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
    # s2 = time.time()
    # print("next times...")
    # print(s1-s)
    # print(s2-s1)

    return state



# arguments copied from spinup PPO
def main(n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):


    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
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

    agent = Agent(state_size=len(run_strace.syscall_map), action_size=32*8, device=device, lr=LR, 
            buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, update_every=UPDATE_EVERY,
             gamma=GAMMA, tau=TAU, seed=0)

    total_coverage = coverage.Coverage()

    inputs = [
        np.array([0, 256] + [0] * 30, dtype=np.int8).tobytes()
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

    while len(input_queue) > 0:
        print('Queue length:', len(input_queue))
        next_input = input_queue.popleft()

        input_buff = bytearray(next_input)
        out_buff = bytearray(next_input)
        s = time.time()

        state = get_observation(path_to_binary, input_buff)
        for edit_input, action in helper.deterministic_edits_2(input_buff, out_buff):
            # newt = time.time()
            # print("loop time:",newt-s)
            # s=newt
            # s = time.time()
            # obs = get_observation(path_to_binary, edit_input)
            # obs = get_observation(edit_input)
            # s1 = time.time()
            # print("time",s1-s)
            # print('state',obs)

            # oldTODO: find a way to increase the transition map size from 256 to other--
            # probably just increase MAPSIZE (sp?) param


            next_state = get_observation(path_to_binary, edit_input)
            
            print(edit_input[:5])#, int(edit_input[0]),int(edit_input[1]))
            _, env_reward, done, info = env.step(edit_input)
            reward = 0

            # don't actually want this for deterministic edits... these are all one step transitions
            # state = next_state

            # print('before',total_coverage.transitions)
            total_coverage.add(info['step_coverage'])
            # print('after',total_coverage.transitions)
            # print()
            
            # oldTODO: double check this is the way global coverage is tracked in AFL
            edit_was_useful = global_coverage.union(info['step_coverage'])
            
            if edit_was_useful:
                
                # always give reward 1 for useful edits during deterministic phase
                reward = 1

                count_edits_were_useful+=1
                input_queue.append(edit_input)
                print("adding input",edit_input)


                print(info['step_coverage'].transitions, edit_input[:4])
                print(("STEP: reward={} done={} " +
                    "step={}/{}/{}").format(
                        reward, done,
                        info['step_coverage'].skip_path_count(),
                        info['step_coverage'].transition_count(),
                        info['step_coverage'].crash_count(),
                    ))
            if done:
                env.reset()
                # print("DONE!")


            # get experiences for deterministic edits as well
            done=False
            agent.memory.add(state, action, reward, next_state, done)

            if edit_was_useful:
                agent.memory.get_good_experience(shorter_len=1)
                print()
                print('good experiences on action',action)
                print(agent.memory.good_experiences)
                # 1/0


            global_transition_count = global_coverage.transition_count()
            # print(global_transition_count)

            # wandb.log({})
            # wandb.log({'global_coverage':global_transition_count, 'useful_edits':count_edits_were_useful})
        
        GOOD_EXPERIENCE_LOOP_COUNTER = 5
        EPOCHS_PER_INPUT = 5
        for i in range(GOOD_EXPERIENCE_LOOP_COUNTER):
            agent.learn_good_experiences()

        for i in range(EPOCHS_PER_INPUT):
            print('epoch',i)
            env.reset()
            current_input = bytearray(next_input)
            # output_buff = bytearray(next_input)

            state = get_observation(path_to_binary, current_input)
            score = 0
            for t in range(max_t):
                edit_action = agent.act(state, eps)
                edit_input = helper.deterministic_bit_edit(current_input, edit_action)

                # run the file through afl, get the transition diagram from info
                _, env_reward, done, info = env.step(edit_input)

                edit_was_useful = global_coverage.union(info['step_coverage'])
                
                if edit_was_useful:
                    count_edits_were_useful+=1

                    input_queue.append(edit_input)
                    print("adding input",edit_input)


                    print(info['step_coverage'].transitions, edit_input[:4])
                    print(("STEP: reward={} done={} " +
                        "step={}/{}/{}").format(
                            reward, done,
                            info['step_coverage'].skip_path_count(),
                            info['step_coverage'].transition_count(),
                            info['step_coverage'].crash_count(),
                        ))
                if done:
                    env.reset()

                # wandb.log({'useful_edits':count_edits_were_useful})

                # get the syscall counts that we use for states
                next_state = get_observation(path_to_binary, edit_input)
                
                #copy the new edited file to the current input
                current_input = edit_input
                
                # dont use env_reward, since its just a sum of the transitions
                # what we actually want is a count-based state exploration reward
                # use the next_state for reward, could use state or even info['step_coverage'].observation
                h.update(info['step_coverage'].observation())
                hsh = h.digest()
                h.reset()
                if hsh in state_count_dict:
                    state_count_dict[hsh] += 1
                else:
                    state_count_dict[hsh] = 1

                reward = 1/np.sqrt(state_count_dict[hsh])

                # print("obs")
                # print(next_state)
                # print('hash', hsh)
                # print('count',state_count_dict[hsh])
                # print('reward',reward)
                # print('action',edit_action)
                # print('eps',eps)
                # print('current_input',current_input)
                # print('global map',global_coverage.transitions)
                eps = max(eps_end, eps_decay*eps) # decrease epsilon

                # wandb.log({'score':score, 'global_coverage':global_coverage.transition_count()})
                done = False
                agent.step(state, edit_action, reward, next_state, done)
                state = next_state
                score += reward

                if edit_was_useful:
                    agent.memory.get_good_experience()
            

            for i in range(GOOD_EXPERIENCE_LOOP_COUNTER):
                agent.learn_good_experiences()

            print('num good epxeriences so far',len(agent.memory.good_experiences))
            print('num useful edits:',count_edits_were_useful)
            

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}\tSteps: {}'.format(epoch, np.mean(scores_window),t))



        epoch+=1



        # if i_episode % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break

        # return scores
        # TODO: implement RL part here. Once deterministic edits are done for a file
        # we want the RL agent to take random edit actions on the file, testing their goodness
        # based on the transition map that gets returned.
        #
        # The hope is that, for a given program, over time we can learn to do better than random,
        # which is all that AFL does at this stage.
        #
        # Should look something like `for i in range(EPOCH_LENGTH): ... do RL exploration`



        # TODO: also implement the AFL version of the random edits sequence, so we can roughly
        # compare against it. Shouldn't be too hard. Important thing will be to make sure 
        # that we execute approximately the same number of total edits as the AFL random
        # during comparisons.

        # Note that we won't create a perfect copy of AFL random, becuase they do some 
        # heuristic fine-tuning of how long to run the random havoc stage based
        # on the performance of the test input in question
        

        # DONE: Build another loop to train on the RL data that was collected. We will 
        # need to build some infrastructure for tracking actions/states/rewards
        # but that should be boilerplate stuff that we can copy from RMABPPO or other
        # openAI gym public repos.


        # Other TODO s:
        # - Experiment with different state spaces
        # - Experiment with a few network architectures/training methods
        # - Experiment with different action spaces
        # - Experiment with choices for reward -- how to get non-zero rewards more often
        #     - for now, I think we can just make the action space [flip 1 bit, flip 2 bits, flip 4 bits, ...]
        #       then it randomly carries out that action somewhere in the file. Just keep it simple.
        #       and as analogous to the AFL random edits as we can.
        # - Experiment with a real program like libpng or something
        # - Experiment with training on libpng v1 and test on libpng v2 or something along those lines
        #     - doing well here would be a win
        # - All the above probably gets us the grade we need already, but feel free to add other things to test.
        # 



        


        

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
