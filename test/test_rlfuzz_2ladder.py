import sys
import time
from collections import deque

import gym
import gym_fuzz1ng.coverage as coverage
# RL imports
import numpy as np
import torch
import xxhash
from fs.tempfs import TempFS
from gym_fuzz1ng.utils import run_strace

import helper
from dqn import Agent

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

    with open(savepath, "wb") as binary_file:
        # Write bytes to file
        binary_file.write(a)

    # store strace output and then get the state from it
    strace_out = run_strace.run_strace(path_to_binary, savepath)
    
    state = run_strace.strace_state(strace_out)

    # just keep read and write!
    state = state[:2]

    return state



# arguments copied from spinup my_dqn
def main(n_episodes=2000, max_t=10, eps_start=1.0, eps_end=0.01, eps_decay=0.995, trial=0):


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
    INPUT_SIZE = int(sys.argv[2])
    # NACTIONS = INPUT_SIZE*8
    NACTIONS = INPUT_SIZE
    
    
    # Prepare for interaction with environment
    start_time = time.time()
    
    ##################################
    # ------------------------------ #
    ##################################


    ####################
    # main fuzzing loop
    ####################

    env = gym.make('FuzzSimple2LadderSyscall-v0')
    print("dict_size={} eof={}".format(env.dict_size(), env.eof()))
    env.reset()

    seed = np.random.randint(10000)

    agent = Agent(state_size=NSTATES, action_size=NACTIONS, device=device, lr=LR, 
            buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, update_every=UPDATE_EVERY,
             gamma=GAMMA, tau=TAU, seed=seed)

    fname='model/2ladder_agent_trained_%s_%s.pickle'%(trial,INPUT_SIZE)
    agent.load_self(fname)


    total_coverage = coverage.Coverage()

    inputs = [
        np.array([0] * (INPUT_SIZE//2) + [0] + [0] * (INPUT_SIZE//2), dtype=np.int8).tobytes(),
        np.array([0] * (INPUT_SIZE//2) + [60] + [0] * (INPUT_SIZE//2), dtype=np.int8).tobytes(),
    ]


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

    time_to_four = 0
    time_to_five = 0
    time_to_six = 0
    time_to_seven = 0
    time_to_eight = 0


    starttime = time.time()
    EPOCHS_PER_INPUT = int(sys.argv[3])
    total_edits = 0

    edits_until_x = np.zeros(5)
    unique_transitions_found = []
    for epoch_i in range(EPOCHS_PER_INPUT):
        eps=0.1

        for next_input in list(input_queue):
            print('epoch',epoch_i)
            env.reset()
            current_input = bytearray(next_input)

            state = get_observation(path_to_binary, current_input)
            score = 0
            for t in range(max_t):
                edit_action = agent.act(state, eps)
                edit_input = helper.network_edit(current_input, edit_action)
                total_edits+=1

                # run the file through afl, get the transition diagram from info
                _, env_reward, done, info = env.step(edit_input)

                edit_was_useful = global_coverage.union(info['step_coverage'])
                
                if edit_was_useful:
                    count_edits_were_useful+=1

                    # input_queue.append(edit_input)
                    print("adding input",edit_input)

                    were_useful.append(np.frombuffer(bytearray(edit_input), dtype=np.uint8))

                    if count_edits_were_useful == 4:
                        stoptime = time.time()
                        time_to_four = stoptime - starttime
                        print("Found 4 edits!")
                        print("Took: %s seconds"%time_to_four)
                        edits_until_x[0] = total_edits

                    if count_edits_were_useful == 5:
                        stoptime = time.time()
                        time_to_five = stoptime - starttime
                        print("Found 5 edits!")
                        print("Took: %s seconds"%time_to_five)
                        edits_until_x[1] = total_edits

                    if count_edits_were_useful == 6:
                        stoptime = time.time()
                        time_to_six = stoptime - starttime
                        print("Found 6 edits!")
                        print("Took: %s seconds"%time_to_six)
                        edits_until_x[2] = total_edits

                    if count_edits_were_useful == 7:
                        stoptime = time.time()
                        time_to_seven = stoptime - starttime
                        print("Found 7 edits!")
                        print("Took: %s seconds"%time_to_seven)
                        edits_until_x[3] = total_edits
                        # exit()

                    if count_edits_were_useful == 8:
                        stoptime = time.time()
                        time_to_eight = stoptime - starttime
                        print("Found 8 edits!")
                        print("Took: %s seconds"%time_to_eight)
                        edits_until_x[4] = total_edits


                    print(info['step_coverage'].transitions, edit_input[:4])
                    print(("STEP: reward={} done={} " +
                        "step={}/{}/{}").format(
                            env_reward, done,
                            info['step_coverage'].skip_path_count(),
                            info['step_coverage'].transition_count(),
                            info['step_coverage'].crash_count(),
                        ))
                if done:
                    env.reset()

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

                if edit_was_useful:
                    print("obs")
                    print(next_state)
                    print('hash', hsh)
                    print('count',state_count_dict[hsh])
                    print('reward',reward)
                    print('action',edit_action)
                    print('eps',eps)
                    print('current_input',current_input)
                    print('current_input',np.frombuffer(current_input,dtype=np.uint8))
                    print('global map',global_coverage.transitions)

                unique_transitions_found.append(count_edits_were_useful)

                done = False
                # agent.step(state, edit_action, reward, next_state, done)
                state = next_state
                score += reward

                if edit_was_useful:
                    agent.memory.get_good_experience()

            print("obs")
            print(next_state)
            print('hash', hsh)
            print('count',state_count_dict[hsh])
            print('reward',reward)
            print('action',edit_action)
            print('eps',eps)
            print('current_input',current_input)
            print('current_input',np.frombuffer(current_input,dtype=np.uint8))
            print('global map',global_coverage.transitions)
            

            # for i in range(GOOD_EXPERIENCE_LOOP_COUNTER):
            #     agent.learn_good_experiences()

            

            print('num good epxeriences so far',len(agent.memory.good_experiences))
            print('num useful edits:',count_edits_were_useful)
            print(were_useful)
            

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}\tSteps: {}'.format(epoch_i, np.mean(scores_window),t))



        epoch+=1

    print('time to four',time_to_four)
    print('time to five',time_to_five)
    print('time to six',time_to_six)
    print('time to seven',time_to_seven)
    print('time to eight',time_to_eight)


    times = [[time_to_four, time_to_five, time_to_six, time_to_seven, time_to_eight]]

    edits_until_x = [edits_until_x]

    print(len(unique_transitions_found))
    print(unique_transitions_found)

    
    import pandas as pd 
    pd.DataFrame(times,columns=['tt4','tt5','tt6','tt7','tt8s']).to_csv('csv/test_times_%s_%s.csv'%(trial, INPUT_SIZE),index=False)

    pd.DataFrame(edits_until_x,columns=['et4','et5','et6','et7','et8s']).to_csv('csv/test_edits_until_x_%s_%s.csv'%(trial, INPUT_SIZE),index=False)

    pd.DataFrame([unique_transitions_found]).to_csv('csv/test_transitions_per_edit_%s_%s.csv'%(trial, INPUT_SIZE),index=False)



        # if i_episode % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break


if __name__ == "__main__":
    main(trial=sys.argv[1])