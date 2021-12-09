import gym
import gym_fuzz1ng.coverage as coverage
import numpy as np
from collections import deque
import helper
from gym_fuzz1ng.envs.fuzz_simple_bits_env import FuzzSimpleBitsEnvSmall
import time
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
import wandb

wandb.init(project="CS263", entity="cs263")

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
program_name = "SimpleBits-v0-"
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
def main(actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=40, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):


    # PPO setup - copied from spinup ppo

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    # env = env_fn()


    # observation_space = np.arange(DUMMY)
    observation_space = np.arange(NSTATES)
    action_space = gym.spaces.Discrete(N_EDIT_TYPES)

    obs_dim = observation_space.shape
    act_dim = action_space.shape

    # Create actor-critic module
    ac = actor_critic(observation_space, action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = my_ppo.PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

        # wandb.log('LossPi':pi_l_old, 'LossV':v_l_old,
        #              'KL':kl, 'Entropy':ent, 'ClipFrac':cf,
        #              'DeltaLossPi':(loss_pi.item() - pi_l_old),
        #              'DeltaLossV':(loss_v.item() - v_l_old))

        # wandb.log({"loss": loss})

    
    
    
    # Prepare for interaction with environment
    start_time = time.time()
    
    ##################################
    # ------------------------------ #
    ##################################


    ####################
    # main fuzzing loop
    ####################

    # env = gym.make('FuzzSimpleBits-v0')
    env = FuzzSimpleBitsEnvSmall()
    print("dict_size={} eof={}".format(env.dict_size(), env.eof()))

    env.reset()
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

    while len(input_queue) > 0:
        print('Queue length:', len(input_queue))
        next_input = input_queue.popleft()

        input_buff = bytearray(next_input)
        out_buff = bytearray(next_input)
        s = time.time()
        for edit_input in helper.deterministic_edits(input_buff, out_buff):
            newt = time.time()
            print("loop time:",newt-s)
            s=newt
            # s = time.time()
            # obs = get_observation(path_to_binary, edit_input)
            # obs = get_observation(edit_input)
            # s1 = time.time()
            # print("time",s1-s)
            # print('state',obs)

            # TODO: find a way to increase the transition map size from 256 to other--
            # probably just increase MAPSIZE (sp?) param

            print(edit_input[:5])#, int(edit_input[0]),int(edit_input[1]))
            obs, reward, done, info = env.step(edit_input)
            # print('before',total_coverage.transitions)
            total_coverage.add(info['step_coverage'])
            # print('after',total_coverage.transitions)
            # print()
            
            # TODO: double check this is the way global coverage is tracked in AFL
            edit_was_useful = global_coverage.union(info['step_coverage'])
            
            if edit_was_useful:
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

            # print(("STEP: reward={} done={} " +
            #     "step={}/{}/{} total={}/{}/{} " +
            #     "sum={}/{}/{} action={}").format(
            #         reward, done,
            #         info['step_coverage'].skip_path_count(),
            #         info['step_coverage'].transition_count(),
            #         info['step_coverage'].crash_count(),
            #         info['total_coverage'].skip_path_count(),
            #         info['total_coverage'].transition_count(),
            #         info['total_coverage'].crash_count(),
            #         total_coverage.skip_path_count(),
            #         total_coverage.transition_count(),
            #         total_coverage.crash_count(),
            #         edit_input[:13],
            #     ))
            if done:
                env.reset()
                # print("DONE!")
        
        # TODO: implement RL part here. Once deterministic edits are done for a file
        # we want the RL agent to take random edit actions on the file, testing their goodness
        # based on the transition map that gets returned.
        #
        # The hope is that, for a given program, over time we can learn to do better than random,
        # which is all that AFL does at this stage.
        #
        # Should look something like `for i in range(EPOCH_LENGTH): ... do RL exploration`

        current_input = bytearray(next_input)
        output_buff = bytearray(next_input)


        # TODO: implement get_observation()
        # state space should eventually be both file and and program state
        # obs, ep_ret, ep_len = get_observation(current_input), 0, 0
        obs = get_observation(path_to_binary, edit_input)
        ep_ret, ep_len = 0, 0
        # Main loop: collect experience in env and update/log each epoch
        # for epoch in range(epochs):
        for t in range(local_steps_per_epoch):

            # print("Getting RL action for round %s:"%t)
            a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            # ignore the action for now and just pass a dummy edit
            edit_input = helper.random_edits(current_input, a)

            
            _, env_reward, done, info = env.step(edit_input)

            # next_obs = get_observation(edit_input)
            next_obs = get_observation(path_to_binary, edit_input)
            current_input = edit_input

            # dont use env_reward, since its just a sum of the transitions
            # what we actually want is a count-based state exploration reward
            # get state hash
            h.update(obs)
            hsh = h.digest()
            h.reset()
            if hsh in state_count_dict:
                state_count_dict[hsh] += 1
            else:
                state_count_dict[hsh] = 1

            reward = 1/np.sqrt(state_count_dict[hsh])
            print("obs")
            print(obs)
            print('hash', hsh)
            print('count',state_count_dict[hsh])
            print('reward',reward)

            ep_ret += reward
            ep_len += 1

            # save and log
            buf.store(obs, a, reward, v, logp)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t==local_steps_per_epoch-1

            # if terminal or epoch_ended:
            if epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                print('finishing path')
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    wandb.log({"EpRet":ep_ret, "EpLen":ep_len})

                # don't need this unless we want to train on a given files for mulitple epochs
                # o, ep_ret, ep_len = env.reset(), 0, 0


            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)


        logger.dump_tabular()

        epoch += 1

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
