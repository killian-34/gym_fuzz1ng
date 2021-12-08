import gym
import gym_fuzz1ng.coverage as coverage
import numpy as np
from collections import deque
import helper
from gym_fuzz1ng.envs.fuzz_simple_bits_env import FuzzSimpleBitsEnvSmall

def main():
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

    while len(input_queue) > 0:
        print('Queue length:', len(input_queue))
        next_input = input_queue.popleft()

        input_buff = bytearray(next_input)
        out_buff = bytearray(next_input)
        for edit_input in helper.deterministic_edits(input_buff, out_buff):

            # TODO: asynchronously pass edit_input to the strace call to get back a state
            # to use in RL

            # TODO: find a way to increase the transition map size from 256 to other--
            # probably just increase MAPSIZE (sp?) param

            # print(edit_input[:5])#, int(edit_input[0]),int(edit_input[1]))
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

        # TODO: also implement the AFL version of the random edits sequence, so we can roughly
        # compare against it. Shouldn't be too hard. Important thing will be to make sure 
        # that we execute approximately the same number of total edits as the AFL random
        # during comparisons.
        

        # TODO: Build another loop to train on the RL data that was collected. We will 
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
