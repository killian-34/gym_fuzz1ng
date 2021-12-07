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
            # print(edit_input[:5])#, int(edit_input[0]),int(edit_input[1]))
            obs, reward, done, info = env.step(edit_input)
            # print('before',total_coverage.transitions)
            total_coverage.add(info['step_coverage'])
            # print('after',total_coverage.transitions)
            # print()
            edit_was_useful = global_coverage.union(info['step_coverage'])
            
            if edit_was_useful:
                input_queue.append(edit_input)
                print("adding input",edit_input)


            # print(info['step_coverage'].transitions, edit_input[:4])
            # print(("STEP: reward={} done={} " +
            #     "step={}/{}/{}").format(
            #         reward, done,
            #         info['step_coverage'].skip_path_count(),
            #         info['step_coverage'].transition_count(),
            #         info['step_coverage'].crash_count(),
            #     ))

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

        


        

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
