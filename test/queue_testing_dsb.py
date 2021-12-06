import gym
import gym_fuzz1ng.coverage as coverage
import numpy as np
from collections import deque
import helper

def main():
    env = gym.make('FuzzSimpleBits-v0')
    print("dict_size={} eof={}".format(env.dict_size(), env.eof()))

    env.reset()
    total_coverage = coverage.Coverage()

    inputs = [
        [1, 256] + [0] * 62,
        [256] + [0] * 63,
        [1, 1, 256] + [0] * 61,
        [1, 1, 256] + [0] * 61,

        [1, 1, 256] + [0] * 61,
        [12, 256] + [0] * 62,
        [12, 7, 256] + [0] * 61,

        [1, 256] + [0] * 62,
        [1, 2, 256] + [0] * 61,
        [1, 2, 3, 256] + [0] * 60,
        [1, 2, 3, 4, 256] + [0] * 59,
        [1, 2, 3, 4, 5, 256] + [0] * 58,
        [1, 1, 256] + [0] * 61,

        [1, 1, 256] + [0] * 61,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 256] + [0] * 53,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 256] + [0] * 52,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 0, 256] + [0] * 51,
        [1, 1, 256] + [0] * 61,
    ]


    inputs = [
        [0, 256] + [0] * 62,
    ]

    di_list = []

    input_queue = deque(inputs)

    global_coverage = coverage.Coverage()

    while len(input_queue) > 0:
        print('Queue length:', len(input_queue))
        next_input = input_queue.popleft()

        for edit_input in helper.deterministic_edits(next_input):
            print(edit_input[:10])
            obs, reward, done, info = env.step(np.array(next_input))
            # print('before',total_coverage.transitions)
            total_coverage.add(info['step_coverage'])
            # print('after',total_coverage.transitions)
            # print()
            edit_was_useful = global_coverage.union(info['step_coverage'])
            
            if edit_was_useful:
                input_queue.append(edit_input)
                print("adding input",edit_input)


            print(info['step_coverage'].transitions, edit_input[:4])
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
