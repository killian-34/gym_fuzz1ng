import gym
import gym_fuzz1ng.coverage as coverage
import numpy as np


def main():
    env = gym.make('FuzzSimpleBits-v0')
    print("dict_size={} eof={}".format(env.dict_size(), env.eof()))

    env.reset()
    c = coverage.Coverage()

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
        [1, 256] + [0] * 62,
        [1, 256] + [0] * 62,
        [1, 256] + [0] * 62,
        [1, 256] + [0] * 62,
    ]

    di_list = []
    for i in inputs:
        obs, reward, done, info = env.step(np.array(i))
        c.add(info['step_coverage'])
        # TODO: verify if the transition dict always comes back the same
        # print(info['total_coverage'].transitions)
        # import pdb; pdb.set_trace()
        di_list.append(info['step_coverage'])


        print(("STEP: reward={} done={} " +
               "step={}/{}/{} total={}/{}/{} " +
               "sum={}/{}/{} action={}").format(
                   reward, done,
                   info['step_coverage'].skip_path_count(),
                   info['step_coverage'].transition_count(),
                   info['step_coverage'].crash_count(),
                   info['total_coverage'].skip_path_count(),
                   info['total_coverage'].transition_count(),
                   info['total_coverage'].crash_count(),
                   c.skip_path_count(),
                   c.transition_count(),
                   c.crash_count(),
                   i[:13],
               ))
        if done:
            env.reset()
            print("DONE!")

    obs = [di_list[i].observation() for i in range(len(di_list))]
    all_equal = True
    for i in range(len(di_list)):
        for j in range(i+1, len(di_list)):
            all_equal &= np.array_equal(obs[i], obs[j])
    print(all_equal)

    print("unique values")
    for i in range(len(di_list)):
        print(set(obs[i].reshape(-1)))

    print("unique positions")
    for i in range(len(di_list)):
        print(np.argwhere(obs[i].reshape(-1) > 0).reshape(-1))


    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
