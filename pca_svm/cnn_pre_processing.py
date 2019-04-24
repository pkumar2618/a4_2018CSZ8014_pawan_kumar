import os
import sys
import multiprocessing as mp
from cnn_utils import frame_reward_to_seqs_stack_XY, frame_reward_to_seqs_stack_XY_test1


train_test = int(sys.argv[1])
start_path = sys.argv[2]
n_episodes = float(sys.argv[3])

"""
processing the train data requires you to provide train_test = 0
start_path = path of the parent directory which has all the episodes
"""
if train_test == 0:

    # path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
    # start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
    # start_path = '/home/pawan/train_dataset/'

    results = []  # used to store the result of async processes
    # # running parallel processes to get images stored as matrix of grayscale pixel.
    dirs = next(os.walk(start_path))[1]
    dirs.sort()
    n_episodes = int(n_episodes)
    episodes = dirs[:n_episodes]
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(frame_reward_to_seqs_stack_XY,
                                       args=(start_path, episode_dir, 1)) for episode_dir in episodes]
    pool.close()
    pool.join()

if train_test == 1:
    """
    processing the test data requires you to pass train_test = 1

    """
    # results = []  # used to store the result of async processes
    # # # running parallel processes to get images stored as matrix of grayscale pixel.
    # dirs = next(os.walk(start_path))[1]
    # dirs.sort()
    # episodes = dirs[:n_episodes]
    # pool = mp.Pool(mp.cpu_count())
    #
    # result_objects = [pool.apply_async(frame_reward_to_seqs_stack_XY_test1,
    #                                    args=(start_path, episode_dir, 1)) for episode_dir in episodes]
    # pool.close()
    # pool.join()
    frame_reward_to_seqs_stack_XY_test1(start_path=start_path, sample_fraction=n_episodes)
