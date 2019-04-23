import os
import sys
import multiprocessing as mp
from cnn_utils import frame_reward_to_seqs_stack_XY

# path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
# start_path = '/home/pawan/train_dataset/'


results = []  # used to store the result of async processes
# # running parallel processes to get images stored as matrix of grayscale pixel.
dirs = next(os.walk(start_path))[1]
dirs.sort()
episodes = dirs[:2]
pool = mp.Pool(mp.cpu_count())
result_objects = [pool.apply_async(frame_reward_to_seqs_stack_XY,
                                   args=(start_path, episode_dir, 1)) for episode_dir in episodes]
pool.close()
pool.join()
