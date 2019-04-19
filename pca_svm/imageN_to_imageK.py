import os
import multiprocessing as mp

from utils import frame_reward_to_matrix_XY
results = []
# path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
# start_path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/'

dirs = next(os.walk(start_path))[1]
# path = os.path.join(start_path, dir)
dirs.sort()
episodes = dirs[:4]
pool = mp.Pool(mp.cpu_count())
result_objects = [pool.apply_async(frame_reward_to_matrix_XY, args=(start_path, episode_dir, 0.5)) for episode_dir in episodes]
pool.close()
pool.join()
