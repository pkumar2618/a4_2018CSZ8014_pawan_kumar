import os
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import mean_squared_error
import pandas as pd

from utils import frame_reward_to_matrix_XY
from utils import transforming_with_pca
from utils import load_all_dataset_XY
from utils import pickle_store, pickle_load


# path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
# start_path = '/home/pawan/train_dataset/'

# results = []  # used to store the result of async processes
# # # running parallel processes to get images stored as matrix of grayscale pixel.
# dirs = next(os.walk(start_path))[1]
# dirs.sort()
# episodes = dirs[:2]
# pool = mp.Pool(mp.cpu_count())
# result_objects = [pool.apply_async(frame_reward_to_matrix_XY,
#                                    args=(start_path, episode_dir, 0.01)) for episode_dir in episodes]
# pool.close()
# pool.join()


"""
loading the csv for each episode and running the incremental PCA after 
all the data is loaded, the function returs reduced data_set
"""
train_XY_reduced, ipca = transforming_with_pca(root_path = start_path,
                                             top_n_episodes=2, n_components=10, batch_size=10)

# pickling the reduced dataset
pickle_store(train_XY_reduced, root_path=start_path, file_name="pickle_train_XY_reduced")
pickle_store(ipca, root_path=start_path, file_name="pickle_ipca")

# loading the pickled dataset
# train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
# ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")

# # # to measure reconstruction error we need the original dataset
# train_X_reduced = train_XY_reduced.drop('Y', axis = 1).values
# reconstructed_X = ipca.inverse_transform(train_X_reduced)
# train_XY_original = load_all_dataset_XY(root_path=start_path, top_n_episodes=2)
# error_curr = mean_squared_error(reconstructed_X, train_XY_original.drop('Y', axis=1).values)
# print("sequential error using function", error_curr)