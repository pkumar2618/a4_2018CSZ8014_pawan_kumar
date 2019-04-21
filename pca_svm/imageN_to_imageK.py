import os
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# from svmutil import *
import matplotlib.pyplot as plt

from utils import frame_reward_to_matrix_XY, load_all_dataset_XY
from utils import transforming_with_pca, train_XY_to_seq_XY
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
# train_XY_reduced, ipca, data_scaler = transforming_with_pca(root_path = start_path,
#                                              top_n_episodes=2, n_components=10, batch_size=10)

# pickling the reduced dataset
# pickle_store(train_XY_reduced, root_path=start_path, file_name="pickle_train_XY_reduced")
# pickle_store(ipca, root_path=start_path, file_name="pickle_ipca")
# pickle_store(data_scaler, root_path=start_path, file_name="pickle_data_scaler")

# loading the pickled dataset
# train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
# ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
# data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")

# # to measure reconstruction error we need the original dataset
# train_X_reduced = train_XY_reduced.drop('Y', axis = 1).values
# reconstructed_X = ipca.inverse_transform(train_X_reduced)
# train_XY_original = load_all_dataset_XY(root_path=start_path, top_n_episodes=2)
# error_curr = mean_squared_error(reconstructed_X, train_XY_original.drop('Y', axis=1).values)
# print("sequential error using function", error_curr)

"""
Sequence sampling
"""
# the sequence will be known by the first frame in the sequence
# loading the pickled dataset
train_XY = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
# ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")


# the train_XY_reduced stores all the episode, separated by -1 in 'Y' label.
#
seq_train_XY = train_XY_to_seq_XY(train_XY, y_at_start_of_episode=-1,
                       safe_to_csv=True, root_path=start_path, file_name="seq_train_XY")
print(seq_train_XY.shape)


"""
    binary classifier using linear as well as gaussian SVM
# """
#         m = svm_train(train_Y, train_X_normed, '-t 0')
#         # m = svm_train(train_Y, train_X_normed, '-g 0.05 -t 2')
#
#         svm_predict(test_Y, test_X_normed, m)
#         p_label, p_acc, p_val, = svm_predict(dev_Y, dev_X_normed, m)
