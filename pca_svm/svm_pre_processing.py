import os
import sys
import multiprocessing as mp
import pandas as pd
from utils import frame_reward_to_matrix_XY, frame_reward_to_matrix_XY_test_type1, pickle_load, pickle_store
from sklearn.preprocessing import StandardScaler


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
    episodes = dirs[:n_episodes]
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(frame_reward_to_matrix_XY,
                                       args=(start_path, episode_dir, 1)) for episode_dir in episodes]
    pool.close()
    pool.join()

    ### apply pca to raw frame_reward matrix

    # loading the csv for each episode and running the incremental PCA after
    # all the data is loaded, the function returns reduced data_set

    train_XY_reduced, ipca, data_scaler = transforming_with_pca(root_path=start_path,
                                                 top_n_episodes=2, n_components=50, batch_size=None)

    # pickling the reduced dataset
    pickle_store(train_XY_reduced, root_path=start_path, file_name="pickle_train_XY_reduced")
    pickle_store(ipca, root_path=start_path, file_name="pickle_ipca")
    pickle_store(data_scaler, root_path=start_path, file_name="pickle_data_scaler")

if train_test == 1:
    """
    processing the test data requires you to pass train_test = 1

    """
    # path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
    # start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
    # start_path = '/home/pawan/train_dataset/'

    ## no need to run parallel process for this transforming this
    sample_fraction = n_episodes
    frame_reward_to_matrix_XY_test_type1(start_path=start_path, sample_fraction=sample_fraction)

    test_X, test_Y = pickle_load(root_path=start_path, file_name="pickle_raw_test_XY_type1_tuple")
    test_X = test_X.values
    test_Y = test_Y.values



    # transform this data using pca object used during training
    # the function returns reduced data_set
    data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")
    test_X = data_scaler.fit_transform(test_X)

    ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
    test_X_reduced = ipca.fit_transform(test_X)

    # pickling the reduced test dataset by joining XY
    # test_Y = pd.Series(test_Y, name='Y')
    test_XY_reduced = (test_X_reduced, test_Y)

    pickle_store(test_XY_reduced, root_path=start_path, file_name="pickle_test_XY_reduced_tuple")
