import os
import re
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA


def frame_reward_to_matrix_XY(start_path, episode_dir, sample_fraction=0.1):
    path = os.path.join(start_path, episode_dir)
    files = os.listdir(path)
    files.sort()
    # only take .png files
    files_png = []
    for x in files:
        if x.endswith(".png"):
            files_png.append(x)
    n_files_contained = len(files_png)

    # selecting only a fraciton of files useful during debugging
    t_frac = sample_fraction
    test_population = int(t_frac*n_files_contained)
    data_set_indices = [x for x in range(test_population)]

    # when selecting random files
    # data_set_indices = random.sample([x for x in range(n_files_contained)], test_population)
    # data_set_indices = [x for x in range(n_files_contained)]

    ### finding the rewards
    rewards = pd.read_csv(os.path.join(path, "rew.csv"), header=None)
    rewards = rewards.values
    rewards = rewards.astype('f')
    # the sequence will be identified by the last frame in the sequence,
    # here however we will only pair frames and their rewards using pandas Dataframe, with
    # the reward of the first frame being None.
    # reading all the images and storing them into a matrix of mxn dimension
    # the rows(m) are different different frames and columns are features obtained by flattening a single frame

    frame_rewards = np.array([])
    frame_stack = np.array([], dtype='f')

    for i in data_set_indices:
        frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
        frame_id = int(frame_id[0])
        if frame_id == 0:
            frame_rewards = np.append(frame_rewards, -1)
        else:
            frame_rewards = np.append(frame_rewards, rewards[frame_id - 1])  # reward for the is stored in rew.csv

        frame = cv2.imread(os.path.join(path, files_png[i]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('grayed image', frame)
        # cv2.waitKey(0)
        frame = np.array(frame, dtype='f').flatten() / 255
        frame_stack = np.append(frame_stack, frame)

    frame_size = frame.size # getting the shape of the last frame
    frame_stack = frame_stack.reshape(-1, frame_size)
    frame_rewards = frame_rewards.reshape(-1, 1).astype(np.float32)

    # Creating a dataframe
    frame_stack = pd.DataFrame(frame_stack)
    frame_rewards = pd.Series(frame_rewards.flatten(), name="Y")

    train_XY = pd.concat((frame_stack, frame_rewards), axis=1)
    train_XY.to_csv(os.path.join(start_path, "train_XY"+episode_dir+".csv"), index=False)
    # print(frame_stack.shape)


def transforming_with_pca(root_path, top_n_episodes, n_components =10, batch_size=100):
    # # loading the csv for each episode and running the PCA
    files = next(os.walk(root_path))[2]
    files.sort()
    files_csv = []
    for x in files:
        if x.endswith(".csv"):
            files_csv.append(x)

    n_components = n_components
    episodes = files_csv[:top_n_episodes]
    all_train_XY = pd.DataFrame()
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for train_XY_csv in episodes:
        path = os.path.join(root_path, train_XY_csv)
        train_XY = pd.read_csv(path)
        all_train_XY = pd.concat((all_train_XY, train_XY), axis=0)

    # separating features and rewards
    # print('reading complete')
    train_X = all_train_XY.drop('Y', axis=1).values
    train_Y = all_train_XY['Y'].values

    # scalind data to zero mean, we won't use unit variance here
    data_scaler = StandardScaler(with_std=False)
    train_X = data_scaler.fit_transform(train_X)

    # applying pca on the scaled data
    ipca.partial_fit(train_X)
    train_X_reduced = ipca.transform(train_X)
    train_X_reduced = pd.DataFrame(train_X_reduced)
    train_Y = pd.Series(train_Y, name='Y')
    train_XY = pd.concat((train_X_reduced, train_Y), axis=1)
    return train_XY, ipca


def load_all_dataset_XY(root_path, top_n_episodes):
    # # loading the csv for each episode and running the PCA
    files = next(os.walk(root_path))[2]
    files.sort()
    files_csv = []
    for x in files:
        if x.endswith(".csv"):
            files_csv.append(x)

    episodes = files_csv[:top_n_episodes]
    all_train_XY = pd.DataFrame()

    for train_XY_csv in episodes:
        path = os.path.join(root_path, train_XY_csv)
        train_XY = pd.read_csv(path)
        all_train_XY = pd.concat((all_train_XY, train_XY), axis=0)

    return all_train_XY


def pickle_store(data, root_path, file_name):
    # using  binary mode
    path = os.path.join(root_path, file_name)
    file_handle = open(path, 'wb')

    pickle.dump(data, file_handle)
    file_handle.close()


def pickle_load(root_path, file_name):
    # reading using binary mode
    path = os.path.join(root_path, file_name)
    file_handle = open(path, 'rb')
    data = pickle.load(file_handle)
    file_handle.close()
    return data

