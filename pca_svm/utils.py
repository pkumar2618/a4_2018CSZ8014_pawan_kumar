import os
import re
import random
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
SEED = 0

def frame_reward_to_matrix_XY(start_path, episode_dir, sample_fraction=0.1):
    # # # this function may be used to run parallel process to get images stored as matrix of grayscale pixel.
    """
    the first frame of an episode has its rewards set as -1
    :param start_path:
    :param episode_dir:
    :param sample_fraction: how much of the frames in the episodes has to be read, another way to randomly select
    :return:
    """
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

def frame_reward_to_matrix_XY_test_type1(start_path, sample_fraction=1):
    """
    type 1 assumes that each episodes has only 5 frames.
    :param start_path:
    :param sample_fraction: how much of the frames in the episodes has to be read, another way to randomly select
    :return:
    """
    dirs = next(os.walk(start_path))[1]
    files = next(os.walk(start_path))[2]
    dirs.sort()

    ### finding the rewards, assuming the reward.csv is in the start_path
    pattern = re.compile("rew[a-zA-Z]+\\.csv")
    for x in files:
        if pattern.match(x):
            rewards = pd.read_csv(os.path.join(start_path, x), header=None, index_col=0)
            rewards = rewards.values
            rewards = rewards.astype('f')
            break

    # selecting only a fraciton of episodes useful during debugging
    m_dirs_contained = len(dirs)
    t_frac = sample_fraction
    test_population = int(t_frac * m_dirs_contained)

    # when selecting random dirs
    random.seed(SEED)
    episodes_indices = random.sample([x for x in range(m_dirs_contained)], test_population, )
    # not random
    # episodes_indices = [x for x in range(test_population)]

    # the sequence will be identified by the dir_id
    # reading all the episode and creating sequence from it
    # the rows(m) are different frames and columns are features obtained by flattening a single frame
    # we will only return frames stacks in order from each episodes, so that it can be scaled using
    # data_scaler used for training data during transforming in to PCA
    frame_rewards = np.array([])
    frame_stack = np.array([], dtype='f')

    for d in episodes_indices:
        dir_id = int(dirs[d])
        path = os.path.join(start_path, dirs[d])
        files = os.listdir(path)
        files_png = []
        # only take .png files
        for x in files:
            if x.endswith(".png"):
                files_png.append(x)

        n_files_contained = len(files_png)
        files_png.sort() # files_png might have files stored in a random order
        for i in range(0, n_files_contained, 5):
            frame_id = re.split("[.]", files_png[i])
            frame_id = int(frame_id[0])
            if len(frame_rewards) == 0:
                frame_rewards = rewards[dir_id]
            else:
                frame_rewards = np.append(frame_rewards, rewards[dir_id])  # reward from same episode considred same

            #stacking up images taking 5 in order
            for j in range(5):
                frame = cv2.imread(os.path.join(path, files_png[i+j]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.array(frame, dtype='f').flatten() / 255
                frame = frame.reshape(1, -1)
                # frame = np.stack(frame, axis=0)
                # cv2.imshow('grayed image', frame)
                # cv2.waitKey(0)
                if frame_stack.size == 0:
                    frame_stack = frame
                else:
                    frame_stack = np.concatenate((frame_stack, frame), axis=0)

    frame_rewards = frame_rewards.reshape(-1, 1).astype(np.float32)

    # Creating a dataframe
    frame_stack = pd.DataFrame(frame_stack)
    frame_rewards = pd.Series(frame_rewards.flatten(), name="Y")

    test_XY = (frame_stack, frame_rewards) # pass as tuple
    pickle_store(test_XY, start_path, "pickle_raw_test_XY_type1_tuple")
    # train_XY.to_csv(os.path.join(start_path, "raw_test_XY.csv"), index=False)
    # print(frame_stack.shape)

def transforming_with_pca(root_path, top_n_episodes, n_components =10, batch_size=100):
    # # loading the csv for each episode and running the PCA
    """
    loading the csv for each episode having images stored in grayscale, and running the incremental PCA after
    all the data is loaded, the function returns reduced data_set
    """
    files = next(os.walk(root_path))[2]
    files.sort()
    files_csv = []
    pattern = re.compile("train_XY[\d]+\.csv")
    for x in files:
        if pattern.match(x):
            files_csv.append(x)

    n_components = n_components
    episodes = files_csv[:top_n_episodes]
    all_train_XY = pd.DataFrame()
    #ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca = PCA(n_components=n_components)
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
    # ipca.partial_fit(train_X)
    ipca.fit(train_X)
    train_X_reduced = ipca.transform(train_X)
    train_X_reduced = pd.DataFrame(train_X_reduced)
    train_Y = pd.Series(train_Y, name='Y')
    train_XY = pd.concat((train_X_reduced, train_Y), axis=1)
    return train_XY, ipca, data_scaler


def train_XY_to_seq_XY(train_XY, y_at_start_of_episode=-1, safe_to_csv=True, root_path=None, file_name="seq_train_XY"):
    # the sequence will be known by the first frame in the sequence
    """
    :param train_XY: contains pca tranasformed images with their rewards,
    :param y_at_start_of_episode: the first frame in an episodes has reward stored as -1
    :param safe_to_csv:
    :param root_path:
    :param file_name:
    :return: saves the csv which is in sequence
    """
    train_Y = train_XY.loc[:, 'Y']
    train_X = train_XY.drop('Y', axis=1)
    episodes_boundaries = train_Y[train_Y[:] == y_at_start_of_episode].index.tolist()
    frame_size = train_XY.shape[1] - 1
    seq_stack = np.array([[]])
    seq_rewards = np.array([])
    m_data_size = train_Y.size
    for e_start in range(len(episodes_boundaries)):
        if e_start == (len(episodes_boundaries) - 1):  # the last episode
            episode_seq_ids = train_Y[episodes_boundaries[e_start]:m_data_size - 7].index.tolist()
        else:
            episode_seq_ids = train_Y[episodes_boundaries[e_start]:episodes_boundaries[e_start + 1] - 7].index.tolist()

        for i in episode_seq_ids:  # get the indices for an episode
            try:
                seq_rewards = np.append(seq_rewards,
                                        train_Y.iloc[i + 7])  # reward for a sequence is reaward at the 10th frame
            except IndexError:
                print("frame for reward not found, index error")

            # creating the matrix from the image sequence a vector of length (5)*pixcels
            # the rows(m) are different sequences and columns are features obtained by flattening (5) frames
            # in the sequence
            seq_vector = np.array([])
            indices_after_missed_two = random.sample([x for x in range(6)], 4)  # the 7ths frame always stays
            indices_after_missed_two.sort()
            for j in indices_after_missed_two:
                frame_vector = train_X.iloc[i + j, :]
                seq_vector = np.append(seq_vector, frame_vector)
            # and the last frame
            frame_vector = train_X.iloc[i + 6, :]
            seq_vector = np.append(seq_vector, frame_vector)
            # stacking the sequences
            seq_stack = np.append(seq_stack, seq_vector)

    seq_rewards = seq_rewards.reshape(-1, 1)
    seq_stack = seq_stack.reshape(-1, 5 * frame_size)

    seq_stack = pd.DataFrame(seq_stack)
    seq_rewards = pd.Series(seq_rewards.flatten(), name="Y")

    train_XY = pd.concat((seq_stack, seq_rewards), axis=1)
    if safe_to_csv == True:
        train_XY.to_csv(os.path.join(root_path, file_name), index=False)
    return train_XY

def test_XY_to_seq_XY_type1(test_XY, root_path, save_to_csv=True, file_name="seq_test_XY_type1"):
    # type-1 assumes that each episodes has only 5 frames or multiple of 5
    # and reward is identified by its directory
    # 5 times as many frames as the number of rewards.
    """
    :param test_XY: contains pca tranasformed images with their rewards as tuple,
    :param save_to_csv:
    :param root_path:
    :param file_name: the name to be used to save the sequences
    :return: saves the csv which is in sequence
    """
    test_X, test_Y = test_XY
    frame_size = test_X.shape[1]
    seq_stack = np.array([[]])
    n_frames = test_X.shape[0]
    n_seq= test_Y.size
    seq_rewards = test_Y
    seq_vector = np.array([])
    frame_vector = np.array([])
    for i in range(0, n_frames, 5):
        seq_vector = np.array([])
        for j in range(5):
            frame_vector = test_X[i + j, :]
            if frame_vector.size == 0:
                seq_vector = frame_vector
            else:
                seq_vector = np.append(seq_vector, frame_vector)  # this will be flattend

        # stacking the sequences
        if seq_stack.size == 0:
            seq_vector = seq_vector.reshape(1, -1)
            seq_stack = seq_vector
        else:
            seq_vector = seq_vector.reshape(1, -1)
            seq_stack = np.concatenate((seq_stack, seq_vector), axis=0)

    seq_rewards = seq_rewards.reshape(-1, 1)
    # seq_stack = seq_stack.reshape(-1, 5 * frame_size) # don't require reshaping

    seq_stack = pd.DataFrame(seq_stack)
    seq_rewards = pd.Series(seq_rewards.flatten(), name="Y")

    test_XY = pd.concat((seq_stack, seq_rewards), axis=1)
    if save_to_csv == True:
        test_XY.to_csv(os.path.join(root_path, file_name), index=False)
    # return test_XY

def load_all_dataset_XY(root_path, top_n_episodes):
    # # loading the csv for each episode and running the PCA
    """
    read the root directory and create
    :param root_path:
    :param top_n_episodes:
    :return:
    """
    files = next(os.walk(root_path))[2]
    files.sort()
    files_csv = []
    pattern = re.compile("train_XY[\d]+\.csv")
    for x in files:
        if pattern.match(x):
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


