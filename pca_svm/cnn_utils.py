import os
import re
import random
import cv2
import pandas as pd
import pickle
import numpy as np
SEED = 0
from sklearn.model_selection import train_test_split

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


def model_architecture(input_shape=(210, 160, 5)):
    """
    defing  model and stacking the layers
    :return:
    """
    model = Sequential()
    # first layer
    model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # second layer
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))

    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1):
    # # # can be called in paralle to get images and store as stack of sequences where each stack is stack of frames
    # as matrix of grayscale pixel.
    # create the stack of sequence, the grayscale frames are stacked in RGB channel, channel first
    """
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
    data_set_indices = [x for x in range(test_population-7)]

    ### finding the rewards
    rewards = pd.read_csv(os.path.join(path, "rew.csv"), header=None)
    rewards = rewards.values
    rewards = rewards.astype('f')
    # the sequence will be identified by the last frame in the sequence,
    # here we will only stack of frames and their rewards using pandas Dataframe, with
    # the reward of the first frame being None.
    # reading all the images in bunch of 5 and stakcing them up in RGB channel
    # the rows(m) are number of sequences made of stack of frames

    # seqs_reward = []
    # seqs_stack = []
    seqs_reward = np.array([])
    seqs_stack = np.array([])

    for i in data_set_indices:
        frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
        frame_id = int(frame_id[0])
        # frame_stack = []
        frame_stack = np.array([])
        # frame = cv2.imread(os.path.join(path, files_png[frame_id + 6]))
        # frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255)
        # frame_stack = np.expand_dims(frame, axis=2)

        indices_after_missed_two = random.sample([x for x in range(6)], 4)  # the 7ths frame always stays
        indices_after_missed_two.sort()
        indices_after_missed_two.append(6)

        for j in indices_after_missed_two:
            # print(frame_stack.shape)
            frame = cv2.imread(os.path.join(path, files_png[frame_id+j]))
            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255)
            frame = np.expand_dims(frame, axis=2)
            if frame_stack.size == 0:
                frame_stack = np.copy(frame)
            else:
                frame_stack = np.concatenate((frame_stack, frame), axis=2)

            # cv2.imshow('grayed image', frame)
            # cv2.waitKey(0)
            # frame_stack = np.stack((frame_stack, frame), axis=2)

        frame_stack = np.expand_dims(frame_stack, axis=3)
        # seqs_reward.append(rewards[frame_id + j])
        # seqs_stack.append(frame_stack)

        if seqs_stack.size == 0:
            seqs_stack = np.copy(frame_stack)
            seqs_reward = np.concatenate((seqs_reward, rewards[frame_id + j]))
        else:
            seqs_stack = np.concatenate((seqs_stack, frame_stack), axis=3)
            seqs_reward = np.concatenate((seqs_reward, rewards[frame_id + j]))

    seqs_stack_XY = (seqs_stack, seqs_reward)
    file_name = "pickle_seq_stack_XY"+episode_dir
    pickle_store(seqs_stack_XY, start_path, file_name )
    # train_XY.to_csv(os.path.join(start_path, "train_XY"+episode_dir+".csv"), index=False)
    # print(seqs_stack.shape, '\n', frame_stack.shape, '\n', frame.shape)
    return seqs_stack, seqs_reward

def load_all_seqs_stack_XY(root_path, top_n_episodes=2):
    # # loading the csv for each episode and running the PCA
    """
    read the root directory and create
    :param root_path:
    :param top_n_episodes:
    :return:
    """
    files = next(os.walk(root_path))[2]
    files.sort()
    files_seqs_stacks = []
    pattern = re.compile("pickle_seq_stack_XY[\d]")
    for x in files:
        if pattern.match(x):
            files_seqs_stacks.append(x)

    episodes = files_seqs_stacks[:top_n_episodes]
    # all_train_XY = pd.DataFrame()

    # path = os.path.join(root_path, episodes[0])
    seqs_stack_X = []
    seqs_stack_Y = []
    seqs_stack_X, seqs_stack_Y = pickle_load(root_path, episodes[0])

    for seqs_stack_XY in episodes[1:]:
        temp_seqs_stack_X, temp_seqs_stack_Y = pickle_load(root_path, seqs_stack_XY)
        seqs_stack_X = np.concatenate((seqs_stack_X, temp_seqs_stack_X), axis=3)
        seqs_stack_Y = np.concatenate((seqs_stack_Y, temp_seqs_stack_Y))

    return seqs_stack_X, seqs_stack_Y

def load_all_seqs_stack_XY_balanced(root_path, top_n_episodes=2):
    # # load balanced dataset
    """
    read the root directory and create
    :param root_path:
    :param top_n_episodes:
    :return:
    """
    files = next(os.walk(root_path))[2]
    files.sort()
    files_seqs_stacks = []
    pattern = re.compile("pickle_seq_stack_XY[\d]")
    for x in files:
        if pattern.match(x):
            files_seqs_stacks.append(x)

    episodes = files_seqs_stacks[:top_n_episodes]
    # all_train_XY = pd.DataFrame()

    # path = os.path.join(root_path, episodes[0])
    seqs_stack_X = np.array([])
    seqs_stack_Y = np.array([])
    # seqs_stack_X, seqs_stack_Y = pickle_load(root_path, episodes[0])

    for seqs_stack_XY in episodes:
        if seqs_stack_X.size == 0:
            seqs_stack_X, seqs_stack_Y = pickle_load(root_path, seqs_stack_XY)
            true_indices= (seqs_stack_Y == 1).reshape(-1)
            temp_true_indices = np.copy(true_indices)
            np.random.seed(0)
            np.random.shuffle(temp_true_indices)
            false_indices = temp_true_indices
            balance_indices = true_indices+false_indices
            seqs_stack_Y = seqs_stack_Y[balance_indices]
            seqs_stack_X = seqs_stack_X[:, :, :, balance_indices]
        else:

            temp_seqs_stack_X, temp_seqs_stack_Y = pickle_load(root_path, seqs_stack_XY)
            true_indices = (temp_seqs_stack_Y == 1).reshape(-1)
            temp_true_indices = np.copy(true_indices)
            np.random.seed(0)
            np.random.shuffle(temp_true_indices)
            false_indices = temp_true_indices
            balance_indices = true_indices + false_indices
            temp_seqs_stack_Y = temp_seqs_stack_Y[balance_indices]
            temp_seqs_stack_X = temp_seqs_stack_X[:, :, :, balance_indices]
            seqs_stack_X = np.concatenate((seqs_stack_X, temp_seqs_stack_X), axis=3)
            seqs_stack_Y = np.concatenate((seqs_stack_Y, temp_seqs_stack_Y))

    return seqs_stack_X, seqs_stack_Y


def frame_reward_to_seqs_stack_XY_test1(start_path, sample_fraction=0.1):
    # # # can be called in paralle to get images and store as stack of sequences where each stack is stack of frames
    # as matrix of grayscale pixel.
    # create the stack of sequence, the grayscale frames are stacked in RGB channel, channel first
    # type1 has only 5 frames in an episodes, and sequence is identified by episodes name
    """
    :param start_path:
    :param episode_dir:
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
    episodes_indices = random.sample([x for x in range(m_dirs_contained)], test_population)
    # not random
    # episodes_indices = [x for x in range(test_population)]

    # the sequence will be identified by the last frame in the sequence,
    # here we will only stack of frames and their rewards using pandas Dataframe, with
    # reading all the images in bunch of 5 and stakcing them up in RGB channel
    # the rows(m) are number of sequences made of stack of frames

    seqs_reward = np.array([])
    seqs_stack = np.array([])

    for d in episodes_indices:
        dir_id = int(dirs[d])
        path = os.path.join(start_path, dirs[d])
        files = os.listdir(path)
        files_png = []
        # only take .png files
        for x in files:
            if x.endswith(".png"):
                files_png.append(x)

        # n_files_contained = len(files_png)
        files_png.sort()  # files_png might have files stored in a random order

        frame_stack = np.array([])
        for i in range(5): # assuming the test has one sequence in each episodes
            # frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
            # frame_id = int(frame_id[0])
            # print(frame_stack.shape)
            frame = cv2.imread(os.path.join(path, files_png[i]))
            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255)
            frame = np.expand_dims(frame, axis=2)
            if frame_stack.size == 0:
                frame_stack = np.copy(frame)
            else:
                frame_stack = np.concatenate((frame_stack, frame), axis=2)

        frame_stack = np.expand_dims(frame_stack, axis=3)

        if seqs_stack.size == 0:
            seqs_stack = np.copy(frame_stack)
            seqs_reward = np.concatenate((seqs_reward, rewards[dir_id]))
        else:
            seqs_stack = np.concatenate((seqs_stack, frame_stack), axis=3)
            seqs_reward = np.concatenate((seqs_reward, rewards[dir_id]))

    seqs_stack_XY = (seqs_stack, seqs_reward)
    file_name = "pickle_seq_stack_XY_test1_tuple"
    pickle_store(seqs_stack_XY, start_path, file_name)
    # return seqs_stack, seqs_reward

def dirID_to_seqs_stack_X_ID(start_path, dirs_batch):
    # # # can be called in paralle to get images and store as stack of sequences where each stack is stack of frames
    # as matrix of grayscale pixel.
    # create the stack of sequence, the grayscale frames are stacked in RGB channel, channel first
    # type1 has only 5 frames in an episodes, and sequence is identified by episodes name
    """
    :param start_path:
    :param episode_dir:

    """
    # the sequence will be identified by the dir id,
    # here we will only stack of frames and their rewards using pandas Dataframe, with
    # reading all the images in bunch of 5 and stakcing them up in RGB channel
    # the rows(m) are number of sequences made of stack of frames

    seqs_id = np.array([])
    seqs_stack = np.array([])
    for i in range(len(dirs_batch)):
        dir_id = int(dirs_batch[i])
        path = os.path.join(start_path, dirs_batch[i])
        files = os.listdir(path)
        files_png = []
        # only take .png files
        for x in files:
            if x.endswith(".png"):
                files_png.append(x)

        # n_files_contained = len(files_png)
        files_png.sort()  # files_png might have files stored in a random order

        frame_stack = np.array([])
        for i in range(5):  # assuming the test has one sequence in each episodes
            # frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
            # frame_id = int(frame_id[0])
            # print(frame_stack.shape)
            frame = cv2.imread(os.path.join(path, files_png[i]))
            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255)
            frame = np.expand_dims(frame, axis=2)
            if frame_stack.size == 0:
                frame_stack = np.copy(frame)
            else:
                frame_stack = np.concatenate((frame_stack, frame), axis=2)

        frame_stack = np.expand_dims(frame_stack, axis=3)

        if seqs_stack.size == 0:
            seqs_stack = np.copy(frame_stack)
            seqs_id = np.append(seqs_id, dir_id)
        else:
            seqs_stack = np.concatenate((seqs_stack, frame_stack), axis=3)
            seqs_id = np.append(seqs_id, dir_id)

    # seqs_stack_XY = (seqs_stack, seqs_id)
    # file_name = "pickle_seq_stack_XID_compete_tuple"
    # pickle_store(seqs_stack_XY, start_path, file_name)
    return seqs_stack, seqs_id


def frame_dirID_to_seqs_stack_XID_compete(start_path, sample_fraction=0.1):
    # # # can be called in paralle to get images and store as stack of sequences where each stack is stack of frames
    # as matrix of grayscale pixel.
    # create the stack of sequence, the grayscale frames are stacked in RGB channel, channel first
    # type1 has only 5 frames in an episodes, and sequence is identified by episodes name
    """
    :param start_path:
    :param episode_dir:
    :param sample_fraction: how much of the frames in the episodes has to be read, another way to randomly select
    :return:
    """
    dirs = next(os.walk(start_path))[1]
    # files = next(os.walk(start_path))[2]
    dirs.sort()

    # ### finding the rewards, assuming the reward.csv is in the start_path
    # pattern = re.compile("rew[a-zA-Z]+\\.csv")
    # for x in files:
    #     if pattern.match(x):
    #         rewards = pd.read_csv(os.path.join(start_path, x), header=None, index_col=0)
    #         rewards = rewards.values
    #         rewards = rewards.astype('f')
    #         break

    # selecting only a fraciton of episodes useful during debugging
    m_dirs_contained = len(dirs)
    t_frac = sample_fraction
    test_population = int(t_frac * m_dirs_contained)

    # when selecting random dirs
    random.seed(SEED)
    episodes_indices = random.sample([x for x in range(m_dirs_contained)], test_population)
    # not random
    # episodes_indices = [x for x in range(test_population)]

    # the sequence will be identified by the last frame in the sequence,
    # here we will only stack of frames and their rewards using pandas Dataframe, with
    # reading all the images in bunch of 5 and stakcing them up in RGB channel
    # the rows(m) are number of sequences made of stack of frames

    seqs_reward = np.array([])
    seqs_stack = np.array([])

    for d in episodes_indices:
        dir_id = int(dirs[d])
        path = os.path.join(start_path, dirs[d])
        files = os.listdir(path)
        files_png = []
        # only take .png files
        for x in files:
            if x.endswith(".png"):
                files_png.append(x)

        # n_files_contained = len(files_png)
        files_png.sort()  # files_png might have files stored in a random order

        frame_stack = np.array([])
        for i in range(5): # assuming the test has one sequence in each episodes
            # frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
            # frame_id = int(frame_id[0])
            # print(frame_stack.shape)
            frame = cv2.imread(os.path.join(path, files_png[i]))
            frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255)
            frame = np.expand_dims(frame, axis=2)
            if frame_stack.size == 0:
                frame_stack = np.copy(frame)
            else:
                frame_stack = np.concatenate((frame_stack, frame), axis=2)

        frame_stack = np.expand_dims(frame_stack, axis=3)

        if seqs_stack.size == 0:
            seqs_stack = np.copy(frame_stack)
            seqs_reward = np.append(seqs_reward, dir_id)
        else:
            seqs_stack = np.concatenate((seqs_stack, frame_stack), axis=3)
            seqs_reward = np.append(seqs_reward, dir_id)

    seqs_stack_XY = (seqs_stack, seqs_reward)
    file_name = "pickle_seq_stack_XID_compete_tuple"
    pickle_store(seqs_stack_XY, start_path, file_name)
    # return seqs_stack, seqs_reward

def load_all_seqs_stack_XY_test1(root_path, top_n_episodes=2):
    # # loading the csv for each episode and running the PCA
    """
    read the root directory and create
    :param root_path:
    :param top_n_episodes:
    :return:
    """
    files = next(os.walk(root_path))[2]
    files.sort()
    files_seqs_stacks = []
    pattern = re.compile("pickle_seq_stack_XY_test1_tuple")
    for x in files:
        if pattern.match(x):
            files_seqs_stacks.append(x)

    episodes = files_seqs_stacks[:top_n_episodes]
    # all_train_XY = pd.DataFrame()

    # path = os.path.join(root_path, episodes[0])
    seqs_stack_X = []
    seqs_stack_Y = []
    seqs_stack_X, seqs_stack_Y = pickle_load(root_path, episodes[0])

    for seqs_stack_XY in episodes[1:]:
        temp_seqs_stack_X, temp_seqs_stack_Y = pickle_load(root_path, seqs_stack_XY)
        seqs_stack_X = np.concatenate((seqs_stack_X, temp_seqs_stack_X), axis=3)
        seqs_stack_Y = np.concatenate((seqs_stack_Y, temp_seqs_stack_Y))

    return seqs_stack_X, seqs_stack_Y

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
