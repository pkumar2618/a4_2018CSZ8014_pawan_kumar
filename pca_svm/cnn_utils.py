import os
import re
import random
import cv2
import pandas as pd
import pickle


def model_architecture():
    """
    defing  model and stacking the layers
    :return:
    """
    model = Sequential()
    # first layer
    model.add(Conv2D(32, (3, 3), strides = (2,2), input_shape=train_X.shape[1:], padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2), strides=2, border_modes='valid'))

    # second layer
    model.add(Conv2D(64, (3, 3), strides = (2,2), input_shape=train_X.shape[1:], padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size= (2,2), strides=2, border_modes='valid'))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))

    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

# def seqs_matrix_to_seqs_stack(seqs_matrix_XY=pd.DataFrame(), frames_in_seq = 5, frame_shape=None):
#     """
#
#     :param seqs_matrix_XY: matrix of number of seqence, where 5*frame_size are columns
#     :param frames_in_seq:
#     :param frame_shape:
#     :return:
#     """
#     frame_size = seqs_matrix_XY.shape[1] / frames_in_seq
#     seqs_matrix_X = seqs_matrix_XY.drop('Y', axis=1).to_numpy(copy=True)
#     seqs_matrix_Y = seqs_matrix_XY.loc[:,'Y'].to_numpy(copy=True)
#     n_samples = seqs_matrix_XY.shape[0]
#
#     if frame_shape != None:
#         frame_shape = frame_shape
#     else:
#         frame_shape = [frame_size/2, frame_size/2]
#
#     seqs_stack_X = np.array()
#     seqs_stack_Y = np.array()
#     # frame_stack = np.array()
#     for i in range(n_samples):
#         frame_stack = seqs_matrix_X[i,:].reshape((frame_shape,frames_in_seq))
#         seqs_stack_X = np.append(seqs_stack_X, frame_stack)
#         seqs_stack_Y = np.append(seqs_stack_Y, seqs_matrix_Y[i])
#
#     return (seqs_stack_X, seqs_stack_Y)


def frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1):
    # # # can be called in paralle to get images and store as stack of sequences where each stack is stack of frames
    # as matrix of grayscale pixel.
    # create the stack of sequence, the grayscale frames are stacked in RGB channel, channel first
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

    seqs_reward = []
    seqs_stack = []

    for i in data_set_indices:
        frame_id = re.split("[.]", files_png[i])  # files_png might have files stored in a random order
        frame_id = int(frame_id[0])
        frame_stack = []
        indices_after_missed_two = random.sample([x for x in range(6)], 4)  # the 7ths frame always stays
        indices_after_missed_two.sort()
        # indices_after_missed_two_last = indices_after_missed_two.append(6)
        indices_after_missed_two.append(6)
        for j in indices_after_missed_two:
            frame = cv2.imread(os.path.join(path, files_png[frame_id+j]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
            # cv2.imshow('grayed image', frame)
            # cv2.waitKey(0)
            frame_stack.append(frame)

        seqs_reward.append(rewards[frame_id + j])
        seqs_stack.append(frame_stack)

    seqs_stack_XY = (seqs_stack, seqs_reward)
    file_name = "pickle_seq_stack_XY"+episode_dir
    pickle_store(seqs_stack_XY, start_path, file_name )
    # train_XY.to_csv(os.path.join(start_path, "train_XY"+episode_dir+".csv"), index=False)
    # print(seqs_stack.shape, '\n', frame_stack.shape, '\n', frame.shape)
    return seqs_stack, seqs_reward

def load_all_seqs_stack_XY(root_path, top_n_episodes):
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
        seqs_stack_X.append(temp_seqs_stack_X)
        seqs_stack_Y.append(temp_seqs_stack_Y)

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
