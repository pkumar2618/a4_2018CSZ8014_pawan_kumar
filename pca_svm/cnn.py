import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
# import tensorflow as tf
# from sklearn import StandardScaler
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import load_model
from cnn_utils import load_all_seqs_stack_XY, frame_reward_to_seqs_stack_XY
from cnn_utils import model_architecture

train_test = int(sys.argv[1])
start_path = sys.argv[2]
n_episodes = int(sys.argv[3])

if train_test == 0:
    # train the model
    # start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'

    # episode_dir ='00000001'
    # frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)
    # episode_dir ='00000002'
    # frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)

    # load all the grayscale images stacke in 5 alog RGB channel, channel second last last,
    seqs_stack_X, seqs_stack_Y = load_all_seqs_stack_XY(start_path, n_episodes)

    # # see the images first few
    # for i in range(100, 110):
    #     for j in range(0, 5):
    #     # for frame in seqs_stack_X[:, :, :, i]:
    #         # for frame in seq_stack:
    #         cv2.imshow('grayed image', seqs_stack_X[:,:,j,i])
    #         cv2.waitKey(0)
    #     print(seqs_stack_Y[i])
    # # print(count, '\n')

    # splitting the data into train and validation test
    m_samples = seqs_stack_X.shape[3]
    # print(seqs_stack_X.shape)
    m_train = int(m_samples*1)
    train_X = np.array([seqs_stack_X[:, :, :, i] for i in range(m_train)])
    train_Y = seqs_stack_Y[:m_train]
    # print(train_X.shape)

    # val_X =np.array([seqs_stack_X[:, :, :, i] for i in range(m_train, m_samples)])
    # val_Y = seqs_stack_Y[m_train:]
    # print(val_X.shape)

    # # # see the images first few
    # for i in range(100, 110):
    #     for j in range(0, 5):
    #     # for frame in seqs_stack_X[:, :, :, i]:
    #         # for frame in seq_stack:
    #         cv2.imshow('grayed image', train_X[i,:,:,j])
    #         cv2.waitKey(0)
    #     print(train_Y[i])

    input_shape = train_X[0, :, :, :].shape
    # print(input_shape)

    ## creating Model Architecture

    model = model_architecture(input_shape)
    # model.summary()

    epochs = 20
    batch_size = 100

    # # model configuration
    model.compile(loss ='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

    # # model fitting
    training_history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # # saving the model
    model.save_weights('weights_cnn_32_64_2k_binary_e20_b100_size%d'%m_samples+'.h5')
    model.save('cnn_32_64_2k_binary_e20_b100_size%d'%m_samples+'.h5')
    model.save('cnn_32_64_2k_binary_e20_b100_size%d'%m_samples+'.model')

    # # plotting accuracy
    train_acc = training_history.history['acc']
    val_acc = training_history.history['val_acc']
    train_loss = training_history.history['loss']
    val_loss = training_history.history['val_acc']
    epoch_list = range(1, len(train_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    line1 = ax1.plot(epoch_list, train_acc, label='train accuracy')
    line2 = ax1.plot(epoch_list, val_acc, label='validation accuracy')
    ax1.legend()
    ax1.set_xlabel("epach")
    ax1.set_ylabel("accuracy")
    ax1.set_title("train and validation accuracy")

    line3 = ax2.plot(epoch_list, train_loss, label='train loss')
    line4 = ax2.plot(epoch_list, val_loss, label='validation loss')
    ax2.legend()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title("train and validation loss")

    plt.tight_layout()
    fig_name = os.path.join(start_path, "cnn_loss_curve_size%d" %m_samples+".png")
    plt.savefig(fig_name, format='png')
#    plt.show()


if train_test == 1:
    # test the trained model
    # load a saved model
    model = load_model('./cnn_32_64_2k_binary_itr300.model')

    # # model evalution on test data
    model.evaluate(val_X, val_Y)
