import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import mean_squared_error, f1_score
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

# from keras.models import load_model

from cnn_utils import load_all_seqs_stack_XY, load_all_seqs_stack_XY_balanced,\
    frame_reward_to_seqs_stack_XY, pickle_load, pickle_store
from cnn_utils import model_architecture

train_test = int(sys.argv[1])
start_path = sys.argv[2]

if train_test == 0:
    # train the model
    # start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'

    # episode_dir ='00000001'
    # frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)
    # episode_dir ='00000002'
    # frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)

    n_episodes = int(sys.argv[3])
    # load all the grayscale images stacke in 5 alog RGB channel, channel second last last,
    seqs_stack_X, seqs_stack_Y = load_all_seqs_stack_XY_balanced(start_path, n_episodes)

    # # see the images first few
    # for i in range(100, 110):
    #     for j in range(0, 5):
    #     # for frame in seqs_stack_X[:, :, :, i]:
    #         # for frame in seq_stack:
    #         cv2.imshow('grayed image', seqs_stack_X[:,:,j,i])
    #         cv2.waitKey(0)
    #     print(seqs_stack_Y[i])
    # # print(count, '\n')

    # splitting the data for balancing
    # print(seqs_stack_X.shape)
    # reversing the order of the sequence channel
    m_samples = seqs_stack_X.shape[3]
    m_train = int(m_samples*1)
    train_X = np.array([seqs_stack_X[:, :, :, i] for i in range(m_train)])
    train_Y = seqs_stack_Y[:m_train]
    # print(train_X.shape)

    # val_X =np.array([seqs_stack_X[:, :, :, i] for i in range(m_train, m_samples)])
    # val_Y = seqs_stack_Y[m_train:]
    # print(val_X.shape)

    # # # see the images first few
    # for i in range(1, 10):
    #     for j in range(0, 5):
    #         cv2.imshow('grayed image', train_X[i,:,:,j])
    #         cv2.waitKey(0)
    #     print(train_Y[i])

    input_shape = train_X[0, :, :, :].shape
    # print(input_shape)

    ## creating Model Architecture
    model = model_architecture(input_shape)
    # model.summary()

    epochs = 25
    batch_size = 50

    # # model configuration
    model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # model fitting
    training_history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # # evaluate model
    scores = model.evaluate(train_X, train_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    file_json = os.path.join(start_path, 'cnn_32_64_2k_binary_e%d_b%d_size_%d'%(epochs, batch_size, m_samples)+'.json')
    model_json = model.to_json()
    with open(file_json, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    file_weight = os.path.join(start_path, 'cnn_32_64_2k_binary_e%d_b%d_size_%d'%(epochs, batch_size, m_samples)+'weights.h5')
    model.save_weights(file_weight)
    print("Saved model to disk")

    # # load json and create model
    json_file = open(file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(file_weight)
    print("model has been loaded know compiling it")

    # evaluate loaded model on test data
    loaded_model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(train_X, train_Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


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
    fig_name = os.path.join(start_path, "cnn_loss_curve_size_e%d_b%d_size_%d"%(epochs, batch_size, m_samples)+".png")
    plt.savefig(fig_name, format='png')
#    plt.show()


if train_test == 1:
    # test the trained model
    # load all the seq stack obtained from frame and club them into one seq stack
    # n_episodes = int(sys.argv[3])
    # load all the grayscale images stacke in 5 alog RGB channel, channel second last to last,
    seqs_stack_X, seqs_stack_Y = pickle_load(root_path=start_path, file_name="pickle_seq_stack_XY_test1_tuple")

    # changing the channel for sequences
    m_samples = seqs_stack_X.shape[3]
    # print(seqs_stack_X.shape)
    m_test = int(m_samples * 1)
    test_X = np.array([seqs_stack_X[:, :, :, i] for i in range(m_test)])
    test_Y = seqs_stack_Y[:m_test]
    print(test_X.shape)

    # # see the images first few
    # for i in range(1, 10):
    #     for j in range(0, 5):
    #         cv2.imshow('grayed image', test_X[i,:,:,j])
    #         cv2.waitKey(0)
    #     print(test_Y[i])

    input_shape = test_X[0, :, :, :].shape
    print(input_shape)

    # load a saved model
    # load json and create model
    file_json = os.path.join(start_path, 'cnn_model.json')
    json_file = open(file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        # model = load_model('my_model.h5')
        loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    file_weight = os.path.join(start_path, 'cnn_weight.h5')
    loaded_model.load_weights(file_weight)
    print("model has been loaded know compiling it")

    # compile the loaded model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # predict accuracy
    test_Y_pred = loaded_model.predict(test_X)
    test_Y_pred = (test_Y_pred >= 0.5).astype(int)
    test_f1_score = f1_score(test_Y, test_Y_pred, average='binary')
    print('test f1-score accuracy', test_f1_score)
    # # model evalution on test data

if train_test == 7:
    # predicting test data for competion
    # load all the seq stack obtained from frame and club them into one seq stack
    # n_episodes = int(sys.argv[3])
    # load all the grayscale images stacke in 5 alog RGB channel, channel second last to last,
    seqs_stack_X, episode_ID = pickle_load(root_path=start_path, file_name="pickle_seq_stack_XID_compete_tuple")

    # changing the channel for sequences
    m_samples = seqs_stack_X.shape[3]
    # print(seqs_stack_X.shape)
    m_test = int(m_samples * 1)
    test_X = np.array([seqs_stack_X[:, :, :, i] for i in range(m_test)])
    test_Y = episode_ID[:m_test]
    print(test_X.shape)

    # # see the images first few
    # for i in range(1, 10):
    #     for j in range(0, 5):
    #         cv2.imshow('grayed image', test_X[i,:,:,j])
    #         cv2.waitKey(0)
    #     print(test_Y[i])

    input_shape = test_X[0, :, :, :].shape
    print(input_shape)

    # load a saved model
    # load json and create model
    file_json = os.path.join(start_path, 'cnn_model.json')
    json_file = open(file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        # model = load_model('my_model.h5')
        loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    file_weight = os.path.join(start_path, 'cnn_weight.h5')
    loaded_model.load_weights(file_weight)
    print("model has been loaded know compiling it")

    # compile the loaded model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # predict accuracy
    test_Y_pred = loaded_model.predict(test_X)
    test_Y_pred = (test_Y_pred >= 0.5).astype(int)
    test_Y_pred = test_Y_pred.reshape(-1)
    test_Y = test_Y.reshape(-1)

    predicted_Y = pd.Series(test_Y_pred, name="model_prediction")
    IDs = pd.Series(test_Y, name="ID")
    ID_pred_Y = pd.concat((IDs, predicted_Y), axis=1)
    file_name = os.path.join(start_path, "result_ID_Y")
    ID_pred_Y.to_csv(file_name, index=False)

    # test_f1_score = f1_score(test_Y, test_Y_pred, average='binary')
    # print('test f1-score accuracy', test_f1_score)
    # # model evalution on test data
