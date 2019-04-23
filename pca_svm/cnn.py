
import os
import gc
import numpy as np
import pandas as pd
import cv2
# import tensorflow as tf
# from sklearn import StandardScaler
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import optimizers

from cnn_utils import load_all_seqs_stack_XY, frame_reward_to_seqs_stack_XY
from cnn_utils import model_architecture
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'

# episode_dir ='00000001'
# frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)
# episode_dir ='00000002'
# frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)

# load all the grayscale images stacke in 5 alog RGB channel, channel first,
seqs_stack_X, seqs_stack_Y = load_all_seqs_stack_XY(start_path, 2)

# # see the images first few
# count = 0
# for i in range(100, 110):
#     for j in range(0, 5):
#     # for frame in seqs_stack_X[:, :, :, i]:
#         # for frame in seq_stack:
#         cv2.imshow('grayed image', seqs_stack_X[:,:,j,i])
#         cv2.waitKey(0)
#
#     count = count + 1
#
#     print(seqs_stack_Y[i])
# # print(count, '\n')

# splitting the data into train and validation test
m_samples = seqs_stack_X.shape[3]
# train_X, val_X, train_Y, val_Y = train_test_split([seqs_stack_X[:,:,:,i] for i in range(m_samples)], seqs_stack_Y, test_size=0.20, random_state=1)


# print(seqs_stack_X.shape)
m_train = int(m_samples*0.8)
train_X = np.array([seqs_stack_X[:, :, :, i] for i in range(m_train)])
train_Y = seqs_stack_Y[:m_train]
print(train_X.shape)

val_X =np.array([seqs_stack_X[:, :, :, i] for i in range(m_train, m_samples)])
val_Y = seqs_stack_Y[m_train:]
print(val_X.shape)

# # # see the images first few
# count = 0
# for i in range(100, 110):
#     for j in range(0, 5):
#     # for frame in seqs_stack_X[:, :, :, i]:
#         # for frame in seq_stack:
#         cv2.imshow('grayed image', train_X[i,:,:,j])
#         cv2.waitKey(0)
#     count = count + 1
#     print(train_Y[i])

input_shape = train_X[0, :, :, :].shape
# print(input_shape)

## creating Model Architecture

model = model_architecture(input_shape)
model.summary()

batch_size = 25
epochs = 50

# # model configuration
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# # model fitting
model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# # model evalution on test data
# model.evaluate(test_data, test_labels_one_hot)
#
# # saving the model
model.save('cnn_32_64_2k_b.model')