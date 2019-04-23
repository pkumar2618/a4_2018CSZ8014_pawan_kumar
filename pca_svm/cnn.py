
import os
import gc
import numpy as np
import pandas as pd
import cv2

# import tensorflow as tf
# from sklearn import StandardScaler
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from cnn_utils import load_all_seqs_stack_XY, frame_reward_to_seqs_stack_XY

from cnn_utils import model_architecture
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'

# episode_dir ='00000001'
# frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)
# episode_dir ='00000002'
# frame_reward_to_seqs_stack_XY(start_path, episode_dir, sample_fraction=0.1)

# load all the grayscale images stacke in 5 alog RGB channel, channel first,
seqs_stack_X, seqs_stack_Y = load_all_seqs_stack_XY(start_path, 2)

# see the images first few
# count = 0
# for seq_stack in seqs_stack_X:
#     for frame in seq_stack:
#         count = count +1
#         cv2.imshow('grayed image', frame)
#         cv2.waitKey(0)
# for i in range(100, 110):
#     for frame in seqs_stack_X[i][:][:][:]:
#         # for frame in seq_stack:
#             cv2.imshow('grayed image', frame)
#             cv2.waitKey(0)
#     count = count + 1
#     print(seqs_stack_Y[i])
# print(count, '\n')

print(seqs_stack_X.count)
seq_size = len(seqs_stack_X)
print(seq_size)

# frame_stack = np.array([])
# temp_frame_stack = np.array([np.array(frame) for frame in seqs_stack_X[0]])
# frame_stack_shape = temp_frame_stack.shape
# for frame_stack in seqs_stack_X[1:]:
#     temp_temp_frame_stack = np.array([np.array(frame) for frame in frame_stack])
#     temp_frame_stack = np.append(temp_frame_stack, temp_temp_frame_stack)
#     # print(temp_frame_stack.shape)
# #     # temp_temp_frame_stack = np.expand_dims(temp_temp_frame_stack, axis=0)
# #     # print(temp_temp_frame_stack.shape)
# #     print(temp_seqs_stack_X.shape)
# #     temp_seqs_stack_X = np.append(temp_seqs_stack_X, temp_temp_frame_stack, axis=0)
#
# temp_frame_stack.reshape((seq_size, frame_stack_shape[0],frame_stack_shape[1], frame_stack_shape[2]))
#     # print(temp_frame_stack.shape)


seqs_stack_X = np.array(seqs_stack_X)
print(seqs_stack_X.shape)
model1 = model_architecture

batch_size = 256
epochs = 100

# # model configuration
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#
# # model fitting
# model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_split = 0.1)
#
# # model evalution on test data
# model.evaluate(test_data, test_labels_one_hot)
#
# # saving the model
# model.save('cnn_32_64_2k_b.model')