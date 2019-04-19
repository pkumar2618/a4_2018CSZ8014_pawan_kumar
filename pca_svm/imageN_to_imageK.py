import os
import re
import cv2
import numpy as np
import pandas as pd

path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
# start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
# start_path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/'

# for dir in os.listdir(start_path):
#     path = os.path.join(start_path, dir)

files = os.listdir(path)
files.sort()
# only take .png files
files_png = []
for x in files:
    if x.endswith(".png"):
        files_png.append(x)
n_files_contained = len(files_png)

# selecting only a fraciton of files useful during debugging
t_frac = 0.2
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
train_XY.to_csv(os.path.join(path, "train_XY.csv"), index=False)
print(frame_stack.shape)
