import os
import re
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
files = os.listdir(path)
# only take .png files
files_png = []
for x in files:
    if x.endswith(".png"):
        files_png.append(x)

# index = random.randrange(0, len(files)-9)
n_files_contained = len(files_png)-10  # so that while selecting the reward we will not heat ceiling

t_frac = 0.1
test_population = int(t_frac*n_files_contained)
test_data_indices = random.sample([x for x in range(n_files_contained)], test_population)

### finding the rewards
rewards = pd.read_csv(os.path.join(path,"rew.csv"), header=None)
# the sequence will be known by the first frame in the sequence
seq_rewards = np.array([])
for i in test_data_indices:  # the order of test_data_indices must not be changed while reading the frames
    frame_id = re.split("[.]", files_png[i])
    seq_id = int(frame_id[0])
    seq_rewards=np.append(seq_rewards, rewards.iloc[seq_id+9-1])  # reward for a sequence is reaward at the 10th frame

seq_rewards = seq_rewards.reshape(-1, 1)

# creating the matrix from the image sequence a vector of length (5)*pixcels*3
# the rows(m) are different sequences and columns are features obtained by flattening (5) frames
# in the sequence
frame = plt.imread(os.path.join(path, files_png[0]))
rows, cols, colors = frame.shape  # gives dimensions for RGB array
frame_size = rows*cols*colors
frame_vector = frame.reshape(frame_size)
# img = frame_vector.reshape(rows,cols,colors)  # get the image back

seq_vector = np.array([])
seq_vector_size = frame_size*(5)
seq_stack = np.array([[]])
for i in test_data_indices:
    seq_vector = np.array([])
    indices_after_missed_two = random.sample([x for x in range(7)], 5)
    indices_after_missed_two.sort()
    for j in indices_after_missed_two:
        frame_vector = plt.imread(os.path.join(path, files_png[i+j])).reshape(frame_size)
        seq_vector = np.append(seq_vector, frame_vector)

    # stacking various sequences
    # seq_vector = seq_vector.reshape(1,-1)
    seq_stack = np.append(seq_stack, seq_vector)

seq_stack = seq_stack.reshape(-1, 5*frame_size)

seq_stack = pd.DataFrame(seq_stack)

print(seq_stack.shape)

# including the directory as well
# random.choice([x for x in os.listdir("../") if os.path.isfile(os.path.join("../", x))])