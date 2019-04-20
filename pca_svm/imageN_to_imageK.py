import os
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import mean_squared_error
import pandas as pd

from utils import frame_reward_to_matrix_XY
from utils import transforming_with_pca


# path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
# start_path = '/home/pawan/train_dataset/'

# results = []  # used to store the result of async processes
# # # running parallel processes to get images stored as matrix of grayscale pixel.
# dirs = next(os.walk(start_path))[1]
# dirs.sort()
# episodes = dirs[:2]
# pool = mp.Pool(mp.cpu_count())
# result_objects = [pool.apply_async(frame_reward_to_matrix_XY,
#                                    args=(start_path, episode_dir, 0.01)) for episode_dir in episodes]
# pool.close()
# pool.join()


# # loading the csv for each episode and running the PCA
files = next(os.walk(start_path))[2]
files.sort()
n_components = 10
episodes = files[:2]

all_train_XY = pd.DataFrame()
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
# ipca = IncrementalPCA(n_components=n_components, batch_size=10)
for train_XY_csv in episodes:
    path = os.path.join(start_path, train_XY_csv)
    train_XY = pd.read_csv(path)
    all_train_XY = pd.concat((all_train_XY, train_XY), axis=0)

# separating features and rewards
# print('reading complete')
train_X = all_train_XY.drop('Y', axis=1).values
train_Y = all_train_XY['Y'].values

# scalind data to zero mean, we won't use unit variance here
data_scaler = StandardScaler(with_std=False)
train_X = data_scaler.fit_transform(train_X)

#applying pca on the scaled data
ipca.partial_fit(train_X)
train_X_reduced = ipca.transform(train_X)
train_X_reduced = pd.DataFrame(train_X_reduced)
train_Y = pd.Series(train_Y, name='Y')
train_XY = pd.concat((train_X_reduced, train_Y), axis=1)

reconstructed_X = ipca.inverse_transform(train_X_reduced.values)
error_curr = mean_squared_error(reconstructed_X, train_X)
print("sequenctial error", error_curr)

fn_train_XY, fn_ipca = transforming_with_pca(root_path = start_path,
                                             top_n_episodes = 2, n_components =10, batch_size=10)
fn_train_X_reduced = fn_train_XY.drop('Y', axis = 1).values
fn_reconstructed_X = fn_ipca.inverse_transform(fn_train_X_reduced)
fn_error_curr = mean_squared_error(fn_reconstructed_X, train_X)
print("sequenctial error using function", fn_error_curr)