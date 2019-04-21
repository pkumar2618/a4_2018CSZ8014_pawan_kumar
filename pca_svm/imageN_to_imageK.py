import os
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from svmutil import *
import matplotlib.pyplot as plt

from utils import frame_reward_to_matrix_XY, load_all_dataset_XY
from utils import transforming_with_pca, train_XY_to_seq_XY
from utils import pickle_store, pickle_load



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
#                                    args=(start_path, episode_dir, 1)) for episode_dir in episodes]
# pool.close()
# pool.join()


"""
loading the csv for each episode and running the incremental PCA after 
all the data is loaded, the function returns reduced data_set
"""
# train_XY_reduced, ipca, data_scaler = transforming_with_pca(root_path=start_path,
#                                              top_n_episodes=2, n_components=50, batch_size=None)

# pickling the reduced dataset
# pickle_store(train_XY_reduced, root_path=start_path, file_name="pickle_train_XY_reduced")
# pickle_store(ipca, root_path=start_path, file_name="pickle_ipca")
# pickle_store(data_scaler, root_path=start_path, file_name="pickle_data_scaler")

# loading the pickled dataset
# train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
# ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
# data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")

# # to measure reconstruction error we need the original dataset
# train_X_reduced = train_XY_reduced.drop('Y', axis = 1).values
# reconstructed_X = ipca.inverse_transform(train_X_reduced)
# train_XY_original = load_all_dataset_XY(root_path=start_path, top_n_episodes=2)
# error_curr = mean_squared_error(reconstructed_X, train_XY_original.drop('Y', axis=1).values)
# print("sequential error using function", error_curr)

"""
Sequence sampling
"""
# the sequence will be known by the first frame in the sequence
# loading the pickled dataset
# train_XY = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
# 
# # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
# seq_train_XY = train_XY_to_seq_XY(train_XY, y_at_start_of_episode=-1,
#                        safe_to_csv=True, root_path=start_path, file_name="seq_train_XY")
# print(seq_train_XY.shape)


"""
binary classifier using linear as well as gaussian SVM
"""
question_part = 'd'
if question_part == 'd': ## parameter tunning
        train_XY = pd.read_csv(os.path.join(start_path, "seq_train_XY"))
        print(train_XY.shape)

        ## creating a validation set
        dev_set = train_XY.sample(frac=0.1, replace = False, random_state=1, axis=0)
        train_XY = train_XY.drop(labels=dev_set.index)

        train_X = train_XY.drop('Y',axis=1).to_numpy(copy=True)
        train_Y = train_XY.loc[:, 'Y'].to_numpy(copy=True)
        dev_X = dev_set.drop('Y', axis=1).to_numpy(copy=True)
        dev_Y = dev_set.loc[:, 'Y'].to_numpy(copy=True)

        dev_acc_lin = []
        dev_acc_gauss = []
        # train_acc = []
        cost_c = [1e-5, 1e-3, 1, 5, 10]
        cwd = os.getcwd()
        for c in cost_c:
            # m = svm_train(train_Y, train_X_normed, '-t 0')
            param_string_lin = '-t 2 -c %f' % c
            param_string_gauss = '-g 0.05 -t 2 -c %f' % c

            m_lin = svm_train(train_Y, train_X, param_string_lin)
            m_gauss = svm_train(train_Y, train_X, param_string_gauss)

            p_label_lin, p_acc_lin, p_val_lin, = svm_predict(dev_Y, dev_X, m_lin)
            p_label_gauss, p_acc_gauss, p_val_gauss, = svm_predict(dev_Y, dev_X, m_gauss)

            dev_acc_lin.append(p_acc_lin[0])
            dev_acc_gauss.append(p_acc_gauss[0])

        print('dev_acc_lin', dev_acc_lin)
        print('dev_acc_gauss', dev_acc_gauss)
        # print('test_acc', test_acc)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        # ax3 = fig1.add_subplot(grid[1, :])

        line1 = ax1.plot(cost_c, dev_acc_lin, label='validation accuracy linear SVM with c')
        ax1.legend()
        ax1.set_xlabel("parameter C")
        ax1.set_ylabel("accuracy")
        ax1.set_title("accuracy vs for linear SVM")

        line2 = ax2.plot(cost_c, dev_acc_gauss, label='validation accuracy with gaussian kernel with c')
        ax2.legend()
        ax2.set_xlabel("parameter C")
        ax2.set_ylabel("accuracy")
        ax2.set_title("accuracy for Gaussina SVM")

        plt.tight_layout()
        fig_name = os.path.join(cwd, "lin_gauss_svm_accuracy_with_c")
        plt.savefig(fig_name, format='png')
        plt.show()
