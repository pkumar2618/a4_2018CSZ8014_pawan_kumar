import os
import sys
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import mean_squared_error, f1_score
from sklearn import svm
from sklearn.model_selection import learning_curve, GridSearchCV
import numpy as np
import pandas as pd

# from svmutil import *
import matplotlib.pyplot as plt

from utils import frame_reward_to_matrix_XY, load_all_dataset_XY
from utils import transforming_with_pca, train_XY_to_seq_XY
from utils import pickle_store, pickle_load

train_test = int(sys.argv[1])
# 9: is tune
# 0: train
# 1: is predict

start_path = sys.argv[2]
n_episodes = int(sys.argv[3])

# path ='/Users/pawan/Documents/ml_assg/assig4/train_dataset/00000001/'
# start_path = '/Users/pawan/Documents/ml_assg/assig4/train_dataset/'
# path = '/home/dell/Documents/2nd_sem/ml_assig/train_dataset/00000001/'
start_path = '/home/pawan/train_dataset/'

if train_test == 9:

        # loading the pickled dataset
        train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
        ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
        data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")

        # # to measure reconstruction error we need the original dataset
        # train_X_reduced = train_XY_reduced.drop('Y', axis = 1).values
        # reconstructed_X = ipca.inverse_transform(train_X_reduced)
        # train_XY_original = load_all_dataset_XY(root_path=start_path, top_n_episodes=2)
        # error_curr = mean_squared_error(reconstructed_X, train_XY_original.drop('Y', axis=1).values)
        # print("sequential error using function", error_curr)


        # the sequence will be known by the first frame in the sequence
        # # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
        seq_train_XY = train_XY_to_seq_XY(train_XY, y_at_start_of_episode=-1,
                               safe_to_csv=True, root_path=start_path, file_name="seq_train_XY")
        # print(seq_train_XY.shape)


        ## parameter tunning using grid search

        train_XY = seq_train_XY
        ## creating a validation set
        # dev_set = train_XY.sample(frac=0.03, replace = False, random_state=1, axis=0)
        # train_XY = train_XY.drop(labels=dev_set.index)
        train_XY = train_XY.sample(frac=0.15, replace = False, random_state=1, axis=0)
        # print(train_XY.shape)
        train_X = train_XY.drop('Y', axis=1).to_numpy(copy=True)
        train_Y = train_XY.loc[:, 'Y'].to_numpy(copy=True)
        # dev_X = dev_set.drop('Y', axis=1).to_numpy(copy=True)
        # dev_Y = dev_set.loc[:, 'Y'].to_numpy(copy=True)

        # dev_acc_lin = []
        # dev_acc_gauss = []
        # dev_f1_score_lin = []
        # dev_f1_score_gauss = []
        # train_acc = []
        # cost_c = [1e-5, 1e-3, 1, 5, 10]
        # cwd = os.getcwd()
        # for c in cost_c:
        #     # m = svm_train(train_Y, train_X_normed, '-t 0')
        #
        # # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        # fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
        # # line1 = ax1.plot(cost_c, dev_acc_lin, label='validation accuracy linear SVM with c')
        # # ax1.legend()
        # # ax1.set_xlabel("parameter C")
        # # ax1.set_ylabel("accuracy")
        # # ax1.set_title("accuracy vs C for linear SVM")
        # #
        # # line2 = ax2.plot(cost_c, dev_acc_gauss, label='validation accuracy with gaussian kernel with c')
        # # ax2.legend()
        # # ax2.set_xlabel("parameter C")
        # # ax2.set_ylabel("accuracy")
        # # ax2.set_title("accuracy vs C for Gaussina SVM")
        #
        # # f1_scores
        # line3 = ax3.plot(cost_c, dev_f1_score_lin, label='validation f1-score linear SVM with c')
        # ax3.legend()
        # ax3.set_xlabel("parameter C")
        # ax3.set_ylabel("f1-score (weighted average)")
        # ax3.set_title("f1-score vs C for linear SVM")
        #
        # line4 = ax4.plot(cost_c, dev_f1_score_gauss, label='validation accuracy with gaussian kernel with c')
        # ax4.legend()
        # ax4.set_xlabel("parameter C")
        # ax4.set_ylabel("f1-score (weighted average)")
        # ax4.set_title("f1-score vs C for Gaussina SVM")
        #
        # plt.tight_layout()
        # fig_name = os.path.join(cwd, "lin_gauss_svm_accuracy_with_c")
        # plt.savefig(fig_name, format='png')
        # plt.show()

        # parameter_candidates = [{'C': [1e-7, 1e-5, 1e-2, 1, 5, 10], 'kernel': ['linear']},
        #                        {'C': [1e-7, 1e-5, 1e-2, 1, 5, 10], 'gamma': [1e-5, 1e-3, 1e-2, 1e-1, 1], 'kernel': ['rbf']},
        #                        ]

        parameter_candidates = [{'C': [1e-2, 1, 1.5], 'kernel': ['linear']},
                                {'C': [1e-2, 1, 1.5], 'gamma': [1e-15, 1e-10, 10, 1000], 'kernel': ['rbf']},
                                ]
        clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
        clf.fit(train_X, train_Y)
        # View the accuracy score
        print('Best score for data1:', clf.best_score_)

        # View the best parameters for the model found using grid search
        print('Best C:', clf.best_estimator_.C)
        print('Best Kernel:', clf.best_estimator_.kernel)
        print('Best Gamma:', clf.best_estimator_.gamma)


if train_test == 0:
        # training the svm classifier
        if train_test_lib == "libsvm":
                # train the model with new set
                train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
                # ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
                data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")

                # the sequence will be known by the first frame in the sequence
                # # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
                seq_train_XY = train_XY_to_seq_XY(train_XY_reduced, y_at_start_of_episode=-1,
                                                  safe_to_csv=True, root_path=start_path, file_name="seq_train_XY")

                train_XY = seq_train_XY
                # train_XY = pd.read_csv(os.path.join(start_path, "seq_train_XY"))

                # dev_set = train_XY.sample(frac=0.05, replace=False, random_state=1, axis=0)
                # train_XY = train_XY.drop(labels=dev_set.index)
                train_XY = train_XY.sample(frac=0.90, replace=False, random_state=1, axis=0)
                train_XY_size = train_XY.shape[0]
                train_X = train_XY.drop('Y', axis=1).to_numpy(copy=True)
                train_Y = train_XY.loc[:, 'Y'].to_numpy(copy=True)

                # dev_X = dev_set.drop('Y', axis=1).to_numpy(copy=True)
                # dev_Y = dev_set.loc[:, 'Y'].to_numpy(copy=True)

                # the classifier linear
                param_string_lin = '-h 0 -t 0 -c 0.01'
                m_lin = svm_train(train_Y, train_X, param_string_lin)
                p_label_lin, p_acc_lin, p_val_lin, = svm_predict(dev_Y, dev_X, m_lin)
                dev_acc_lin.append(p_acc_lin[0])
                dev_f1_score_lin.append(f1_score(dev_Y, p_label_lin, average='binary'))
                print('dev_acc_lin', dev_acc_lin)
                print('dev_f1_score_lin', dev_f1_score_lin)
                svm_save_model('libsvm_lin.model', m)

                # # gaussing classifier
                # param_string_gauss = '-h 0 -g 0.05 -t 2 -c %f' % c
                # m_gauss = svm_train(train_Y, train_X, param_string_gauss)
                # p_label_gauss, p_acc_gauss, p_val_gauss, = svm_predict(dev_Y, dev_X, m_gauss)
                # dev_acc_gauss.append(p_acc_gauss[0])
                # dev_f1_score_gauss.append(f1_score(dev_Y, p_label_gauss, average='binary'))
                # svm_save_model('libsvm_gauss.model', m)
                # print('dev_acc_gauss', dev_acc_gauss)
                # print('dev_f1_score_gauss', dev_f1_score_gauss)

        elif train_test_lib == "sklearn":
                # train the model with new set
                train_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_train_XY_reduced")
                # ipca = pickle_load(root_path=start_path, file_name="pickle_ipca")
                data_scaler = pickle_load(root_path=start_path, file_name="pickle_data_scaler")

                # # to measure reconstruction error we need the original dataset
                # train_X_reduced = train_XY_reduced.drop('Y', axis = 1).values
                # reconstructed_X = ipca.inverse_transform(train_X_reduced)
                # train_XY_original = load_all_dataset_XY(root_path=start_path, top_n_episodes=2)
                # error_curr = mean_squared_error(reconstructed_X, train_XY_original.drop('Y', axis=1).values)
                # print("sequential error using function", error_curr)

                # the sequence will be known by the first frame in the sequence
                # # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
                seq_train_XY = train_XY_to_seq_XY(train_XY_reduced, y_at_start_of_episode=-1,
                                                  safe_to_csv=True, root_path=start_path, file_name="seq_train_XY")

                train_XY = seq_train_XY
                train_XY = pd.read_csv(os.path.join(start_path, "seq_train_XY"))

                # dev_set = train_XY.sample(frac=0.05, replace=False, random_state=1, axis=0)
                # train_XY = train_XY.drop(labels=dev_set.index)
                train_XY = train_XY.sample(frac=0.90, replace = False, random_state=1, axis=0)
                train_XY_size = train_XY.shape[0]
                train_X = train_XY.drop('Y', axis=1).to_numpy(copy=True)
                train_Y = train_XY.loc[:, 'Y'].to_numpy(copy=True)

                # dev_X = dev_set.drop('Y', axis=1).to_numpy(copy=True)
                # dev_Y = dev_set.loc[:, 'Y'].to_numpy(copy=True)

                best_clf = svm.SVC(C=0.01, kernel='linear', class_weight='balanced', random_state=0)
                best_clf.fit(train_X, train_Y)

                train_mean_score = best_clf.score(train_X, train_Y)
                print('train mean accuracy', train_mean_score)
                # pickle_store(best_clf, root_path=start_path, file_name="svm_best_clf"+"%d" %train_XY_size)
                pickle_store(best_clf, root_path=start_path, file_name="svm_best_clf")

                # predict accuracy
                train_Y_pred = best_clf.predict(train_X)
                train_f1_score = f1_score(train_Y, train_Y_pred, average='binary')
                print('train f1-score accuracy', train_f1_score)

                # dev_Y_pred = best_clf.predict(dev_X)
                # dev_f1_score = f1_score(dev_Y, dev_Y_pred, average='binary')
                # print('dev f1-score accuracy', dev_f1_score)

if train_test == 1:
        " predict test accuracy"
        if train_test_lib == 'libsvm':
                # load the test_XY_reduced to be made into sequence
                test_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_test_XY_reduced")

                # the sequence will be known by the first frame in the sequence
                # # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
                seq_test_XY = train_XY_to_seq_XY(test_XY_reduced, y_at_start_of_episode=-1,
                                                 safe_to_csv=True, root_path=start_path, file_name="seq_test_XY")
                test_XY = seq_test_XY
                # test_XY = pd.read_csv(os.path.join(start_path, "seq_test_XY"))

                # test_XY = test_XY.sample(frac=0.2, replace=False, random_state=1, axis=0)
                # print(test_XY.shape)
                test_X = test_XY.drop('Y', axis=1).to_numpy(copy=True)
                test_Y = test_XY.loc[:, 'Y'].to_numpy(copy=True)

                # test set prediction
                # best_clf = svm_load_model('libsvm_lin.model')

                ## using sklearr
                best_clf = pickle_load(root_path=start_path, file_name="svm_best_clf")
                # predict accuracy
                test_Y_pred = best_clf.predict(test_X)
                test_f1_score = f1_score(test_Y, test_Y_pred, average='binary')
                print('test f1-score accuracy', test_f1_score)


        elif train_test_lib == 'sklearn':

                # load the test_XY_reduced to be made into sequence
                test_XY_reduced = pickle_load(root_path=start_path, file_name="pickle_test_XY_reduced")

                # the sequence will be known by the first frame in the sequence
                # # the train_XY_reduced stores all the episode, separated by -1 (y_at_start_of_episode) in 'Y' label.
                seq_test_XY = train_XY_to_seq_XY(test_XY_reduced, y_at_start_of_episode=-1,
                                                  safe_to_csv=True, root_path=start_path, file_name="seq_test_XY")
                test_XY = seq_test_XY
                # test_XY = pd.read_csv(os.path.join(start_path, "seq_test_XY"))

                # test set prediction
                best_clf = pickle_load(root_path=start_path, file_name="svm_best_clf")

                # test_XY = test_XY.sample(frac=0.2, replace=False, random_state=1, axis=0)
                # print(test_XY.shape)
                test_X = test_XY.drop('Y', axis=1).to_numpy(copy=True)
                test_Y = test_XY.loc[:, 'Y'].to_numpy(copy=True)


                # predict accuracy
                test_Y_pred = best_clf.predict(test_X)
                test_f1_score = f1_score(test_Y, test_Y_pred, average='binary')
                print('test f1-score accuracy', test_f1_score)

