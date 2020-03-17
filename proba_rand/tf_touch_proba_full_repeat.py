#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""

# Data Holders
import os
import sys
import errno
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Cheat and import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tf_nn_classifier import DNNClassifier
from user_touches import TouchUserPopulation


class generic():
    pass


class Classifier:
    def __init__(self, clf_model, clf_param):
        self.model = clf_model
        self.model_param = clf_param

    def train_classifier(self, train_d):
        train_data = np.concatenate([train_d[0], train_d[1]])
        n0 = len(train_d[0])
        n1 = len(train_d[1])
        train_label = np.concatenate([np.ones(n0), np.zeros(n1)])
        clf = self.model(**self.model_param)
        clf.fit(train_data, train_label)
        self.classifier = clf

    def test_classifier(self, test_d):
        test_data = np.concatenate([test_d[0], test_d[1]])
        n0 = len(test_d[0])
        n1 = len(test_d[1])
        test_label = np.concatenate([np.ones(n0), np.zeros(n1)])

        result = self.classifier.predict(test_data)
        self.clfs_result.append((result, test_label))


# Main function
# Define this a one iteration loop
def main_run(selection):
    resample_users = True

    # Create a population statistic for the data
    if resample_users:
        data_loc = "./../extracted_touch_features.csv"
        random_loc = "./../random_dump.csv"

        n_feat = 27

        a = TouchUserPopulation(data_loc, n_feat)

        a.normalize_data()
        a.split_user_data(0.3)

        rand_data = pd.read_csv(random_loc, header=0, index_col=0)
        temp_a = pd.get_dummies(rand_data['udlr_flag'])
        rand_data[['1_f', '2_f', '3_f', '4_f']] = temp_a
        rand_data.drop('udlr_flag', axis=1, inplace=True)
        rand_data = rand_data.astype(float)

        rand_data = a.normalize_external(rand_data.values)

        clf_titles = ['RNDF',
                      'LINEAR SVM',
                      'RBF SVM',
                      'ONECLASS RBF SVM',
                      'DNN'
                      ]

        clf_models = [RandomForestClassifier,
                      svm.SVC,
                      svm.SVC,
                      svm.OneClassSVM,
                      DNNClassifier
                      ]

        clf_params = [{'n_jobs': -1, 'n_estimators': 100},
                      {'kernel': 'linear', 'C': 1E4, 'probability': True},
                      {'kernel': 'rbf', 'C': 1E4, 'probability': True},
                      {'kernel': 'rbf', 'nu': 0.1, 'gamma': 0.1, 'probability': True},
                      {'input_shape': (1, n_feat), 'num_epochs': 500,
                       'temp_dir': None},
                      ]

        clf_param = clf_params[selection]
        clf_model = clf_models[selection]
        clf_title = clf_titles[selection]

        cover_data = np.random.rand(1000000, n_feat)

        step = 0.01
        scale = np.arange(0, 1+step, step)

    FPR_holder = []
    TPR_holder = []
    AR_holder = []
    RAR_holder = []

    run_tmp_dir = tempfile.mkdtemp(dir="./models/")  # create dir

    def binary_threshold_counter(a, scale):
        arr = np.array(a)[:, 1]
        return [np.sum(arr > t)/arr.size for t in scale]

    for u in sorted(a.users.keys()):
        print(u)
        target_data, other_data = a.get_train_sets(u, concatenate=False)
        target_test_data, other_test_data = a.get_test_sets(u,
                                                            concatenate=False)

        if 'DNN' in clf_title:
            try:
                tmp_dir = tempfile.mkdtemp(dir=run_tmp_dir)  # create dir
                clf_param['temp_dir'] = tmp_dir
                clf = DNNClassifier(**clf_param)
                clf.train_classifier([target_data, other_data])
                T = clf.predict_proba(target_test_data)
                F = clf.predict_proba(other_test_data)
                Z = clf.predict_proba(cover_data)
                R = clf.predict_proba(rand_data)

                print(T)

                TPR = binary_threshold_counter(T, scale)
                FPR = binary_threshold_counter(F, scale)
                AR = binary_threshold_counter(Z, scale)
                RAR = binary_threshold_counter(R, scale)
                AR_holder.append(AR)
                TPR_holder.append(TPR)
                FPR_holder.append(FPR)
                RAR_holder.append(RAR)

                del clf
            finally:
                pass

        else:
            clf = Classifier(clf_model, clf_param)
            clf.train_classifier([target_data, other_data])
            T = clf.classifier.predict_proba(target_test_data)
            F = clf.classifier.predict_proba(other_test_data)
            Z = clf.classifier.predict_proba(cover_data)
            R = clf.classifier.predict_proba(rand_data)

            TPR = binary_threshold_counter(T, scale)
            FPR = binary_threshold_counter(F, scale)
            AR = binary_threshold_counter(Z, scale)
            RAR = binary_threshold_counter(R, scale)
            AR_holder.append(AR)
            TPR_holder.append(TPR)
            FPR_holder.append(FPR)
            RAR_holder.append(RAR)
            del clf

    return (TPR_holder, FPR_holder, AR_holder, sorted(a.users.keys()), RAR_holder)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir_p(path):
    try:
        shutil.rmtree("./models/")  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            # ENOENT - no such file or directory
            raise  # re-raise exception


if __name__ == "__main__":
    import pickle

    clf_titles = ['rndf',
                  'linsvm',
                  'rbfsvm',
                  'oneclass',
                  'dnn'
                  ]

    if len(sys.argv) < 2:
        selection = 0
    else:
        selection = int(sys.argv[1])

    results_holder = []
    for _ in range(1):
        mkdir_p("./models/")

        mkdir_p("./{}_probaresults/".format(clf_titles[selection]))

        results_holder.append(main_run(selection))
    tmp_file = tempfile.mkstemp(suffix=".pickle",
                                dir="./{}_probaresults/".format(
                                        clf_titles[selection]))
    pickle.dump(results_holder, open(tmp_file[1], 'wb'))
