#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""

import numpy as np
# from sklearn.model_selection import train_test_split
import tensorflow as tf


class DNNClassifier:
    def __init__(self, input_shape, num_epochs, temp_dir, valid_split=0.2):
        self.input_shape = input_shape
        self.feat_col = [tf.feature_column.numeric_column("x",
                                                          shape=input_shape)]
        self.valid_split = valid_split
        self.num_epochs = num_epochs
        self.temp_dir = temp_dir

    def train_classifier(self, train_d):

        data = np.concatenate([train_d[0], train_d[1]])
        n0 = len(train_d[0])
        n1 = len(train_d[1])
        label = np.concatenate([np.ones(n0), np.zeros(n1)])

        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=self.feat_col,
            hidden_units=[256, 512, 512, 512, 256, 128],
            n_classes=2,
            optimizer='Adagrad',
            model_dir=self.temp_dir
        )

        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            y=label,
            num_epochs=None,
            batch_size=50,
            shuffle=True
        )

        self.classifier.train(input_fn=train_input_fn, steps=5000)

    def predict(self, data):

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            num_epochs=1,
            shuffle=False
        )

        # Evaluate accuracy
        predict = self.classifier.predict(input_fn=test_input_fn)
        pred = list(predict)
        return np.array([i['class_ids'][0] for i in pred])

    def predict_proba(self, data):

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            num_epochs=1,
            shuffle=False
        )

        # Evaluate accuracy
        predict = self.classifier.predict(input_fn=test_input_fn)
        pred = list(predict)
        return np.array([i['probabilities'] for i in pred])

    def test_classifier(self, test_d):
        test_data = np.concatenate([test_d[0], test_d[1]])
        n0 = len(test_d[0])
        n1 = len(test_d[1])
        test_label = np.concatenate([np.ones(n0), np.zeros(n1)])

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_label,
            num_epochs=1,
            shuffle=False
        )

        predict = self.classifier.predict(input_fn=test_input_fn)
        result = np.array(predict)
        self.clfs_result.append((result, test_label))

        # Evaluate accuracy
        accuracy_score = self.classifier.evaluate(input_fn=test_input_fn)
        print("\nTest Accuracy: {0:f}%\n".format(
                accuracy_score['Accuracy']*100))
