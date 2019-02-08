import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import GOA
import PSO
import random


def inbuilt_algo(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(max_iter=10000)
    clf.fit(x_train, y_train)
    print("Inbuilt", accuracy_score(y_test, clf.predict(x_test)))


def get_dataset_ready(filename):
    data = pd.read_csv(filename, sep=",", header=None)
    data = data.values
    X = data[:, 0:len(data[0]) - 1]
    Y = data[:, len(data[0]) - 1]
    Y = np.reshape(Y, newshape=(len(Y), 1))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test


def scale(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test


if __name__ == '__main__':
    start_time = time.time()
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [1], [1], [1]]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = x_train
    y_test = y_train
    # x_train, x_test, y_train, y_test = get_dataset_ready('Parkinsons.csv')
    # x_train, x_test = scale(x_train, x_test)
    # inbuilt_algo(x_train, x_test, y_train, y_test)
    PSO.model(x_train, x_test, y_train, y_test, len(x_train[0]), 2, 2, 2)
    print("Execution Time:", time.time() - start_time)
    exit()