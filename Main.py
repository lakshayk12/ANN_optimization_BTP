import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import GOA
import PSO
import settings


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
    # one hot labeling classes needs to be stated as 0 1 2 . . .
    unique_classes = np.unique(Y)
    settings.no_of_classes = len(unique_classes)
    print(settings.no_of_classes, "Classes.")
    one_hot_labels = np.zeros((Y.shape[0], len(unique_classes)))
    for i in range(one_hot_labels.shape[0]):
        one_hot_labels[i, int(Y[i, 0])] = 1
    Y = one_hot_labels
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test


def scale(x_train, x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test


def verify(x_test, y_test, optimal_solution):
    # print("\n\nOPTIMAL SOLUTION:\n", optimal_solution)
    print("\nOPTIMAL ERROR: \n", optimal_solution[0])
    print("\nOPTIMAL ARCHITECTURE:\n", optimal_solution[1])
    print("\nOPTIMAL WEIGHT MATRIX:\n", optimal_solution[2])

    feature_set = optimal_solution[1][0]
    updated_x_test = GOA.updated_X(x_test, feature_set)
    output, error = PSO.generate_output_and_error(updated_x_test, y_test, optimal_solution[2], optimal_solution[1][3],
                                                  optimal_solution[1][4])
    print("\n*** RESULTS ON TESTING DATA ***")
    print("\nError:", error)
    print("Prediction:", output.argmax(axis=1))
    print("Accuracy:", accuracy_score(y_test.argmax(axis=1), output.argmax(axis=1)))


if __name__ == '__main__':
    # old_dim = (4, 10)
    # new_dim = (5, 2)
    # old_matrix = np.random.randn(old_dim[0], old_dim[1])
    # print("OLD :\n", old_matrix)
    # previous_gh = [[1, 1, 1, 1, 0]]
    # gh = [[1, 1, 1, 1, 1]]
    # GOA.make_similar_matrix(old_dim, new_dim, old_matrix, gh, previous_gh)
    # exit()

    start_time = time.time()
    # x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # y_train = [[1, 0], [0, 1], [0, 1], [0, 1]]
    # settings.no_of_classes = 2
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = x_train
    # y_test = y_train

    x_train, x_test, y_train, y_test = get_dataset_ready("datasets/Parkinsons.csv")
    x_train, x_test = scale(x_train, x_test)
    optimal_solution = GOA.algorithm(x_train, y_train)  # accuracy, grasshopper, corresponding_weights
    verify(x_test, y_test, optimal_solution)
    print("\nExecution Time:", time.time() - start_time)
    exit(0)
