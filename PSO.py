import numpy as np
from sklearn import datasets
import math
import random
import copy
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
from numba import jit


@jit(nopython=True)
def sig(z):
    return 1 / (1 + np.exp(z))


def threshold(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i][j] > 0.5:
                z[i][j] = 1
            else:
                z[i][j] = 0
    return z


def initialize_weight_chromosome(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                 no_of_output_neurons):
    # for Hidden layer 1
    wh1 = np.random.randn(no_of_input_neurons, no_of_hidden_neurons1)
    bh1 = np.random.randn(1, no_of_hidden_neurons1)

    # for Hidden layer 2
    wh2 = np.random.randn(no_of_hidden_neurons1, no_of_hidden_neurons2)
    bh2 = np.random.randn(1, no_of_hidden_neurons2)

    # for Output layer
    wo = np.random.randn(no_of_hidden_neurons2, no_of_output_neurons)
    bo = np.random.randn(1, no_of_output_neurons)

    W = [wh1, bh1, wh2, bh2, wo, bo]
    return np.array(W)


def generate_output_and_error(X, Y, W):
    wh1 = W[0]
    bh1 = W[1]
    wh2 = W[2]
    bh2 = W[3]
    wo = W[4]
    bo = W[5]
    output0 = X  # output of input layer

    inputHidden1 = np.dot(output0, wh1) + bh1
    outputHidden1 = sig(inputHidden1)  # hidden layer output 1

    inputHidden2 = np.dot(outputHidden1, wh2) + bh2
    outputHidden2 = sig(inputHidden2)  # hidden layer output 2

    inputForOutputLayer = np.dot(outputHidden2, wo) + bo
    output = sig(inputForOutputLayer)  # final output layer's output
    curr_error = abs(np.sum(np.power(output - Y, 2))) / len(X)
    return output, curr_error


def give_N_weight_chromosomes(n, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                              no_of_output_neurons):
    weights = []
    for i in range(n):
        W = initialize_weight_chromosome(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                         no_of_output_neurons)
        weights.append(W)

    return np.array(weights)


def model(x_train, x_test, y_train, y_test, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
          no_of_output_neurons):
    # initialize random population
    weights = give_N_weight_chromosomes(30, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                        no_of_output_neurons)
    c1 = 2  # const
    c2 = 2  # const
    w = 2  # inertia weight
    wMax = 0.9  # max inertia weight
    wMin = 0.5  # min inertia weight
    dt = 0.8  # Velocity retardation factor
    Max_iteration = 10000
    best = [100000, -1]  # error, weight

    velocities = [0 for i in range(30)]
    local_best_swarm1 = [1000, -1]  # error, weight
    local_best_swarm2 = [1000, -1]  # error, weight
    local_best_swarm3 = [1000, -1]  # error, weight

    for it in range(Max_iteration):
        # swarm 1
        for i in range(0, 10):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i])
            if curr_error < local_best_swarm1[0]:
                local_best_swarm1[0] = curr_error
                local_best_swarm1[1] = weights[i]

        for i in range(10, 20):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i])
            if curr_error < local_best_swarm2[0]:
                local_best_swarm2[0] = curr_error
                local_best_swarm2[1] = weights[i]

        for i in range(20, 30):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i])
            if curr_error < local_best_swarm3[0]:
                local_best_swarm3[0] = curr_error
                local_best_swarm3[1] = weights[i]

        # update global best of all swarms
        if local_best_swarm1[0] < best[0]:
            best[0] = local_best_swarm1[0]
            best[1] = local_best_swarm1[1]
        if local_best_swarm2[0] < best[0]:
            best[0] = local_best_swarm2[0]
            best[1] = local_best_swarm2[1]
        if local_best_swarm3[0] < best[0]:
            best[0] = local_best_swarm3[0]
            best[1] = local_best_swarm3[1]

        # swarm 1
        for i in range(0, 10):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm1[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]

            w = wMin - i * (wMax - wMin) / Max_iteration

        # swarm 2
        for i in range(10, 20):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm2[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]

            w = wMin - i * (wMax - wMin) / Max_iteration

        # swarm 3
        for i in range(20, 30):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm3[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

    print("Final")
    output, curr_error = generate_output_and_error(x_train, y_train, best[1])
    print(output)
    y_pred = threshold(copy.deepcopy(output))
    print("Accuracy: ", accuracy_score(y_train, y_pred))
