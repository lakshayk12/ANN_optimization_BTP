import numpy as np
from sklearn import datasets
import math
import random
import copy
import math
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
from numba import jit
import settings


@jit(nopython=True)
def sig(z):
    return 1 / (1 + np.exp(z))


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


def softmax(A):
    expA = np.exp(A)
    return expA / np.sum(expA, axis=1, keepdims=True)


def relu(x):
    return x * (x > 0)


def tanh(x):
    return np.tanh(x)


def arctan(x):
    return np.arctan(x)


def generate_output_and_error(X, Y, W, tf1, tf2):
    wh1 = W[0]
    bh1 = W[1]
    wh2 = W[2]
    bh2 = W[3]
    wo = W[4]
    bo = W[5]
    output0 = X  # output of input layer

    inputHidden1 = np.dot(output0, wh1) + bh1

    # hidden layer output 1
    if tf1[0] == 0 and tf1[1] == 0:
        outputHidden1 = sig(inputHidden1)
    if tf1[0] == 0 and tf1[1] == 1:
        outputHidden1 = tanh(inputHidden1)
    if tf1[0] == 1 and tf1[1] == 0:
        outputHidden1 = relu(inputHidden1)
    if tf1[0] == 1 and tf1[1] == 1:
        outputHidden1 = arctan(inputHidden1)

    inputHidden2 = np.dot(outputHidden1, wh2) + bh2

    # hidden layer output 2
    if tf2[0] == 0 and tf2[1] == 0:
        outputHidden2 = sig(inputHidden2)
    if tf2[0] == 0 and tf2[1] == 1:
        outputHidden2 = tanh(inputHidden2)
    if tf2[0] == 1 and tf2[1] == 0:
        outputHidden2 = relu(inputHidden2)
    if tf2[0] == 1 and tf2[1] == 1:
        outputHidden2 = arctan(inputHidden2)

    inputForOutputLayer = np.dot(outputHidden2, wo) + bo

    # final output layer's output
    output = softmax(inputForOutputLayer)

    # calculate error
    curr_error = np.sum(-Y * np.log(output))
    return output, curr_error


def give_N_weight_chromosomes(n, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                              no_of_output_neurons):
    weights = []
    for i in range(n):
        W = initialize_weight_chromosome(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                         no_of_output_neurons)
        weights.append(W)
    return np.array(weights)


def model(x_train, y_train, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2, no_of_output_neurons,
          tf1, tf2):
    # initialize random population
    weights = give_N_weight_chromosomes(settings.pso_population_size, no_of_input_neurons, no_of_hidden_neurons1,
                                        no_of_hidden_neurons2,
                                        no_of_output_neurons)
    c1 = 1.48  # const
    c2 = 1.48  # const
    w = 0.729  # inertia weight
    wMax = 0.9  # max inertia weight
    wMin = 0.5  # min inertia weight
    dt = 0.8  # Velocity retardation factory
    Max_iteration = settings.pso_max_iteration
    best = [math.inf, -1]  # error, weights

    velocities = [0 for i in range(30)]
    local_best_swarm1 = [math.inf, -1]  # error, weight
    local_best_swarm2 = [math.inf, -1]  # error, weight
    local_best_swarm3 = [math.inf, -1]  # error, weight

    for it in range(Max_iteration):
        # swarm 1
        for i in range(0, 10):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if curr_error < local_best_swarm1[0]:
                local_best_swarm1[0] = curr_error
                local_best_swarm1[1] = copy.deepcopy(weights[i])

        for i in range(10, 20):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if curr_error < local_best_swarm2[0]:
                local_best_swarm2[0] = curr_error
                local_best_swarm2[1] = copy.deepcopy(weights[i])

        for i in range(20, 30):
            output, curr_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if curr_error < local_best_swarm3[0]:
                local_best_swarm3[0] = curr_error
                local_best_swarm3[1] = copy.deepcopy(weights[i])

        # update global best of all swarms
        if local_best_swarm1[0] < best[0]:
            best[0] = local_best_swarm1[0]
            best[1] = copy.deepcopy(local_best_swarm1[1])
        if local_best_swarm2[0] < best[0]:
            best[0] = local_best_swarm2[0]
            best[1] = copy.deepcopy(local_best_swarm2[1])
        if local_best_swarm3[0] < best[0]:
            best[0] = local_best_swarm3[0]
            best[1] = copy.deepcopy(local_best_swarm3[1])
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
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

        # swarm 3
        for i in range(20, 30):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm3[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

    output, curr_error = generate_output_and_error(x_train, y_train, best[1], tf1, tf2)  # best[1] is optimal weights
    accuracy = accuracy_score(y_train.argmax(axis=1), output.argmax(axis=1))
    return accuracy, best[1]  # accuracy and best_weights
