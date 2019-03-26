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
    if type(W) == int:
        print(W)
        exit()
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
    cross_ent_error = np.sum(-Y * np.log(output)) / (len(X) * settings.no_of_classes)
    return output, cross_ent_error


def give_N_weight_chromosomes(n, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                              no_of_output_neurons, guessed_weights):
    weights = []
    if guessed_weights is not None:
        # print("Weights were guessed already!")
        for i in range(n):
            weights.append(
                initialize_weight_chromosome(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                             no_of_output_neurons) + guessed_weights)
    else:
        for i in range(n):
            W = initialize_weight_chromosome(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2,
                                             no_of_output_neurons)
            weights.append(W)
    return np.array(weights)


def cal_penalty(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2):
    penalty1 = (no_of_input_neurons + no_of_hidden_neurons1 + no_of_hidden_neurons2) / settings.max_no_of_neurons
    penalty2 = 0
    # return penalty1 + penalty2
    return 0


def model(x_train, y_train, no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2, no_of_output_neurons,
          tf1, tf2, guessed_weights=None):
    # initialize random population
    population_size = settings.pso_population_size
    weights = give_N_weight_chromosomes(population_size, no_of_input_neurons, no_of_hidden_neurons1,
                                        no_of_hidden_neurons2,
                                        no_of_output_neurons, guessed_weights)
    c1 = 1.48  # const
    c2 = 1.48  # const
    w = 0.729  # inertia weight
    wMax = 0.9  # max inertia weight
    wMin = 0.5  # min inertia weight
    dt = 0.8  # Velocity retardation factory
    Max_iteration = settings.pso_max_iteration
    best = [math.inf, -1]  # error, weights
    first_run = True
    best_first_update = True

    velocities = [0 for i in range(30)]
    local_best_swarm1 = [math.inf, -1]  # error, weight
    local_best_swarm2 = [math.inf, -1]  # error, weight
    local_best_swarm3 = [math.inf, -1]  # error, weight

    total_penalty = cal_penalty(no_of_input_neurons, no_of_hidden_neurons1, no_of_hidden_neurons2)

    swarm_size = int(population_size / 3)

    for it in range(Max_iteration):
        # swarm 1
        for i in range(0, swarm_size):
            output, cross_ent_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if first_run or cross_ent_error < local_best_swarm1[0]:
                local_best_swarm1[0] = cross_ent_error
                local_best_swarm1[1] = copy.deepcopy(weights[i])
                first_run = False

        for i in range(swarm_size, int(swarm_size * 2)):
            output, cross_ent_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if first_run or cross_ent_error < local_best_swarm2[0]:
                local_best_swarm2[0] = cross_ent_error
                local_best_swarm2[1] = copy.deepcopy(weights[i])
                first_run = False

        for i in range(int(swarm_size * 2), int(swarm_size * 3)):
            output, cross_ent_error = generate_output_and_error(x_train, y_train, weights[i], tf1, tf2)
            if first_run or cross_ent_error < local_best_swarm3[0]:
                local_best_swarm3[0] = cross_ent_error
                local_best_swarm3[1] = copy.deepcopy(weights[i])
                first_run = False

        # update global best of all swarms
        if best_first_update or local_best_swarm1[0] < best[0]:
            best[0] = local_best_swarm1[0]
            best[1] = copy.deepcopy(local_best_swarm1[1])
            best_first_update = False

        if best_first_update or local_best_swarm2[0] < best[0]:
            best[0] = local_best_swarm2[0]
            best[1] = copy.deepcopy(local_best_swarm2[1])
            best_first_update = False

        if best_first_update or local_best_swarm3[0] < best[0]:
            best[0] = local_best_swarm3[0]
            best[1] = copy.deepcopy(local_best_swarm3[1])
            best_first_update = False

        # swarm 1
        for i in range(0, swarm_size):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm1[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

        # swarm 2
        for i in range(swarm_size, int(swarm_size * 2)):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm2[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

        # swarm 3
        for i in range(int(swarm_size * 2), int(swarm_size * 3)):
            velocities[i] = w * velocities[i] + c1 * random.random() * (
                    local_best_swarm3[1] - weights[i]) + c2 * random.random() * (best[1] - weights[i])
            weights[i] = (dt * velocities[i]) + weights[i]
            w = wMin - i * (wMax - wMin) / Max_iteration

    output, curr_error = generate_output_and_error(x_train, y_train, best[1], tf1, tf2)  # best[1] is optimal weights
    accuracy = accuracy_score(y_train.argmax(axis=1), output.argmax(axis=1))
    print("PSO Fitness (CEE):", best[0], "| PSO Accuracy:", accuracy, end=" ")
    return best[0], best[1]  # CEE and best_weights
