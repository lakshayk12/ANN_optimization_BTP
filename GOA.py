import copy
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score

minimum_no_of_hidden_neuron = 2


def give_a_random_solution(no_of_features):
    global max_number_of_hidden_neuron
    max_number_of_hidden_neuron = 2 * (no_of_features + 1)
    vector = [[0 for i in range(no_of_features)]]
    n = math.ceil(math.log(max_number_of_hidden_neuron, 2))

    no_of_selected_features = random.randint(np.floor(no_of_features / 2), no_of_features)
    count = 0

    while count < no_of_selected_features:
        current_set = random.randint(1, no_of_features) - 1
        if vector[0][current_set] != 1:
            vector[0][current_set] = 1
            count += 1

    # For Hidden layer 1
    no_of_hidden_layer_neuron = random.randint(5, max_number_of_hidden_neuron)
    binary_str = "{0:b}".format(no_of_hidden_layer_neuron)
    if len(binary_str) != n:
        to_append = n - len(binary_str)
        while to_append > 0:
            binary_str = '0' + binary_str
            to_append -= 1

    vector.append(binary_str)

    # For Hidden layer 2
    no_of_hidden_layer_neuron = random.randint(5, max_number_of_hidden_neuron)
    binary_str = "{0:b}".format(no_of_hidden_layer_neuron)
    if len(binary_str) != n:
        to_append = n - len(binary_str)
        while to_append > 0:
            binary_str = '0' + binary_str
            to_append -= 1

    vector.append(binary_str)

    # Transfer Function for Hidden Layer 1
    tf_spool = [[0, 0], [0, 1], [1, 0], [1, 1]]
    current_set = random.randint(0, 3)
    vector.append(np.array(tf_spool[current_set]))

    # Transfer Function for Hidden Layer 2
    current_set = random.randint(0, 3)
    vector.append(np.array(tf_spool[current_set]))
    return np.array(vector)


def give_N_random_solutions(n, no_of_features):
    solutions = []
    for i in range(n):
        solutions.append(give_a_random_solution(no_of_features))
    return np.array(solutions)


def updated_X(X, binary_list):
    first_run = False

    if 1 not in binary_list:
        return X

    for i in range(len(binary_list)):
        if binary_list[i] == 1:
            if first_run is False:
                new_x = copy.deepcopy(X[:, i])
                new_x = np.reshape(new_x, newshape=(len(new_x), 1))
                first_run = True
            else:
                new_x = np.concatenate((new_x, np.reshape(X[:, i], newshape=(len(copy.deepcopy(X[:, i])), 1))), axis=1)
    return new_x


def distance(gh1, gh2):
    dist = 0
    for i in range(len(gh1[0])):
        dist += (gh1[0][i] ^ gh2[0][i])
    for i in range(len(gh1[1])):
        dist += (int(gh1[1][i]) ^ int(gh2[1][i]))
    return dist


def normalize_distance(curr_search_agent, best_search_agent):
    normalization_factor = random.randint(2, 4)
    # print("Normakdkf", normalization_factor)
    dist = distance(curr_search_agent, best_search_agent)
    n = len(curr_search_agent[0]) + len(curr_search_agent[1])
    visit = [0 for i in range(n)]

    save = curr_search_agent[1]

    while dist > normalization_factor:
        index = random.randint(1, n)
        if visit[index - 1] == 1:
            continue

        visit[index - 1] = 1

        if index <= len(curr_search_agent[0]) and curr_search_agent[0][index - 1] != best_search_agent[0][index - 1]:
            curr_search_agent[0][index - 1] = best_search_agent[0][index - 1]
            dist -= 1
        elif index > len(curr_search_agent[0]) and curr_search_agent[1][index - len(curr_search_agent[0]) - 1] != \
                best_search_agent[1][
                    index - len(curr_search_agent[0]) - 1]:
            str = curr_search_agent[1]
            mutate_index = index - len(curr_search_agent[0]) - 1
            temp = str[0:mutate_index] + best_search_agent[1][mutate_index] + str[mutate_index + 1:]
            curr_search_agent[1] = temp
            dist -= 1

    if int(curr_search_agent[1], 2) < minimum_no_of_hidden_neuron:
        curr_search_agent[1] = save
    return curr_search_agent


def update_position(gh, change_value):
    if change_value == 0:
        return gh
    n = len(gh[0]) + len(gh[1])
    visit = [0 for i in range(n)]
    backup = gh[1]

    while change_value > 0:
        rand = random.uniform(0, 1)
        if rand < 0.5:
            index = random.randint(1, len(gh[0]))
        else:
            index = random.randint(len(gh[0]) + 1, n)
        if visit[index - 1] == 1:
            continue
        # print("INDEX------", index)
        visit[index - 1] = 1
        if index <= len(gh[0]):
            gh[0][index - 1] = 1 - gh[0][index - 1]
        else:
            str = gh[1]
            mutate_index = index - len(gh[0]) - 1
            temp = str[0:mutate_index] + ('0' if gh[1][mutate_index] == '1' else '1') + str[mutate_index + 1:]
            gh[1] = temp
        change_value -= 1

    if int(gh[1], 2) == 0:
        gh[1] = backup

    return gh


def algorithm(x_train, x_test, y_train, y_test):
    # a = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1], '101000', [1, 0]]
    # b = [[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], '111001', [1, 0]]
    # a = update_position(a, 1)
    # print(a)
    # exit()
    N = 3
    grasshoppers = give_N_random_solutions(N, len(x_train[0]))
    # grasshoppers = [
    #    [[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1], '101000', [1, 0]],
    #    [[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], '111001', [1, 0]]]
    print(N, "random Solutions generated")
    best_sol = [0, -1]  # accuracy, index
    for i in range(len(grasshoppers)):
        no_of_hidden_neurons = int(grasshoppers[i][1], 2)
        print("Running SA on", i, "solution:")
        print(grasshoppers[i])
        # ypre = SA.model(updated_X(x_train, grasshoppers[i][0]), updated_X(x_test, grasshoppers[i][0]), y_train,
        #                 y_test, no_of_hidden_neurons=no_of_hidden_neurons)

        # ypre = bp.model(updated_X(x_train, grasshoppers[i][0]), updated_X(x_test, grasshoppers[i][0]), y_train, y_test,
        #               no_of_hidden_neurons=no_of_hidden_neurons)

        accuracy = accuracy_score(y_test, ypre)
        print(accuracy, "\n")
        if accuracy > best_sol[0]:
            best_sol[0] = accuracy
            best_sol[1] = i

    print("Initial Best", best_sol)

    max_it = 15
    cMax = 1
    cMin = 0.00004
    l = 2
    ub = len(grasshoppers[0][0]) + len(grasshoppers[0][1])
    lb = 0
    while l < max_it:
        c = cMax - l * ((cMax - cMin) / max_it)
        print(l, "iteration, c", c, "----------------------------------------------------------------------------->")
        for i in range(len(grasshoppers)):
            j = 0
            Xi = 0
            while j < len(grasshoppers):
                if j != i:
                    # Normalize
                    grasshoppers[i] = normalize_distance(grasshoppers[i], grasshoppers[j])
                    # grasshoppers[j] = normalize_distance(grasshoppers[j], grasshoppers[i])

                    dist = distance(grasshoppers[j], grasshoppers[i])
                    Xi += c * ((ub - lb) / 2) * (0.5 * np.exp(-dist / 1.5) - np.exp(-dist))
                j += 1

            # print(Xi)
            Xi *= c
            # print(Xi)
            Td = distance(grasshoppers[i], grasshoppers[best_sol[1]])
            # print("Dist", Td)
            Xi += Td
            # print("Final Xi", Xi)
            change_value = np.ceil(Xi)
            grasshoppers[i] = update_position(grasshoppers[i], abs(change_value))

            no_of_hidden_neurons = int(grasshoppers[i][1], 2)

            # ypre = SA.model(updated_X(x_train, grasshoppers[i][0]), updated_X(x_test, grasshoppers[i][0]), y_train,
            #                y_test, no_of_hidden_neurons=no_of_hidden_neurons)

            print("current GOA grasshopper----------------------------------------->>>", grasshoppers[i])

            # ypre = bp.model(updated_X(x_train, grasshoppers[i][0]), updated_X(x_test, grasshoppers[i][0]), y_train,
            #                 y_test, no_of_hidden_neurons=no_of_hidden_neurons)

            accuracy = accuracy_score(y_test, ypre)
            # print(accuracy, "\n")
            if accuracy > best_sol[0]:
                best_sol[0] = accuracy
                best_sol[1] = i
                print("\n\nBEST UPDATED: ", grasshoppers[i])

            print("----------------------------------------------------------------------------->")
            print("Best accuracy so far", best_sol)
        l += 1


if __name__ == '__main__':
    for i in range(5):
        v = give_a_random_solution(10)
        print(v)
