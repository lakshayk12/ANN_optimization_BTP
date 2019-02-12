import copy
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score
import PSO

minimum_no_of_hidden_neuron = 2


def give_a_random_solution(no_of_features):
    global max_number_of_hidden_neuron
    max_number_of_hidden_neuron = 2 * (no_of_features + 1)
    vector = [[0 for i in range(no_of_features)]]
    n = math.ceil(math.log(max_number_of_hidden_neuron, 2))

    no_of_selected_features = random.randint(np.ceil(no_of_features / 2), no_of_features)
    count = 0
    print("no_of_selected_features", no_of_selected_features)
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
    for i in range(len(gh1[2])):
        dist += (int(gh1[2][i]) ^ int(gh2[2][i]))
    for i in range(len(gh1[3])):
        dist += (gh1[3][i] ^ gh2[3][i])
    for i in range(len(gh1[4])):
        dist += (gh1[4][i] ^ gh2[4][i])
    return dist


def normalize_distance(gh, bgh):  # gh = curr_search_agent, bgh = best_search_agent
    normalization_factor = random.randint(1, 4)
    # print("NF ", normalization_factor)
    dist = distance(gh, bgh)
    n = len(gh[0]) + len(gh[1]) + len(gh[2]) + len(gh[3]) + len(gh[4])

    visit = [0 for i in range(n)]

    save1 = gh[1]
    save2 = gh[2]

    while dist > normalization_factor:
        index = random.randint(0, n - 1)
        # print("INDEXXXXX", index)
        if visit[index] == 1:
            continue

        visit[index] = 1

        if index < len(gh[0]):
            if gh[0][index] != bgh[0][index]:
                gh[0][index] = bgh[0][index]
                dist -= 1
        elif index < len(gh[0]) + len(gh[1]):
            if gh[1][index - len(gh[0])] != bgh[1][index - len(gh[0])]:
                str = gh[1]
                mutate_index = index - len(gh[0])
                temp = str[0:mutate_index] + bgh[1][mutate_index] + str[mutate_index + 1:]
                gh[1] = temp
                dist -= 1
        elif index < len(gh[0]) + len(gh[1]) + len(gh[2]):
            if gh[2][index - len(gh[0]) - len(gh[1])] != bgh[2][index - len(gh[0]) - len(gh[1])]:
                str = gh[2]
                mutate_index = index - len(gh[0]) - len(gh[1])
                temp = str[0:mutate_index] + bgh[2][mutate_index] + str[mutate_index + 1:]
                gh[2] = temp
                dist -= 1
        elif index < len(gh[0]) + len(gh[1]) + len(gh[2]) + len(gh[3]):
            if gh[3][index - len(gh[0]) - len(gh[1]) - len(gh[2])] != bgh[3][
                index - len(gh[0]) - len(gh[1]) - len(gh[2])]:
                gh[3][index - len(gh[0]) - len(gh[1]) - len(gh[2])] = bgh[3][
                    index - len(gh[0]) - len(gh[1]) - len(gh[2])]
                dist -= 1
        elif gh[4][index - len(gh[0]) - len(gh[1]) - len(gh[2]) - len(gh[3])] != bgh[4][
            index - len(gh[0]) - len(gh[1]) - len(gh[2]) - len(gh[3])]:
            gh[4][index - len(gh[0]) - len(gh[1]) - len(gh[2]) - len(gh[3])] = bgh[3][
                index - len(gh[0]) - len(gh[1]) - len(gh[2]) - len(gh[3])]
            dist -= 1

    if int(gh[1], 2) < minimum_no_of_hidden_neuron:
        gh[1] = save1
    if int(gh[2], 2) < minimum_no_of_hidden_neuron:
        gh[2] = save2

    print("NORMALIZED . . .")
    for i in gh[0]:
        if i == 1:
            return gh
    gh[0][random.randint(0, len(gh[0]) - 1)] = 1
    return gh


def update_position(gh, change_value):
    if change_value == 0:
        return gh
    n = len(gh[0]) + len(gh[1]) + len(gh[2]) + len(gh[3]) + len(gh[4])
    visit = [0 for i in range(n)]
    backup_HL1 = gh[1]
    backup_HL2 = gh[2]

    while change_value > 0:
        index = random.randint(0, n - 1)
        # print("INDEX------", index)
        if visit[index] == 1:
            continue
        visit[index] = 1
        if index < len(gh[0]):
            gh[0][index] = 1 - gh[0][index]
        elif index < len(gh[0]) + len(gh[1]):
            str = gh[1]
            mutate_index = index - len(gh[0])
            temp = str[0:mutate_index] + ('0' if gh[1][mutate_index] == '1' else '1') + str[mutate_index + 1:]
            gh[1] = temp
        elif index < len(gh[0]) + len(gh[1]) + len(gh[2]):
            str = gh[2]
            mutate_index = index - len(gh[0]) - len(gh[1])
            temp = str[0:mutate_index] + ('0' if gh[2][mutate_index] == '1' else '1') + str[mutate_index + 1:]
            gh[2] = temp
        elif index < len(gh[0]) + len(gh[1]) + len(gh[2]) + len(gh[3]):
            mutate_index = index - len(gh[0]) - len(gh[1]) - len(gh[2])
            gh[3][mutate_index] = 1 - gh[3][mutate_index]
        else:
            mutate_index = index - len(gh[0]) - len(gh[1]) - len(gh[2]) - len(gh[3])
            gh[4][mutate_index] = 1 - gh[4][mutate_index]
        change_value -= 1

    if int(gh[1], 2) < minimum_no_of_hidden_neuron:
        gh[1] = backup_HL1
    if int(gh[2], 2) < minimum_no_of_hidden_neuron:
        gh[2] = backup_HL2

    # check if feature_vector is all zero?
    for i in gh[0]:
        if i == 1:
            return gh
    gh[0][random.randint(0, len(gh[0]) - 1)] = 1
    return gh


def guess_weight():
    pass


def algorithm(x_train, x_test, y_train, y_test):
    N = 3
    grasshoppers = give_N_random_solutions(N, len(x_train[0]))
    # A' = [[list([0, 1]) '101' '110' array([0, 0]) array([0, 0])]
    print(N, "random Solutions generated")
    print(grasshoppers)
    # exit()
    best_sol = [0, -1, -1]  # accuracy, grasshopper, corresponding_weights
    for i in range(len(grasshoppers)):
        no_of_hidden_neurons1 = int(grasshoppers[i][1], 2)
        no_of_hidden_neurons2 = int(grasshoppers[i][2], 2)
        print("Running PSO on", i, "solution:")
        print(grasshoppers[i])
        updated_x_train = updated_X(x_train, grasshoppers[i][0])
        updated_x_test = updated_X(x_test, grasshoppers[i][0])
        accuracy, corresponding_weights = PSO.model(updated_x_train, updated_x_test, y_train, y_test,
                                                    no_of_input_neurons=len(updated_x_train[0]),
                                                    no_of_hidden_neurons1=no_of_hidden_neurons1,
                                                    no_of_hidden_neurons2=no_of_hidden_neurons2, no_of_output_neurons=2)

        print(accuracy, "\n")
        if accuracy > best_sol[0]:
            best_sol[0] = accuracy
            best_sol[1] = grasshoppers[i]
            best_sol[2] = corresponding_weights

    print("Initial Best", best_sol, "\n\n")
    # exit()
    max_it = 15
    cMax = 1
    cMin = 0.00004
    l = 2
    ub = len(grasshoppers[0][0]) + len(grasshoppers[0][1])  ##................?????
    lb = 0
    while l < max_it:
        c = cMax - l * ((cMax - cMin) / max_it)
        print(l, "iteration, c", c, "----------------------------------------------------------------------------->")
        for i in range(len(grasshoppers)):
            j = 0
            Xi = 0

            # for every ith grasshopper's position is updated according to the position of every jth
            while j < len(grasshoppers):
                if j != i:
                    # Normalize
                    grasshoppers[i] = normalize_distance(grasshoppers[i], grasshoppers[j])
                    # print("#######")
                    # exit()
                    # print(grasshoppers[i])
                    # grasshoppers[j] = normalize_distance(grasshoppers[j], grasshoppers[i])

                    dist = distance(grasshoppers[j], grasshoppers[i])
                    Xi += c * ((ub - lb) / 2) * (0.5 * np.exp(-dist / 1.5) - np.exp(-dist))
                j += 1
                # print("heere----->>")

            # print(Xi)
            Xi *= c
            # print(Xi)
            Td = distance(grasshoppers[i], best_sol[1])
            # print("Dist", Td)
            Xi += Td
            # print("Final Xi", Xi)
            change_value = np.ceil(Xi)
            grasshoppers[i] = update_position(grasshoppers[i], abs(change_value))

            print("current GOA grasshopper----------------------------------------->>>", grasshoppers[i])

            no_of_hidden_neurons1 = int(grasshoppers[i][1], 2)
            no_of_hidden_neurons2 = int(grasshoppers[i][2], 2)
            updated_x_train = updated_X(x_train, grasshoppers[i][0])
            updated_x_test = updated_X(x_test, grasshoppers[i][0])
            accuracy, corresponding_weights = PSO.model(updated_x_train, updated_x_test, y_train, y_test,
                                                        no_of_input_neurons=len(updated_x_train[0]),
                                                        no_of_hidden_neurons1=no_of_hidden_neurons1,
                                                        no_of_hidden_neurons2=no_of_hidden_neurons2,
                                                        no_of_output_neurons=2)

            if accuracy > best_sol[0]:
                best_sol[0] = accuracy
                best_sol[1] = grasshoppers[i]
                best_sol[2] = corresponding_weights
                print("\n\nBEST UPDATED: ", best_sol, "\n\n")

            print("----------------------------------------------------------------------------->")
            print("Best accuracy so far", best_sol)
        l += 1

    print("\n\nOptimal Solution:\n", best_sol)
    print("Final verification . . .")
    verify(x_test, y_test, best_sol[1], best_sol[2])


def verify(x_test, y_test, architecture, weights):
    output, error = PSO.generate_output_and_error(updated_X(x_test, architecture[0]), y_test, weights)
    print(output, error)


if __name__ == '__main__':
    v = give_N_random_solutions(20, 2)
    print(v)
    # gh = v[0]
    # nh = update_position(copy.deepcopy(gh), 6)
    # print(distance(gh, nh))
    # print(gh)
    # print(nh)
    # gh = normalize_distance(gh, nh)
    # print(distance(gh, nh))
