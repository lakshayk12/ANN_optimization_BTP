import copy
import math
import random
import numpy as np
import PSO
import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

minimum_no_of_hidden_neuron = 20


def give_a_random_solution(no_of_features):
    global max_number_of_hidden_neuron
    max_number_of_hidden_neuron = 2 * (no_of_features + 1)

    if settings.max_no_of_neurons is None:
        settings.max_no_of_neurons = max_number_of_hidden_neuron * 2

    vector = [[0 for i in range(no_of_features)]]
    n = math.ceil(math.log(max_number_of_hidden_neuron, 2))
    if settings.minimum_no_of_present_features is None:
        settings.minimum_no_of_present_features = np.ceil(no_of_features / 2)
    no_of_selected_features = random.randint(settings.minimum_no_of_present_features, no_of_features)
    count = 0
    # print("no_of_selected_features", no_of_selected_features)
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


def reset_grasshopper(gh):
    Count = 0
    for i in gh[0]:
        if i == 1:
            Count += 1
    if Count >= settings.minimum_no_of_present_features:
        return gh
    Count = 0
    l = []
    for i in range(len(gh[0])):
        if gh[0][i] == 0:
            l.append(i)
        else:
            Count += 1
    while Count < settings.minimum_no_of_present_features:
        index = random.randint(0, len(l) - 1)
        if gh[0][l[index]] == 0:
            gh[0][l[index]] = 1
            Count += 1
    return gh


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

    # ensuring minimum no. of features should be selected
    gh = reset_grasshopper(gh)
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

    # ensuring minimum no. of features should be selected
    gh = reset_grasshopper(gh)
    return gh


def make_similar_matrix(old_dim, new_dim, old_matrix, gh, previous_gh):
    new_mat = np.zeros((new_dim[0], old_dim[1]))
    # print("prev gh:", previous_gh)
    # print("new gh:", gh)
    # 1 0 1 1
    # 1 1 1 1
    j = 0
    k = 0
    if new_dim[0] >= old_dim[0]:
        for i in range(len(gh[0])):
            if gh[0][i] == 1 and gh[0][i] == previous_gh[0][i]:
                new_mat[j] = old_matrix[k, 0:old_dim[1]]
                j += 1
                k += 1
            elif gh[0][i] == 1 and previous_gh[0][i] == 0:
                if j == 0:
                    new_mat[j] = np.random.randn(1, old_dim[1])
                else:
                    new_mat[j] = np.mean(new_mat[0:j, :], axis=0, keepdims=True)
                j += 1

    j = k = 0  # k old mat_index & j new mat
    if new_dim[0] < old_dim[0]:
        for i in range(len(gh[0])):
            if gh[0][i] == 1 and gh[0][i] == previous_gh[0][i]:
                new_mat[j] = old_matrix[k, 0:old_dim[1]]
                j += 1
                k += 1
            elif gh[0][i] == 0 and previous_gh[0][i] == 1:
                k += 1
            elif gh[0][i] == 1 and previous_gh[0][i] == 0:
                if j == 0:
                    new_mat[j] = np.random.randn(1, old_dim[1])
                else:
                    new_mat[j] = np.mean(new_mat[0:j, :], axis=0, keepdims=True)
                j += 1

    # ADD SUITABLE COLUMNS
    # print("new before cols:\n", new_mat)
    # exit()
    if new_dim[1] < old_dim[1]:
        new_mat = new_mat[:, 0:new_dim[1]]
    else:
        no_col_to_concat = new_dim[1] - old_dim[1]
        for i in range(no_col_to_concat):
            new_mat = np.concatenate((new_mat, np.mean(new_mat[:, 0:new_dim[1]], axis=1, keepdims=True)), axis=1)

    # print("NEW:\n", new_mat)
    return new_mat


def guess_weight(gh, previous_gh, old_weights):  # here, gh is new grasshopper
    # print("IN GUESS WEIGHTS. . .")
    old_wh1_dim = old_weights[0].shape
    # old_bh1_dim = old_weights[1].shape
    # old_wh2_dim = old_weights[2].shape
    # old_bh2_dim = old_weights[3].shape
    # old_wo_dim = old_weights[4].shape
    # old_bo_dim = old_weights[5].shape
    # print("Old:", old_wh1_dim, old_bh1_dim, old_wh2_dim, old_bh2_dim, old_wo_dim, old_bo_dim)

    # Targets weights
    new_no_of_inputs = 0
    for each in gh[0]:
        if each == 1:
            new_no_of_inputs += 1
    new_no_of_hl1 = int(gh[1], 2)
    new_no_of_hl2 = int(gh[2], 2)
    new_no_of_outputs = settings.no_of_classes

    new_wh1_dim = (new_no_of_inputs, new_no_of_hl1)
    new_bh1_dim = (1, new_no_of_hl1)
    # new_wh2_dim = (new_no_of_hl1, new_no_of_hl2)
    # new_bh2_dim = (1, new_no_of_hl2)
    # new_wo_dim = (new_no_of_hl2, new_no_of_outputs)
    # new_bo_dim = (1, new_no_of_outputs)
    # print("Required: ", new_wh1_dim, new_bh1_dim, new_wh2_dim, new_bh2_dim, new_wo_dim, new_bo_dim)

    # wh1
    new_wh1 = make_similar_matrix(old_wh1_dim, new_wh1_dim, old_weights[0], gh, previous_gh)
    new_bh1 = np.random.randn(new_bh1_dim[0], new_bh1_dim[1]) + np.mean(old_weights[1])
    new_wh2 = np.random.randn(new_no_of_hl1, new_no_of_hl2) + np.mean(old_weights[2])
    new_bh2 = np.random.randn(1, new_no_of_hl2) + np.mean(old_weights[3])
    new_wo = np.random.randn(new_no_of_hl2, new_no_of_outputs) + np.mean(old_weights[4])
    new_bo = np.random.randn(1, new_no_of_outputs) + np.mean(old_weights[5])

    return np.array([new_wh1, new_bh1, new_wh2, new_bh2, new_wo, new_bo])


def validation_split_of_dataset(x_train, y_train):
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2)
    return x_train, x_validate, y_train, y_validate


def validation_error(x_validate, y_validate, weights, tf1, tf2):
    output, curr_error = PSO.generate_output_and_error(x_validate, y_validate, weights, tf1, tf2)
    accuracy = accuracy_score(y_validate.argmax(axis=1), output.argmax(axis=1))
    mis_classification_error = 1 - accuracy
    return mis_classification_error


def feature_vector_penalty(feature_vector):
    sel = np.count_nonzero(np.array(feature_vector))
    total = len(feature_vector)
    y = np.square(sel - total / 2)
    y_max = np.square(0 - total / 2)
    y_normalized = (y - 0) / (y_max - 0)
    return y_normalized


def algorithm(x_train, y_train):
    if settings.validation_flag:
        # validation split
        x_train, x_validate, y_train, y_validate = validation_split_of_dataset(x_train, y_train)
        print(len(x_validate), "instances for validation.")

    print(len(x_train), "instances for training.")

    N = settings.goa_population_size
    grasshoppers = give_N_random_solutions(N, len(x_train[0]))
    # A' = [[list([0, 1]) '101' '110' array([0, 0]) array([0, 0])]
    print(N, "random Solutions generated")
    print(grasshoppers)
    # exit()
    previous_weights_of_ghs = []  # saves previous weights of ith grasshopper
    previous_ghs = []  # saves previous ith grasshopper
    best_sol = [math.inf, -1, -1]  # error, grasshopper, corresponding_weights
    first_run = True

    # Initial loop for finding initial bests solutions
    for i in range(len(grasshoppers)):
        no_of_hidden_neurons1 = int(grasshoppers[i][1], 2)
        no_of_hidden_neurons2 = int(grasshoppers[i][2], 2)
        tf1 = grasshoppers[i][3]
        tf2 = grasshoppers[i][4]
        print("\nRunning PSO on ", i + 1, " solution:")
        print(grasshoppers[i])
        updated_x_train = updated_X(x_train, grasshoppers[i][0])
        if settings.validation_flag:
            updated_x_validate = updated_X(x_validate, grasshoppers[i][0])
        CEE, corresponding_weights = PSO.model(updated_x_train, y_train,
                                               no_of_input_neurons=len(updated_x_train[0]),
                                               no_of_hidden_neurons1=no_of_hidden_neurons1,
                                               no_of_hidden_neurons2=no_of_hidden_neurons2,
                                               no_of_output_neurons=settings.no_of_classes,
                                               tf1=tf1, tf2=tf2)
        # Fitness of a Grasshopper
        architecture_penalty = (no_of_hidden_neurons1 + no_of_hidden_neurons2) / settings.max_no_of_neurons
        error = settings.CEE_weight * CEE
        if settings.validation_flag:
            error += settings.v_error_weight * validation_error(updated_x_validate, y_validate, corresponding_weights,
                                                                tf1,
                                                                tf2)
        error += settings.arch_penalty_weight * architecture_penalty
        error += settings.feature_penalty_weight * feature_vector_penalty(grasshoppers[i][0])
        print("| GOA fitness:", error)

        previous_ghs.append(copy.deepcopy(grasshoppers[i]))
        previous_weights_of_ghs.append(copy.deepcopy(corresponding_weights))
        if first_run or error < best_sol[0]:
            best_sol[0] = error
            best_sol[1] = copy.deepcopy(grasshoppers[i])
            best_sol[2] = copy.deepcopy(corresponding_weights)
            first_run = False

    print("\nInitial Best", best_sol[0:1], "\n\n")
    max_it = settings.goa_max_iteration
    cMax = 1
    cMin = 0.00004
    l = 1
    ub = len(grasshoppers[0][0]) + len(grasshoppers[0][1]) + len(grasshoppers[0][2]) + len(grasshoppers[0][3]) + len(
        grasshoppers[0][4])
    lb = 0
    while l < max_it:
        c = cMax - l * ((cMax - cMin) / max_it)
        print("\n\nITERATION -> ", l, " with value of C -> ", c,
              "----------------------------------------------------------------------------->")
        for i in range(len(grasshoppers)):
            j = 0
            Xi = 0

            # for every ith grasshopper's position is updated according to the position of every jth
            while j < len(grasshoppers):
                if j != i:
                    # Normalize
                    grasshoppers[i] = normalize_distance(grasshoppers[i], grasshoppers[j])
                    # grasshoppers[j] = normalize_distance(grasshoppers[j], grasshoppers[i])
                    dist = distance(grasshoppers[j], grasshoppers[i])
                    Xi += c * ((ub - lb) / 2) * (0.5 * np.exp(-dist / 1.5) - np.exp(-dist))
                j += 1

            Xi *= c
            Td = distance(grasshoppers[i], best_sol[1])
            Xi += Td
            change_value = np.ceil(Xi)
            grasshoppers[i] = update_position(grasshoppers[i], abs(change_value))

            print("\nCurrent GOA grasshopper------------------>>>", grasshoppers[i])

            no_of_hidden_neurons1 = int(grasshoppers[i][1], 2)
            no_of_hidden_neurons2 = int(grasshoppers[i][2], 2)
            tf1 = grasshoppers[i][3]
            tf2 = grasshoppers[i][4]
            updated_x_train = updated_X(x_train, grasshoppers[i][0])
            if settings.validation_flag:
                updated_x_validate = updated_X(x_validate, grasshoppers[i][0])
            # guess initial weights from previous weights
            # print("PReviouysfnefe\n", previous_weights_of_ghs[i])
            guessed_weights = guess_weight(grasshoppers[i], copy.deepcopy(previous_ghs[i]),
                                           copy.deepcopy(previous_weights_of_ghs[i]))
            # print("guesssss\n", guessed_weights)
            # exit()
            CEE, corresponding_weights = PSO.model(updated_x_train, y_train,
                                                   no_of_input_neurons=len(updated_x_train[0]),
                                                   no_of_hidden_neurons1=no_of_hidden_neurons1,
                                                   no_of_hidden_neurons2=no_of_hidden_neurons2,
                                                   no_of_output_neurons=settings.no_of_classes, tf1=tf1, tf2=tf2,
                                                   guessed_weights=guessed_weights)

            # Fitness of a Grasshopper
            architecture_penalty = (no_of_hidden_neurons1 + no_of_hidden_neurons2) / settings.max_no_of_neurons
            error = settings.CEE_weight * CEE
            if settings.validation_flag:
                error += settings.v_error_weight * validation_error(updated_x_validate, y_validate,
                                                                    corresponding_weights,
                                                                    tf1,
                                                                    tf2)
            error += settings.arch_penalty_weight * architecture_penalty
            error += settings.feature_penalty_weight * feature_vector_penalty(grasshoppers[i][0])
            print("| GOA fitness:", error)

            previous_ghs[i] = copy.deepcopy(grasshoppers[i])
            previous_weights_of_ghs[i] = copy.deepcopy(corresponding_weights)

            if error < best_sol[0]:
                best_sol[0] = error
                best_sol[1] = copy.deepcopy(grasshoppers[i])
                best_sol[2] = copy.deepcopy(corresponding_weights)
                print("\nBEST UPDATED: ", best_sol[0:1], "\n\n")

            print("\n-------------------------------------------------------------------------------")
            print("Best GOA Fitness so far", best_sol[0:1])
            print("-------------------------------------------------------------------------------\n")
        l += 1

    return best_sol


if __name__ == '__main__':
    v = [[1, 0, 0, 1, 0, 0]]
    settings.minimum_no_of_present_features = 2
    v = reset_grasshopper(v)
    print(v)
