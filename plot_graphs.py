import numpy as np
from matplotlib import pyplot
import math


# def feature_vector_penalty(feature_vector):
#     sel = np.count_nonzero(np.array(feature_vector))
#     total = len(feature_vector)
#     y = np.square(sel - total / 2)
#     print(y)
#     y_max = np.square(0 - total / 2)
#     print(y_max)
#     y_normalized = (y - 0) / (y_max - 0)
#     return y_normalized
#
#
# print(feature_vector_penalty([1, 1, 1, 1, 0, 0]))
#
# total_feature = 6
# sel = np.array([i for i in range(0, total_feature + 1)])
# print(sel)
#
# y = np.square(sel - total_feature / 2)
# print(y)
#
# y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
# print(y_normalized)

def fun(change):
    y = [0 for i in range(0, 101)]
    j = 0
    fill = -1
    for i in range(len(y)):
        if j < len(change) and change[j][0] == i:
            fill = change[j][1]
            j += 1
        y[i] = fill
    return y


def plot_all(x, y):
    import pylab
    labels = ['GOA+PSO', 'GOA+PSO(W/O Penalty)', 'GOA+BP']
    colors = ['r', 'g', 'b']

    # loop over data, labels and colors
    for i in range(len(y)):
        pylab.plot(x, y[i], color=colors[i], label=labels[i])

    pylab.xlabel("iteration no. ->")
    pylab.ylabel("GOA fitness ->")
    pylab.legend(loc=1)
    pylab.title("Thyroid")
    pylab.show()


x = [i for i in range(0, 101)]

with_penalty = [(0, 0.1342), (1, 0.1263), (2, 0.12607), (5, 0.1093), (6, 0.0940), (10, 0.09286), (12, 0.08940),
                (17, 0.08226), (29, 0.07681), (31, 0.0688)]
y1 = fun(with_penalty)

without_penalty = [(0, 0.06009), (2, 0.0554), (11, 0.04132)]
y2 = fun(without_penalty)

bp = [(0, 0.127), (5, 0.102), (8, 0.10), (11, 0.099), (26, 0.09), (51, 0.0888)]
y3 = fun(bp)

y = [y1, y2, y3]
plot_all(x, y)

# exit()
# pyplot.xlabel("iteration no. ->")
# pyplot.ylabel("GOA fitness ->")
# pyplot.plot(x, y1)
# pyplot.plot(x, y2)
# pyplot.plot(x, y3)
# [x] = pyplot.plot(y1, y2, y3)
# pyplot.legend([y1, y2, y3], ["y1", "y2", "y3"], loc=1)
# pyplot.show()
