import pandas as pd
import numpy as np
from sklearn import datasets


def handle(s):
    return list.index(s)


# boston = datasets.load_breast_cancer()
# x = boston.data
# y = boston.target
# y = np.reshape(y, newshape=(len(y), 1))
# print(x.shape)
# print(y.shape)
# data = np.concatenate((x, y), axis=1)
# print(data.shape)
# print(data[0])
# np.savetxt("breast_cancer.csv", data, delimiter=',')

a = np.genfromtxt('datasets/BankNote_Authentication.csv', delimiter=',', skip_header=True)
print(a.shape)
# b = np.genfromtxt('Hill_gValley_with_noise_Testing.csv', delimiter=',', skip_header=True)
# print(b.shape)
# c = np.concatenate((a, b), axis=0)
# print(c.shape)
np.savetxt("datasets/BankNote_Authentication_cln.csv", a, delimiter=',')

exit()
df = pd.read_csv('kr-vs-kp.csv', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Y']
unq = np.unique(np.array(df['buying']))
list = unq.tolist()
df['buying'] = df.buying.apply(handle)

unq = np.unique(np.array(df['maint']))
list = unq.tolist()
df['maint'] = df.maint.apply(handle)

unq = np.unique(np.array(df['doors']))
list = unq.tolist()
df['doors'] = df.doors.apply(handle)

unq = np.unique(np.array(df['persons']))
list = unq.tolist()
df['persons'] = df.persons.apply(handle)

unq = np.unique(np.array(df['lug_boot']))
list = unq.tolist()
df['lug_boot'] = df.lug_boot.apply(handle)

unq = np.unique(np.array(df['safety']))
list = unq.tolist()
df['safety'] = df.safety.apply(handle)

unq = np.unique(np.array(df['Y']))
list = unq.tolist()
df['Y'] = df.Y.apply(handle)
print(df)
data = df.values
np.savetxt("car_evaluation1.csv", data, delimiter=',')
