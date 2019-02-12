import pandas as pd
import numpy as np


def handle(s):
    return list.index(s)


df = pd.read_csv('car_evaluation.csv', header=None)
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
