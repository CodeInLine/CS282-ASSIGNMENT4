import csv
import numpy as np
import matplotlib.pyplot as plt

data_path = './iris.data'
LABEL2NUM = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
#np.random.seed(123)
iteration = 101

dataset = []
with open(data_path, encoding='utf-8') as f:
    for data in csv.DictReader(f):
        new_data = []
        for v in data.values():
            if v in LABEL2NUM:
                new_data.append(LABEL2NUM[v])
            else:
                new_data.append(float(v))
        dataset.append(new_data)
dataset = np.asarray(dataset)
x = dataset[:, :-1]

def label(x, m):
    x_mat = np.tile(x[:-1], (m.shape[0], 1))
    m = m[:, :-1]
    diff = x_mat - m
    distance_mat = np.matmul(diff, diff.T)
    distance = np.zeros(m.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = distance_mat[i][i]
    idx = np.argmin(distance)
    x[-1] = idx
    return x

def SSE(x, m):
    result = 0.0
    for i in range(m.shape[0]):
        mask = x[:, -1] == i
        for j in range(x[mask].shape[0]):
            diff = x[mask][j] - m[i]
            distance = np.matmul(diff, diff.T)
            result += np.sum(distance)
    return result

def kmeans(x, k):

    # initialize k centers
    idx = np.random.choice(x.shape[0], k, replace=False)
    m = x[idx, :]
    m = np.concatenate([m, np.asarray(range(k)).reshape(m.shape[0], 1)], axis=-1)
    for i in range(k):
        diff = x[0] - m[i][:-1]
    
    x = np.concatenate([x, np.zeros([x.shape[0], 1])], axis=-1)
    
    for _ in range(iteration):
        for i in range(x.shape[0]):
            x[i] = label(x[i], m)
        
        for i in range(k):
            mask = x[:, -1] == i
            m[i, :-1] = np.mean(x[mask][:, :-1])
        
    return SSE(x, m)

result = []
for k in range(1, 9):
    r = kmeans(x, k)
    print(f'k= {k}: {r}')
    result.append(r)

plt.cla()
plt.plot(range(1, 9), result, color='blue', label='SSE value')
plt.xlabel('k')
plt.ylabel('SSE')
plt.savefig('q1.jpg')