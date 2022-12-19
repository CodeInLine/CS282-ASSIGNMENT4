import csv
import numpy as np
from scipy import sparse
import osqp
import matplotlib.pyplot as plt

data_path = './iris.data'
LABEL2NUM = {'Iris-setosa': -1, 'Iris-versicolor': 1, 'Iris-virginica': 0}
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
dataset = dataset[dataset[:, -1] != 0][:, 2:]
point = dataset[:, :-1]
label = dataset[:, -1:]

q = np.zeros(3)
P = sparse.csc_matrix(np.eye(3))
P[0][0] = 0
l = np.ones(point.shape[0])
A = np.concatenate([l.reshape(l.shape[0], 1), point], axis=-1)
for i in range(A.shape[0]):
    A[i] = A[i] * label[i]
A = sparse.csc_matrix(A)

solver = osqp.OSQP()
solver.setup(P, q, A, l, alpha=1.0)
res = solver.solve()

x = np.concatenate([np.ones(shape=[point.shape[0], 1]), point], axis=1)
w = np.asarray(res.x).reshape([len(res.x), 1])
result = x @ w

plt.cla()
mask = np.sign(result.flatten())==1
plt.scatter(list(point[mask, 0]), list(point[mask, 1]), color='blue', label='+1')
mask = np.sign(result.flatten())==-1
plt.scatter(list(point[mask, 0]), list(point[mask, 1]), color='red', label=' -1')
plt.legend()
plt.savefig('q2.jpg')
