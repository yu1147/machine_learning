import numpy as np
import matplotlib.pyplot as plt

X = np.array([
            [-1, 6],
            [1, 5],
            [1, 7],
            [3, 3],
            [5, 4],
            [2, 0]])
y = np.array([1, 1, 1, -1, -1, -1])
u = []
print(X[1])
u.append(np.sum(X.dot(X[1])))
print(u)


