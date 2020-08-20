import numpy as np
import matplotlib.pyplot as plt
from get2d import GetProjectivePoint_2D

data_file = open('D://mechain_learning//第三章//3.5//watermelon3.csv')
dataset = np.loadtxt(data_file, delimiter=",")

X = dataset[:, 1:3]
y = dataset[:, 3]
'''
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
ax1.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.title('watermelon')
ax1.legend(loc = 'upper right')
# plt.show()
'''
u = []
for i in range(2):
    u.append(np.mean(X[y == i], axis=0))

# print(u)
print(X[1])
m, n = np.shape(X)
# print(m, n)

'''
类内散度
'''
Sw = np.zeros((n, n))
for i in range(m):
    x_tmp = X[i].reshape(n, 1)
    if y[i] == 0:
        u_tmp = u[0].reshape(n, 1)
    if y[i] == 1:
        u_tmp = u[1].reshape(n, 1)
    Sw += np.dot(x_tmp - u_tmp, (x_tmp - u_tmp).T)

Sw = np.mat(Sw)
U, sigma, V = np.linalg.svd(Sw)

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))
print(w)
print(w[1, 0], w[0, 0])
#ax2 = fig.add_subplot(1, 2, 2)
plt.xlim(-0.2, 1)
plt.ylim(-0.5, 0.7)
p0_x0 = -X[:, 0].max()

p0_x1 = (w[1, 0]/w[0, 0])*p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = (w[1, 0]/w[0, 0])*p1_x0
print(p0_x0, p0_x1, p1_x0, p1_x1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1]) # 必经过（0,0）

plt.title('LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.legend(loc='upper right')

for i in range(m):
    x_p = GetProjectivePoint_2D([X[i, 0], X[i, 1]], [w[1, 0]/w[0, 0], 0])
    if y[i] == 0:
        plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
    if y[i] == 1:
        plt.plot(x_p[0], x_p[1], 'go', markersize=5)
    plt.plot([])
    plt.plot([X[i, 0], x_p[0]], [X[i, 1], x_p[1]], 'c--', linewidth=0.3)
# plt.scatter(0, 0, marker='o', color='yellow', s=50)
plt.show()

