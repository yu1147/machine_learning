import numpy as np

import matplotlib.pyplot as plt

dataset = np.loadtxt('D:/mechain_learning/第三章/3.3/watermelon3.csv', delimiter=",", encoding='gb18030')

x = dataset[:, 1:3]
y = dataset[:, 3]

f1 = plt.figure(1)

plt.title('watermelon_3a')

plt.xlabel('density')
plt.ylabel('ratio_sugar')

plt.scatter(x[y == 0, 0], x[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', color='g', s=100, label='good')
plt.legend(loc='upper right')
plt.show()
