import numpy as np
import matplotlib.pyplot as plt

iris_data = open('D:/mechain_learning/第三章/3.5/watermelon3.csv')
dataset = np.loadtxt(iris_data, delimiter=",")

x = dataset[:, 1:3]
y = dataset[:, 3]

#画图
# f1 = plt.figure(1)
# plt.title('watermelon3a')
# plt.xlabel('密集度')
# plt.ylabel('甜度')
# plt.scatter(x[y == 0, 0], x[y == 0, 1], marker='o', color='b', s=100, label='bad')
# plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', color='r', s=100, label='good')
# plt.legend(loc='upper right')
# #plt.show()

#lda模型建造
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt

#生成训练测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5, random_state=0)

#训练模型
lda_model = LinearDiscriminantAnalysis().fit(x_train,y_train)

#模型验证
y_pred = lda_model.predict(x_test)

#模型评估，生成混淆矩阵
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

#画出决策边界
f2 = plt.figure(2)
h=0.01
x0_min, x0_max = x[:, 0].min()-0.1, x[:, 0].max()+0.1
x1_min, x1_max = x[:, 1].min()-0.1, x[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
z = lda_model.predict(np.c_[x0.ravel(), x1.ravel()])
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z)

plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(x[y == 0, 0], x[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', color='g', s=100, label='good')
plt.show()

