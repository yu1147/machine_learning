from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as lrs
import pandas as pa
import numpy as np

iris_data = np.array(pa.read_csv('D:/mechain_learning/第三章/3.4/data/iris.data'))
iris_data = iris_data[iris_data[:, 4] != "Iris-virginica"]

mapper = {"Iris-setosa":0,"Iris-versicolor":1}

for i in iris_data:
    i[4]=mapper[i[4]]

#预处理wine
wine_data = np.array(pa.read_csv('D:/mechain_learning/第三章/3.4/data/wine.data'))
print(wine_data.shape[0])
wine_data = wine_data[:129]

kf1 = KFold(10, True)

scores11 = []

for train, test in kf1.split(iris_data):
    x_train = iris_data[train]
    x_test = iris_data[test]
    lr = lrs()
    lr.fit(x_train[:, 0:4], x_train[:, 4].astype('int'))
    scores11.append(lr.score(x_test[:, 0:4], x_test[:, 4].astype('int')))

kf2 = KFold(10,True)
scores12 = []

for train, test in kf2.split(wine_data):
    x_train = wine_data[train]
    x_test = wine_data[test]
    lr = lrs()
    lr.fit(x_train[:, 1:], x_train[:, 0].astype('int'))
    scores12.append(lr.score(x_test[:, 1:], x_test[:, 0].astype('int')))

kf3 = KFold(iris_data.shape[0], True)
scores21 = []

for train, test in kf3.split(iris_data):
    x_train = iris_data[train]
    x_test = iris_data[test]
    lr = lrs()
    lr.fit(x_train[:, 0:4], x_train[:, 4].astype('int'))
    scores21.append(lr.score(x_test[:, 0:4], x_test[:, 4].astype('int')))

kf4 = KFold(wine_data.shape[0],True)
scores22 = []
for train, test in kf4.split(wine_data):
    x_train = wine_data[train]
    x_test = wine_data[test]
    lr = lrs()
    lr.fit(x_train[:, 1:], x_train[:, 0].astype('int'))
    scores22.append(lr.score(x_test[:, 1:], x_test[:, 0].astype('int')))

print("iris:"+str(1-np.mean(scores11)), str(1-np.mean(scores21)), "wine:"+str(1-np.mean(scores12)),str(1-np.mean(scores22)))