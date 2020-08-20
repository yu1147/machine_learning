import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris_data = pd.read_csv('iris.data')
iris_data.columns = ['sepal_length(cm)', 'sepal_width(cm)', 'petal_length(cm)', 'petal_width(cm)', 'class']
sns.pairplot(iris_data.dropna(), hue='class')

# plt.show()

X = iris_data[['sepal_length(cm)', 'sepal_width(cm)', 'petal_length(cm)', 'petal_width(cm)']].values
y = iris_data['class'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dsc = dtc.score(x_test, y_test)
print(dsc)
