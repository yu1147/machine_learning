import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
from pydotplus import graphviz

data = pd.read_csv("D:/mechain_learning/第四章/4_3sk/laichuan.csv",header=None)

data.columns = ['Season', 'After 8', 'Wind', 'Lay bed']

vec = DictVectorizer(sparse=False)

x = data[['Season', 'After 8', 'Wind']]
# x = vec.fit_transform(feature.to_dict(orient='record'))
y = data['Lay bed']
#  print('show feature\n', feature)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x = vec.fit_transform(x.to_dict(orient='record'))

# x_test = vec.transform(x_test.to_dict(orient='record'))
# y_train = vec.transform(y_train.to_dict(orient='record'))
# y_test = vec.transform(y_test.to_dict(orient='record'))

clt = tree.DecisionTreeClassifier(criterion="gini")
clt.fit(x, y)
# y_pre = clt.predict(x_test)

with open("out.dot", 'w') as f:
    f = tree.export_graphviz(clt, out_file=f, feature_names=vec.get_feature_names())


