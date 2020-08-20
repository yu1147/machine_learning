import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

data1 = "D:/mechain_learning/第六章/data/watermelon3a.csv"

df = pd.read_csv(data1, header=None, encoding='GBK')
df.columns = ['id', 'density', 'sugar_content', 'label']
df.set_index(['id'])

X = df[['density', 'sugar_content']].values
y = df['label'].values

for fig_num, kernel in enumerate(('linear', 'rbf')):
    svc = svm.SVC(C=1000, kernel=kernel)
    svc.fit(X, y)
    sv = svc.support_vectors_

    plt.figure(fig_num)
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired, zorder=10)
    plt.scatter(sv[:, 0], sv[:, 1], edgecolors='k', facecolors='none', s=80, linewidths=2, zorder=10)
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.axis('tight')

plt.show()
