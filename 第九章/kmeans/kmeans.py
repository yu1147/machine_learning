import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
vote = pd.read_csv("114_congress.csv")
dist = euclidean_distances(vote.iloc[0, 3:].values.reshape(1, -1), vote.iloc[1, 3:].values.reshape(1, -1))
# print(dist)
kmeans = KMeans(n_clusters=2, random_state=1)
senator = kmeans.fit_transform(vote.iloc[:, 3:].values)
# print(senator)
label = kmeans.labels_
print(pd.crosstab(label, vote["party"]))
democratic = vote[(label == 0) & (vote["party"] == "D")]
# print(democratic)
x = senator[:, 0]
y = senator[:, 1]
extremist = (senator ** 3).sum(axis=1)
vote["extremist"] = extremist
vote.sort_values("extremist", inplace=True, ascending=False)
print(vote.head(10))

plt.scatter(x, y, c=label)
plt.show()
