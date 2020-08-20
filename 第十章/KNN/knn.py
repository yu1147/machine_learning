import numpy as np
import operator


def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataset, labels, k):
    datasetSize = dataset.shape[0]
    diffMat = np.tile(intX, (datasetSize, 1)) - dataset
    sqdiffMat = diffMat ** 2
    sqDistance = sqdiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sorteddistIndicies = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorteddistIndicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


if __name__ == '__main__':
    group, labels = createDataset()
    test = classify0([0.9, 1.2], group, labels, 3)
    print(test)
