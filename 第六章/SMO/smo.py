import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd

# calculate kernel value
def calcKernalValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    num_sample = matrix_x.shape[0]
    kernelValue = mat(zeros((num_sample, 1)))
    if kernelType == 'linear':
        kernelValue = matrix_x*sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(num_sample):
            diff = matrix_x[i, :]-sample_x
            kernelValue[i] = exp(diff*diff.T/(-2.0*sigma**2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue

# c alculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
    num_samples = train_x.shape[0]
    kernelMatrix = mat(zeros((num_samples, num_samples)))
    for i in range(num_samples):
        kernelMatrix[:, i] = calcKernalValue(train_x,train_x[i, :], kernelOption)
    return kernelMatrix

# define a struct just for storing variables and data
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.num_samples = dataSet.shape[0]
        self.alphas = mat(zeros((self.num_samples, 1)))
        self.b = 0
        self.errorCache = mat(zeros((self.num_samples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = mat(calcKernelMatrix(self.train_x, self.kernelOpt))
        print(train_x.shape)

# calculate the error for alpha k
def calcError(svm, alpha_k):
    # print("alpha:" + str(type(svm.train_y))+str(svm.train_y.shape))
    output_k = float(multiply(svm.alphas, svm.train_y).T*svm.kernelMat[:, alpha_k])+svm.b
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k

# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]

# select alpha j which has the biggest step
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]
    candidateAlphalist = nonzero(svm.errorCache[:, 0].A)[0]
    maxstep = 0
    alpha_j = 0
    error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphalist) > 1:
        for alpha_k in candidateAlphalist:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_i-error_k)>maxstep:
                maxstep = abs(error_i-error_k)
                alpha_j = alpha_k
                error_j = error_k

    # if came in this loop first time, we select alpha j randomly
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.num_samples))
        error_j = calcError(svm, alpha_j)
    return alpha_j, error_j

# the inner loop for optimizing alpha i and alpha j


'''
# check and pick up the alpha who violates the KKT condition
# satisfy KKT condition
# 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
# 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
# 3) yi*f(i) <= 1 and alpha == C (between the boundary)
# violate KKT condition
# because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
# 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
# 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
# 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
'''


def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
            (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        'step1:select alpha_j'
        alpha_j, error_j = selectAlpha_j(svm, alpha_i,error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        'step2:calculate the boundary L and H for alpha_j'
        if svm.train_y[alpha_i]!=svm.train_y[alpha_j]:
             L = max(0, svm.alphas[alpha_j]-svm.alphas[alpha_i])
             H = min(svm.C, svm.C+svm.alphas[alpha_j]-svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H :
            return 0

        'step3:calculate eta (the similarity of sample i and j)'
        eta = 2*svm.kernelMat[alpha_i, alpha_j]-svm.kernelMat[alpha_i, alpha_i]-svm.kernelMat[alpha_j, alpha_j]
        # print("yes"+str(svm.kernelMat[alpha_i,alpha_j]))
        if eta >= 0:
            return 0

        'step4:update alpha_j'
        svm.alphas[alpha_j] -= svm.train_y[alpha_j]*(error_i-error_j)/eta

        'step5:clip the alpha_j'
        if svm.alphas[alpha_j]>H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j]<L:
            svm.alphas[alpha_j] = L

        'step6: if alpha j not moving enough,just return(终止条件)'
        if abs(alpha_j_old-svm.alphas[alpha_j])<0.00001:
            updateError(svm, alpha_j)
            return 0

        'step7:update alpha i after optimizing aipha j'
        svm.alphas[alpha_i] += svm.train_y[alpha_i]*svm.train_y[alpha_j]*(alpha_j_old-svm.alphas[alpha_j])

        'step8:update threshold b'
        b1 = svm.b-error_i-svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernelMat[alpha_i, alpha_i]\
            - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)* svm.kernelMat[alpha_i, alpha_j]

        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)* svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        'step9:update error cache for alpha i, j after optimize alpha i, j and b'
        updateError(svm, alpha_i)
        updateError(svm, alpha_j)
        return 1
    else:
        return 0

# the main training procedure
def trainSVM(train_x, train_y, C ,toler, maxIter, kernelOption=('rbf',1.0)):
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)  # 创建一个类
    Iter = 0  # 迭代次数
    entireSet = True
    alphaPairsChanged = 0  # alpha变化次数

    while (Iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(svm.num_samples):
                alphaPairsChanged += innerLoop(svm, i)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (Iter, i, alphaPairsChanged))
            Iter += 1

        else:
            nonBoundIs = nonzero((svm.alphas.A > 0) * (svm.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLoop(svm, i)
            print("non-bound, iter: %d i:%d, pairs changed %d" % (Iter, i, alphaPairsChanged))
            Iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
    return svm

def testSVM(svm, test_x, test_y):
    test_x = mat(test_x)
    test_y = mat(test_y)
    num_samples = test_x.shape[0]
    supVectorIndex = nonzero(svm.alphas.A > 0)[0]

    supVector = svm.train_x[supVectorIndex]
    supVectorLabels = svm.train_y[supVectorIndex]
    supVectorAlphas = svm.alphas[supVectorIndex]
    print("There are %d support vector"% shape(supVector)[0])
    count = 0
    for i in range(num_samples):
        kernelValue = calcKernalValue(supVector, test_x[i, :], svm.kernelOpt)
        # print("yeah:"+str(kernelValue.shape)+" y2:"+str(supVectorLabels.shape))
        predict = kernelValue.T*multiply(supVectorLabels, supVectorAlphas)+svm.b
        if sign(predict) == sign(test_y[i]):
            count += 1
    accuracy = float(count)/num_samples
    #  print("准确率为%d"% acuuracy)
    return accuracy

def showSvm(svm):
    for i in range(svm.num_samples):
        print("第三："+str(svm.train_y[2]))
        print("第四：" + str(svm.train_y[3]))
        if svm.train_y[i] == -1:
            plt.scatter(svm.train_x[i, 0].tolist(), svm.train_x[i, 1].tolist(), marker='o', color='k', s=10, label='bad')  # 橙色为反例
        if svm.train_y[i] == 1:
            plt.scatter(svm.train_x[i, 0].tolist(), svm.train_x[i, 1].tolist(), color='blue', s=10, label='good')  # 色为正例
        supVectorIndex = nonzero(svm.alphas.A>0)[0]
    for t in supVectorIndex:
        plt.scatter(svm.train_x[t, 0], svm.train_x[t, 1], marker='o', color='y', s=20)
    # plt.show()

    # 画分界线
    w = zeros((2, 1))

    for i in supVectorIndex:
        w += multiply(svm.alphas[i]*svm.train_y[i], svm.train_x[i, :].T)
        print("oh"+str(type(svm.train_x))+str(svm.train_x.shape))
    min_x = min(svm.train_x[:, 0])[0, 0]
    max_x = max(svm.train_x[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlim(-2,10)
    plt.ylim(-8,8)
    plt.show()



data1 = 'D:/mechain_learning/第六章/SMO/T9l9.txt'
data_file = open(data1)
dataset = np.loadtxt(data_file, delimiter='\t')
print(dataset.shape)
train_x = dataset[0:80, 0:2]
train_y = dataset[0:80, 2]

test_x = dataset[80:100, 0:2]
test_y = dataset[80:100, 2]

train_y = mat(train_y).T
test_y = mat(test_y).T
print("happy1"+str(test_y.shape)+str(type(test_y)))
print("happy2"+str(test_x.shape))
print(type(train_x))
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 1.0))
print(svmClassifier.train_x.shape, svmClassifier.alphas.shape)
accuracy = testSVM(svmClassifier, test_x, test_y)
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showSvm(svmClassifier)











