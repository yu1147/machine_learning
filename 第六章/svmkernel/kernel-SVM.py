from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

def loadDataSet(path):#根据路径加载数据集
    dataset=pd.read_csv(path)
    dataMat=dataset[dataset.columns[:-1]].values.tolist()
    labelMat=dataset[dataset.columns[-1]].values.tolist()
    return dataMat, labelMat


def selectJrand(i, m):#随机选择j
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def show(dataArr, labelArr, alphas, b):
    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'or')
        elif labelArr[i] == 1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'Dg')
    # print alphas.shape, mat(labelArr).shape, multiply(alphas, mat(labelArr)).shape
    c = sum(multiply(multiply(alphas.T, mat(labelArr)), mat(dataArr).T), axis=1)
    minY = min(m[1] for m in dataArr)
    maxY = max(m[1] for m in dataArr)
    plt.plot([sum((- b - c[1] * minY) / c[0]), sum((- b - c[1] * maxY) / c[0])], [minY, maxY])
    plt.plot([sum((- b + 1 - c[1] * minY) / c[0]), sum((- b + 1 - c[1] * maxY) / c[0])], [minY, maxY])
    plt.plot([sum((- b - 1 - c[1] * minY) / c[0]), sum((- b - 1 - c[1] * maxY) / c[0])], [minY, maxY])
    plt.show()

def kernelTrans(X, A, kTup):#A为行向量，X中的每一列与A进行高斯核运算，
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':#lin为线性核函数，rbf为高斯核函数
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T#计算欧式距离
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:#定义类
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):#kTup=['模式',阈值]
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C#精度
        self.tol = toler#软间隔容忍度
        self.m = shape(dataMatIn)[0]#data的维度
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))#存放E(i)=f(x)-y(i)
        self.K = mat(zeros((self.m, self.m)))#半正定矩阵
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)#对核矩阵逐列进行核变换

def calcEk(oS, k):#计算E(k)
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):#启发式，根据|E(i)-E(j)|选择j
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]#返回E中非0元素索引
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:#若索引长度为0或1，既E中非0个数为0或1，无法根据|E(i)-E(j)|选取j,则随机选取j,并计算E(j)
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#更新E(k)
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def clipAlpha(aj, H, L):#根据上下限确定alpha[j]
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C))\
            or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:#规定异类上下限L，H
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:#规定同类上下限L，H
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] \
                - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] \
                - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1#成功更新alpha[i]、alpha[j]、b
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 1)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler,kTup)#创建一个类
    Iter = 0#记录迭代次数
    entireSet = True
    alphaPairsChanged = 0#记录alpha变化次数
    while Iter < maxIter and (alphaPairsChanged > 0 or entireSet):#若超出迭代次数或上一次循环中没发生alpha的变化则停止循环
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (Iter, i, alphaPairsChanged))
            Iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (Iter, i, alphaPairsChanged))
            Iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % Iter)
    return oS.b, oS.alphas

def showRBF(dataArr, labelArr, alphas):
    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'or')#橙色为反例
        elif labelArr[i] == 1:
            plt.plot(dataArr[i][0], dataArr[i][1], 'Dg')#绿色为正例
    dataMat = mat(dataArr)
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    for i in range(shape(sVs)[0]):
        plt.plot(dataArr[i][0], dataArr[i][1], 'ob')#蓝色为支持向量
    plt.show()

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet("kernel-train.csv")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', 1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet("kernel-test.csv")
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', 1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))    
    showRBF(dataArr, labelArr, alphas)

testRbf()
