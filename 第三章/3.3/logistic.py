from numpy import *
from math import *


#sigmoid函数
def sigmoid(inX):
    alt=[]
    for inx in inX:
        alt.append(1.0/(1+exp(-inx)))
    return mat(alt).transpose()

#梯度上升算法
def gradAscent(dataMatIn,LabelMatIn):
    dataMatrix=mat(dataMatIn)
    labelMatrix=mat(LabelMatIn).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001#向目标移动的步长
    maxCycles=50000#迭代次数
    weights=ones((n,1))#全1矩阵
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMatrix-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights


def plotBestFit(weights,dataMat,labelMat):
    import matplotlib.pyplot as plt
    #dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int (labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-0,1,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel("X1");plt.ylabel("X2")
    plt.show()



#读取csv文件

dataMat=[]
labelMat=[]
fr=open('D:/mechain_learning/第三章/3.3/watermelon3.csv')
for line in fr.readlines():
    lineArr=line.strip().split(",")
    dataMat.append([1.0,float(lineArr[1]),float(lineArr[2])])
    labelMat.append(int(lineArr[3]))
plotBestFit(gradAscent(dataMat,labelMat).getA(),dataMat,labelMat)