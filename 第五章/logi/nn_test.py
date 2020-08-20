import numpy as np
import pandas as pd
import time

def load_data(path):
    data=pd.read_csv(path)
    X=data.iloc[:,:-1].values.T
    y=data.iloc[:,-1].values.reshape(np.shape(X)[1],1)
    return X,y

def sigmoid(z):
    return 1/(1+np.exp(-z))

class NN():#定义神经网络
    def __init__(self,layers=2,units=[4,1],alpha=0.01,N=10000):#定义神经网络的参数和超参数
        self.X=[]#训练集数据
        self.y=[]#训练集标签
        self.layers=layers#神经网络层数
        self.units=units#各层的单元数
        self.W=[]#各层的权重
        self.B=[]#各层的偏移
        self.Z=[]#各层的线性输出
        self.A=[]#各层的激活输出
        self.alpha=alpha#学习率
        self.N=N#迭代次数
    
    def init_para(self):#初始化W,B,Z,A矩阵
        for i in range(self.layers):
            if i == 0:
                self.W.append(np.random.randn(len(self.X),self.units[i]))
                self.A.append(self.X)
            else:
                self.W.append(np.random.randn(self.units[i-1],self.units[i]))
                self.A.append(np.random.randn(self.units[i],len(self.y)))
            self.B.append(np.random.randn(self.units[i],1))
            self.Z.append(np.random.randn(self.units[i],len(self.y)))
        self.A.append(np.random.randn(self.units[i],len(self.y)))

    def train(self,X,y):#在训练集和训练集标签上进行训练
        a=time.time()
        self.X=X
        self.y=y
        self.init_para()
        for i in range(self.N):
            self.fp()
            self.bp()
        b=time.time()
        print('Training used time:%.3fs'%(b-a))
    
    def fp(self):#正向传播
        for i in range(self.layers):
            self.Z[i]=np.dot(self.W[i].T,self.A[i])+self.B[i]
            self.A[i+1]=sigmoid(self.Z[i])
    
    def bp(self):#反向传播
        for i in range(self.layers):
            if i == 0:
                dZ=self.A[-(i+1)]-self.y.T
            else:
                dZ=np.dot(self.W[-i],dZ)*(sigmoid(self.Z[-(i+1)])*(1-sigmoid(self.Z[-(i+1)])))
            dW=np.dot(self.A[-(i+2)],dZ.T)/len(self.y)
            dB=np.sum(dZ,axis=1,keepdims=True)/len(self.y)
            self.W[-(i+1)]-=self.alpha*dW
            self.B[-(i+1)]-=self.alpha*dB
            
    def test(self, X, y):# 在测试机和测试机标签上进行验证并给出错误率
        y_pred=X.copy()
        for i in range(self.layers):
            y_pred=sigmoid(np.dot(self.W[i].T,y_pred)+self.B[i])
        errorcounts=0
        np.set_printoptions(suppress=True,precision=2)
        print('y_pred is:\n', y_pred)
        for i in range(len(y)):
            if np.abs(y_pred[0][i]-y[i])>0.5:
                errorcounts+=1
        return errorcounts/len(y)

X,y=load_data('table4-5.csv')    
nn=NN(layers=2,units=[2,1],alpha=1,N=2500)
nn.train(X,y)
print('errorrate is %.1f%%'%(nn.test(X,y)*100))
    
