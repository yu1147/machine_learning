# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:24:10 2020

@author: LUM
"""

import numpy as np
import pandas as pd

def load_data(path):
    data=pd.read_csv(path)
    return data

def find_column_index_discrate(xi):#返回离散特征的列索引
    # print(np.shape(data)[1])
    for i in range(np.shape(data)[1]):
        for j in data[data.columns[i]].isin([xi]):
            if j:
                # print(data.columns[i])
                return data.columns[i]

def P_singal_discrete(xi):#计算离散数据的独立概率
    column_index=find_column_index_discrate(xi)
    print(np.shape(data[data[column_index].isin([xi])]))
    return np.shape(data[data[column_index].isin([xi])])[0]/np.shape(data)[0]

def P_multi_discrete(xi,label):#计算离散数据在特定label下的独立概率
    column_index=find_column_index_discrate(xi)
    data_label=data.loc[data['label']==label]
    return np.shape(data_label[data_label[column_index].isin([xi])])[0]/np.shape(data_label)[0]

def P_singal_continuous(xi,title):#计算连续数据的正态分布概率
    data_xi=data[title].values
    xi_mean=np.mean(data_xi)
    xi_std=np.std(data_xi)
    return 1/(np.sqrt(2*np.pi)*xi_std)*np.exp(-((xi-xi_mean)**2)/(2*xi_std**2))

def P_multi_continuous(xi,label,title):#计算连续数据在指定label下的正态分布概率
    data_xi=data.loc[data['label']==label][title].values
    xi_mean=np.mean(data_xi)
    xi_std=np.std(data_xi)
    return (1/(np.sqrt(2*np.pi)*xi_std))*np.exp(-((xi-xi_mean)**2)/(2*xi_std**2))

def P(label,X,**kw):#计算在指定特征下是否为指定label
    product1=len(data[data['label']==label].index)/np.shape(data)[0]
    product0=len(data[data['label']!=label].index)/np.shape(data)[0]
    for xi in X:#计算离散数据
        product1=product1*P_multi_discrete(xi,label)
        product0=product0*P_multi_discrete(xi,0-label)
    for title in kw:#计算连续数据
        product1=product1*P_multi_continuous(kw[title],label,title)
        product0=product0*P_multi_continuous(kw[title],0-label,title)
    if product1>product0:
        return True
    else:
        return False

def test():
    dataset=data[data.columns[:-1]]
    labelset=data[data.columns[-1]]
    discrate_col_index=[]#离散数据列索引
    continuous_col_index=[]#连续数据列索引
    # print(dataset[dataset.columns[4]].dtype)
    for i in range(np.shape(dataset)[1]):
        if dataset[dataset.columns[i]].dtype=='object':#判断为离散数据
            discrate_col_index.append(i)
        else:#判断为连续数据
            continuous_col_index.append(i)
    dataset_dicrete=dataset.iloc[:,discrate_col_index].values.tolist()#切片出离散数据
    dataset_continuous=dataset.iloc[:,continuous_col_index].values.tolist()#切片出连续数据
    rightcount=0
    for i in range(np.shape(data)[0]):
        if P(labelset[i], dataset_dicrete[i], density=dataset_continuous[i][0], sugar=dataset_continuous[i][1]):
            rightcount +=1
    return rightcount/len(labelset)

data=load_data('dataset3.0.csv')
rightrate=test()
print("正确率为：",str(rightrate))  
