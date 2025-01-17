import warnings

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width',None)

RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)

def cal_weights(input_matrix):
    input_matrix = np.array(input_matrix)
    n, n1 = input_matrix.shape
    assert n == n1, '不是一个方阵'
    for i in range(n):
        for j in range(n):
            if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                raise ValueError('不是反互对称矩阵')

    eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

    max_idx = np.argmax(eigenvalues)
    max_eigen = eigenvalues[max_idx].real
    eigen = eigenvectors[:, max_idx].real
    eigen = eigen / eigen.sum()

    if n > 9:
        CR = None
        warnings.warn('无法判断一致性')
    elif n == 1:
        CR = 0
    else:
        CI = (max_eigen - n) / (n - 1)
        CR = CI / RI[n - 1]
        if CR > 0.1:
            raise Exception("CR>0.1，一致性检验不通过")
    return max_eigen, CR, eigen


def grs_count(X,s,criteria,c,columns):
    # 0、计算权重
    W = []
    for i in range(len(c)):
        max_eigen, CR, w = cal_weights(c[i])  # 权重
        W.append(w)
        # print(i)
    # print(W)
    max_eigen, CR, weight = cal_weights(criteria)  # AHP计算一级指标的权重
    # print(weight)
    weight_tmp = W.copy()
    weight_tmp[0] = W[0] * weight[0]
    weight_tmp[1] = W[1] * weight[1]
    weight_tmp[2] = W[2] * weight[2]
    weight_combine = []
    for item in weight_tmp:
        weight_combine.extend(item.tolist())
    # print(weight_combine)

    # 1、均值归一化
    x_mean=X.mean(axis=0)
    for i in range(X.columns.size):
        X.iloc[:, i] = X.iloc[:, i] / x_mean[i]
    # print("****归一化后的矩阵X*****")
    # print("X=", X)


    # 2、提取参考队列，Y为参考队列
    Y=[None]*X.columns.size
    s_list=[]
    for item in s:
        for value in item:
            s_list.append(value)
    # print("y===",Y)
    # print("slist===",s_list)
    for j in range(X.columns.size):
        if s_list[j]==1:
            Y[j] =max(X.iloc[:, j])
        else:
            Y[j] = min(X.iloc[:, j])
    # print("****参考序列Y*****")
    # print("Y=",Y)


    # YY=pd.DataFrame( index=columns,data=Y)
    # print("YY=",YY)
    # print(type(YY))
    # YYY=X.max(axis=0)
    # print("YYY=",YYY)
    # print(type(YYY))

    # 3、比较队列与参考队列相减
    t=pd.DataFrame()
    for j in range(X.index.size):
        temp=pd.Series(X.iloc[j,:]-Y)
        t=t.append(temp,ignore_index=False)
    # print("****比较序列与参考序列相减*****")
    t=t[columns]
    # print("t=",t)

    # 4、求最大差和最小差
    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho=0.5
    # print("****求最大差和最小差*****")
    # print("mmax=",mmax,"  mmin=",mmin)


    # 5、求关联系数
    ksi=(mmin+rho*mmax)/(abs(t)+rho*mmax)
    # print("****求关联系数*****")
    # print("ksi=",ksi)


    # 6、求关联度
    # print("weight_combine=",weight_combine)
    ksi=ksi*weight_combine
    # print("ksi quanzhong",ksi)

    #r=ksi.sum(axis=1)/ksi.columns.size
    r=ksi.sum(axis=1)

    # print("****求关联度*****")
    # print(r)

    # 7、关联度排序
    result=r.sort_values(ascending=False)
    # print("****关联度排序*****")
    # print(result)
    return result,r,ksi,W,weight,weight_combine
if __name__ == '__main__':

    # 0、数据初始化
    data=[[0.96,2.528,7,0.00001,0.00001],[0.948,3.093,17,0.353,0.0645],[0.919,4.347,31,0.00001,0.1508]]
    columns=['AUC','APL','nodeNumber','duplicateSubTree','duplicateSubAttr']
    index=["tree1","tree2","tree3"]



    # data = [[0.7,9,8,0.8,0.4],[0.7,9,8,0.8,0.4],[0.6,8,9,0.7,0.3],[0.8,10,7,0.9,0.5]]
    # columns=['m1','m2','m3','m4','m5']
    # index=["model1","model2","model3","model4"]

    X = pd.DataFrame(data=data, index=index, columns=columns)
    # print("****原矩阵X*****")
    # print("X=", X)

    s1 = np.array([1])
    s2 = np.array([-1, -1])
    s3 = np.array([-1, -1])
    # s4 = np.array([1, 1])
    s = [s1, s2, s3]
    # AHP 评价矩阵
    # 准则层重要性矩阵
    criteria = np.array([[1, 3, 4],
                         [1 / 3, 1, 2],
                         [1 / 4, 1 / 2, 1]])
    # 方案层
    c1 = np.array([[1]])
    c2 = np.array([[1, 1],
                   [1, 1]])
    c3 = np.array([[1, 1],
                   [1, 1]])
    # c4 = np.array([[1, 3],
    #    [1/3, 1]])
    c = [c1, c2, c3]


    result,result_origin,ksi,w2,w1,w3=grs_count(X,s,criteria,c,columns)
