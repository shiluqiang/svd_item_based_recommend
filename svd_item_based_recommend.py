# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:22:08 2019

@author: lj
"""
import numpy as np

def load_data():
    '''导入用户商品数据
    output:data(mat):用户商品信息矩阵
    '''
    data = [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
    return np.mat(data)

def cos_sim(x,y):
    '''计算余弦相似度
    input:x(mat):行向量矩阵，项目信息
          y(mat):行向量矩阵，项目信息
    output:x和y的余弦相似度
    '''
    nemerator = x * y.T #x与y的内积
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T) #x与y范数的乘积
    return (nemerator / denominator)[0,0]

def svd_dim(data):
    '''对原始用户商品矩阵进行降维
    input:data(mat):原始的用户商品矩阵
    output:newdata(mat):降维后的用户商品矩阵
    '''
    n = np.shape(data)[1]
    U,Sigma,VT = np.linalg.svd(data)
    sig2 = Sigma ** 2
    cut = 0
    for i in range(n):
        if sum(sig2[:i]) / sum(sig2) > 0.9:
            cut = i
            break
    Sig4 = np.mat(np.eye(cut)*Sigma[:cut]) #arrange Sig4 into a diagonal matrix
    xformedItems = data.T * U[:,:cut] * Sig4.I  #create transformed items
    newdata = xformedItems.T
    return newdata
    

def similarity(data):
    '''计算矩阵中任意两行之间的相似度
    input:data(mat):任意矩阵
    output:w(mat):矩阵中任意两行之间的相似度
    '''
    m = np.shape(data)[0] #用户数量
    #初始化相似度矩阵
    w = np.mat(np.zeros((m,m)))
    
    for i in range(m):
        for j in range(i,m):
            if j != i:
                #计算任意两行之间的相似度
                w[i,j] = cos_sim(data[i,:],data[j,:])
                w[j,i] = w[i,j]
            else:
                w[i,j] = 0
    return w

def item_based_recommend(data, w, user):
    '''基于商品相似度为用户user推荐商品
    input:  data(mat):商品用户矩阵
            w(mat):商品与商品之间的相似性
            user(int):用户的编号
    output: predict(list):推荐列表
    '''
    m, n = np.shape(data) # m:商品数量 n:用户数量
    interaction = data[:,user].T # 用户user的互动商品信息
    
    # 1、找到用户user没有互动的商品
    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0: # 用户user未打分项
            not_inter.append(i)
            
    # 2、对没有互动过的商品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(interaction) # 获取用户user对商品的互动信息
        for j in range(m): # 对每一个商品
            if item[0, j] != 0: # 利用互动过的商品预测
                if x not in predict:
                    predict[x] = w[x, j] * item[0, j]
                else:
                    predict[x] = predict[x] + w[x, j] * item[0, j]
    # 按照预测的大小从大到小排序
    return sorted(predict.items(), key=lambda d:d[1], reverse=True)

def top_k(predict, k):
    '''为用户推荐前k个商品
    input:  predict(list):排好序的商品列表
            k(int):推荐的商品个数
    output: top_recom(list):top_k个商品
    '''
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom

if __name__ == "__main__":
    # 1、导入用户商品数据
    print ("------------ 1. load data ------------")
    data = load_data()
    # 利用SVD对原始数据进行降维
    newdata = svd_dim(data)
    # 将用户商品矩阵转置成商品用户矩阵
    data_T = data.T
    newdata = newdata.T
    # 2、计算商品之间的相似性
    print ("------------ 2. calculate similarity between items -------------")    
    w = similarity(newdata)
    # 3、利用用户之间的相似性进行预测评分
    print ("------------ 3. predict ------------")    
    predict = item_based_recommend(data_T, w, 0)
    # 4、进行Top-K推荐
    print ("------------ 4. top_k recommendation ------------")
    top_recom = top_k(predict, 2)
    print (top_recom)
