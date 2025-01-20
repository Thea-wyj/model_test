#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input,Reshape
import cleverhans
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

"""FGSM attack algorithm

        Args:
            args: Various Parameters
            dataset: original image but conform to network input format
            
        Returns:
            data_result: csv文件格式, head = "nat_lable", "advx_lable"
            x_adv_list: Matrix after adding disturbance
        """

def fgsm(model,dataset,eps):
    result_list = []
    advx_list = []

    #=====换成信号样本
    for i in range(dataset.shape[0]):
        data = dataset[i]
        d_shp = data.shape
        data = np.array(data).reshape((1,1024,2))
        data_norm = np.linalg.norm(data)
        nat_lable = tf.argmax(model.predict(data),1) #原样本预测标签
        if (data_norm >= 220 and data_norm<300):
            advx = fast_gradient_method(model,data,eps*4,np.inf)
        elif(data_norm>=300):
            advx = fast_gradient_method(model, data, eps * 5.2, np.inf)
        else:
            advx = fast_gradient_method(model,data,eps,np.inf)
        advx = fast_gradient_method(model,data,eps,np.inf)
        advx_lable = tf.argmax(model.predict(advx),1) #攻击样本预测标签
        pertur = np.sqrt(np.sum(np.square(advx-data)))

        result_dict = {}
        result_dict.update({'nat_lable':int(nat_lable),'advx_lable':int(advx_lable),'perturbation':pertur,'data_norm':data_norm})
        result_list.append(result_dict)
        advx_list.append(advx)


    data_results = pd.DataFrame(result_list)
    advx_list = np.array(advx_list)
    return data_results,advx_list

def pgd(model,dataset,eps,eps_iter,nb_iter):
    result_list = []
    advx_list = []


    #=====换成信号样本

    for i in range(dataset.shape[0]):
        data = dataset[i]
        d_shp = data.shape

        data = np.array(data).reshape((1, 1024, 2))
        data_norm = np.linalg.norm(data)
        nat_lable = tf.argmax(model(data),1) #原样本预测标签
        if (data_norm >= 220 and data_norm<300):
            advx = projected_gradient_descent(model, data, eps * 4, eps_iter+0.002,nb_iter+50, np.inf)  # 参数？？？？
        elif(data_norm>=300):
            advx = projected_gradient_descent(model, data, eps * 5.2, eps_iter+0.004, nb_iter+50, np.inf)
        else:
            advx = projected_gradient_descent(model, data, eps*1.2, eps_iter+0.01, nb_iter+10, np.inf)  # 参数？？？？
        advx_lable = tf.argmax(model(advx),1) #攻击样本预测标签
        pertur = np.sqrt(np.sum(np.square(advx-data)))

        result_dict = {}
        result_dict.update({'nat_lable':int(nat_lable),'advx_lable':int(advx_lable),'perturbation':pertur,'data_norm':data_norm})
        result_list.append(result_dict)
        advx_list.append(advx)

    data_results = pd.DataFrame(result_list)
    advx_list = np.array(advx_list)
    return data_results,advx_list

def BIM(model,dataset,eps,eps_iter,nb_iter):
    result_list = []
    advx_list = []

    #=====换成信号样本

    for i in range(dataset.shape[0]):
        data = dataset[i]
        d_shp = data.shape
        data = np.array(data).reshape((1, 1024, 2))
        data_norm = np.linalg.norm(data)
        nat_lable = tf.argmax(model(data),1) #原样本预测标签
        if (data_norm >= 220 and data_norm < 300):
            advx = basic_iterative_method(model, data, eps * 4, eps_iter+0.002, nb_iter+50, np.inf)  # 参数？？？？
        elif (data_norm >= 300):
            advx = basic_iterative_method(model, data, eps * 5.2,eps_iter+0.004, nb_iter+50, np.inf)
        else:
            advx = basic_iterative_method(model, data, eps, eps_iter, nb_iter, np.inf)  # 参数？？？？
        advx_lable = tf.argmax(model(advx),1) #攻击样本预测标签
        pertur = np.sqrt(np.sum(np.square(advx-data)))


        result_dict = {}
        result_dict.update({'nat_lable':int(nat_lable),'advx_lable':int(advx_lable),'perturbation':pertur,'data_norm':data_norm})
        result_list.append(result_dict)
        advx_list.append(advx)

    data_results = pd.DataFrame(result_list)
    advx_list = np.array(advx_list)
    return data_results,advx_list

def MIM(model,dataset,eps,eps_iter,nb_iter):
    result_list = []
    advx_list = []

    #=====换成信号样本

    for i in range(dataset.shape[0]):
        data = dataset[i]
        d_shp = data.shape
        data = np.array(data).reshape((1, 1024, 2))
        data_norm = np.linalg.norm(data)
        nat_lable = tf.argmax(model(data),1) #原样本预测标签
        if(data_norm>=160 and data_norm<200):
            advx = momentum_iterative_method(model, data, eps, eps_iter + 0.001, nb_iter + 40, np.inf)  # 参数？？？？
        elif (data_norm >= 200 and data_norm < 300):
            advx = momentum_iterative_method(model, data, eps*1 , eps_iter+0.002, nb_iter+40, np.inf)  # 参数？？？？ 1.2 1.5 3
        elif (data_norm >= 300 and data_norm<400):
            advx = momentum_iterative_method(model, data, eps * 1.2, eps_iter+0.003, nb_iter+40, np.inf)
        elif(data_norm>=400):
            advx = momentum_iterative_method(model, data, eps * 1.5, eps_iter + 0.004, nb_iter + 40, np.inf)
        else:
            advx = momentum_iterative_method(model, data, eps, eps_iter, nb_iter, np.inf)  # 参数？？？？
        advx_lable = tf.argmax(model(advx),1) #攻击样本预测标签
        pertur = np.sqrt(np.sum(np.square(advx-data)))

        result_dict = {}
        result_dict.update({'nat_lable':int(nat_lable),'advx_lable':int(advx_lable),'perturbation':pertur,'data_norm':data_norm})
        result_list.append(result_dict)
        advx_list.append(advx)

    data_results = pd.DataFrame(result_list)
    advx_list = np.array(advx_list)
    return data_results,advx_list
        

# 
# advx=fast_gradient_method(model,data,0.1,np.inf)
# predic=torch.argmax(model(advx),dim=1)
