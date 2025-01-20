#!/usr/bin/env python
# coding: utf-8

import sys
import os
import h5py
import numpy as np
from keras.optimizers import adam_v2



path = os.path.dirname('E:/RadioAttack2/')
sys.path.append(path)
print(path)
from Models.resNet import Net
from attack.attackMode import fgsm,pgd,BIM,MIM

os.environ["KERAS_BACKEND"] = "tensorflow"


if __name__ == '__main__':
    
    classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
           'FM', 'GMSK', 'OQPSK']

    datafile = 'E:/RadioAttack2/naturalDatasets/part0.h5'
    f = h5py.File(datafile, 'r')
    ########读取数据#######
    datax = f['X'][:]
    f.close()
    print(datax.shape)
    in_shape = datax.shape
    model = Net(in_shape, len(classes))
    adam = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    model.load_weights('E:/RadioAttack2/Models/ResNet_Model.h5')
    # model.predict(X_data)

    #====加载数据集
    # step = [0,2,17,18,21,22]
    # step = [0,1,2,3,9,17, 18,20,21,22,23]
    step = [6, 8, 10, 11, 12, 13, 14, 15, 16, 19]
    # for i in range(24):
    for i in step:

        datafile = 'E:/RadioAttack2/naturalDatasets/part'+str(i)+'.h5'
        f = h5py.File(datafile,'r')
        ########读取数据#######
        X_data = f['X'][:]
        Y_data = f['Y'][:]
        Z_data = f['Z'][:]
        f.close()
        print("aaaaaaaaaaaaaaaa")
        print(X_data[0])
        print(X_data[1200])
        print(X_data[2400])
        print(X_data[3600])
        print(X_data[4800])
        print(X_data[6000])
        print(X_data[7200])
        print(X_data[8400])
        print(X_data[9600])
        print(X_data[10800])
        print(X_data[17999])

        # if i==0:
        #     X_data = datax
        #     Y_data = datay
        #     Z_data = dataz
        # else:
        #     X_data = np.vstack((X_data,datax))
        #     Y_data = np.vstack((Y_data,datay))
        #     Z_data = np.vstack((Z_data,dataz))
        # adv_dir = 'E:/RadioAttack2/attResults/'
        # eps = 0.037
        # nb_iter = 50
        # eps_iter = 0.006
        # if (i == 0 or i == 1 or i == 2):
        #     eps = 0.07
        #     eps_iter = 0.008
        #     nb_iter = 70
        # elif(i==17 or i==18):
        #     eps = 0.2
        # elif(i==3 or i==9 or i==20 or i==21 or i==22 or i==23):
        #     eps = 0.045
        #     eps_iter = 0.007
        #     nb_iter = 70

        # # FGSM
        # fgsm_result, fgsm_advx = fgsm(model, X_data,eps)
        # # 攻击结果表
        # fdir = adv_dir + 'fgsm_advx/fgsm_resultPart'+str(i)+'.csv'
        # fgsm_result.to_csv(fdir, index=False)
        # # 存储攻击样本
        # advfile = adv_dir + 'fgsm_advx/fgsm_advxPart'+str(i)+'.h5'
        # fw = h5py.File(advfile, 'w')
        # fw['X'] = np.vstack(fgsm_advx)
        # fw['Y'] = np.vstack(Y_data)
        # fw['Z'] = np.vstack(Z_data)
        
        
        # print('X shape:', fw['X'].shape)
        # print('Y shape:',fw['Y'].shape)
        # print('Z shape:',fw['Z'].shape)
        # fw.close()
        
        # # PGD
        # pgd_result, pgd_advx = pgd(model, X_data ,eps,eps_iter,nb_iter)
        # pdir = adv_dir + 'pgd_advx/pgd_resultPart'+str(i)+'.csv'
        # pgd_result.to_csv(pdir, index=False)
        # advfile = adv_dir + 'pgd_advx/pgd_advxPart'+str(i)+'.h5'
        # fw = h5py.File(advfile, 'w')
        # fw['X'] = np.vstack(pgd_advx)
        # fw['Y'] = np.vstack(Y_data)
        # fw['Z'] = np.vstack(Z_data)
        # print('X shape:', fw['X'].shape)
        # print('Y shape:', fw['Y'].shape)
        # print('Z shape:', fw['Z'].shape)
        # fw.close()
        
        # BIM
        # bim_result, bim_advx = BIM(model, X_data,eps,eps_iter,nb_iter)
        # bdir = adv_dir + 'bim_advx/bim_resultPart'+str(i)+'.csv'
        # bim_result.to_csv(bdir, index=False)
        # advfile = adv_dir + 'bim_advx/bim_advxPart'+str(i)+'.h5'
        # fw = h5py.File(advfile, 'w')
        # fw['X'] = np.vstack(bim_advx)
        # fw['Y'] = np.vstack(Y_data)
        # fw['Z'] = np.vstack(Z_data)
        # print('X shape:', fw['X'].shape)
        # print('Y shape:', fw['Y'].shape)
        # print('Z shape:', fw['Z'].shape)
        # fw.close()

        # # MIM
        # mim_result, mim_advx= MIM(model, X_data,eps,eps_iter,nb_iter)
        # mdir = adv_dir + 'mim_advx/mim_resultPart'+str(i)+'.csv'
        # mim_result.to_csv(mdir, index=False)
        # advfile = adv_dir + 'mim_advx/mim_advxPart'+str(i)+'.h5'
        
        # fw = h5py.File(advfile, 'w')
        # fw['X'] = np.vstack(mim_advx)
        # fw['Y'] = np.vstack(Y_data)
        # fw['Z'] = np.vstack(Z_data)
        # print('X shape:', fw['X'].shape)
        # print('Y shape:', fw['Y'].shape)
        # print('Z shape:', fw['Z'].shape)
        # fw.close()






