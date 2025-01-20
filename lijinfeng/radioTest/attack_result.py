import csv

import pandas as pd
import numpy as np
from pylab import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import json
import sys
import os
import h5py
import argparse
import numpy as np
import random

from radioAttack.attack.single_pic import plot_picture
from keras.optimizers import adam_v2

from radioAttack.Models.resNet import Net
from radioAttack.attack.attackMode import fgsm,pgd,BIM,MIM


def accuracy(file,name,i):
    # for i in range(24):
    #     mat = [file['nat_lable'] != file['advx_lable']].count()[0]
    #     type_acc = round((mat / file) *100, 3)
    #     print(name+" single mode " + str(i) + " accuracy on adversarial samples:   " + str(type_acc))
    #     res.append(type_acc)
    num = file['nat_lable'].count()
    match = file[file['nat_lable'] != file['advx_lable']].count()[0]
    acc = round((match / num) * 100, 3)
    print(name + " single mode " + str(i) + " accuracy on adversarial samples:   " + str(acc))

    return acc,num,match

def perturbation(file):
    with open(file,'r') as f:
        data = csv.reader(f,delimiter=',')
        next(f)
        r_adv = 0
        for row in data:
            r_adv += round(float(row[2])/float(row[3]),3)
    return r_adv



# with open(dirm,'r') as f:
#     data = csv.reader(f,delimiter=',')
#     next(f)
#     r_adv = sum(float(row[2]) for row in data)
#     r_adv = round(r_adv / sample_size, 3)
#     print(r_adv)


def attack_Result(augmentation_methods):

    dir = 'attResults'
    fgsm_res = []
    pgd_res = []
    bim_res = []
    mim_res = []
    # 生成的扰动小于0.05的攻击样本的数量
    fgsm_num = 0
    pgd_num = 0
    bim_num = 0
    mim_num = 0
    # 攻击成功的数量
    fgsm_matchNum = 0
    pgd_matchNum = 0
    bim_matchNum = 0
    mim_matchNum = 0

    #扰动比例的均值
    f_per = 0
    p_per = 0
    b_per = 0
    m_per = 0
    fgsm_perList = []
    pgd_perList = []
    bim_perList = []
    mim_perList = []

    step = [6]
    for i in step:
        for method in augmentation_methods:
            if method == 'fgsm':
                dirf = dir + '/fgsm_advx/fgsm_resultPart' + str(i) + '.csv'
                file1 = pd.read_csv(dirf)
                fAcc, num1, match1 = accuracy(file1, "fgsm", i)
                fgsm_res.append(fAcc)
                fgsm_num += num1
                fgsm_matchNum += match1
                f_per = round((perturbation(dirf)/num1)*100,2) #应该是match1吧
                fgsm_perList.append(f_per)

            elif method == 'pgd':
                dirp = dir + '/pgd_advx/pgd_resultPart' + str(i) + '.csv'
                file2 = pd.read_csv(dirp)
                pAcc, num2, match2 = accuracy(file2, "pgd", i)
                pgd_res.append(pAcc)
                pgd_num += num2
                pgd_matchNum += match2
                p_per = round((perturbation(dirp)/num2)*100,2)
                pgd_perList.append(p_per)

            elif method == 'bim':
                dirb = dir + '/bim_advx/bim_resultPart' + str(i) + '.csv'
                file3 = pd.read_csv(dirb)
                bAcc, num3, match3 = accuracy(file3, "bim", i)
                bim_res.append(bAcc)
                bim_num += num3
                bim_matchNum += match3
                b_per = round((perturbation(dirb)/num3)*100,2)
                bim_perList.append(b_per)

            elif method == 'mim':
                dirm = dir + '/mim_advx/mim_resultPart' + str(i) + '.csv'
                file4 = pd.read_csv(dirm)
                mAcc, num4, match4 = accuracy(file4, "mim", i)
                mim_res.append(mAcc)
                mim_num += num4
                mim_matchNum += match4
                m_per = round((perturbation(dirm)/num4)*100,2)
                mim_perList.append(m_per)
            

    print(fgsm_num)
    print(fgsm_matchNum)
    print(pgd_num)
    print(pgd_matchNum)
    print(bim_num)
    print(bim_matchNum)
    print(mim_num)
    print(mim_matchNum)

    ASR_data = {}
    for method in augmentation_methods:
        if method == 'fgsm':
            acc = round((fgsm_matchNum / fgsm_num) * 100, 3)
        elif method == 'pgd':
            acc = round((pgd_matchNum / pgd_num) * 100, 3)
        elif method == 'bim':
            acc = round((bim_matchNum / bim_num) * 100, 3)
        elif method == 'mim':
            acc = round((mim_matchNum / mim_num) * 100, 3)
        ASR_data[method] = acc


    with open("attack_result/result.json", "w") as f:
        json.dump(ASR_data, f)

    # print(" accuracy on allover fgsm adversarial samples:   " + str(facc))
    # print(" accuracy on allover pgd adversarial samples:   " + str(pacc))
    # print(" accuracy on allover bim adversarial samples:   " + str(bacc))
    # print(" accuracy on allover mim adversarial samples:   " + str(macc))

    # f_per = round((f_per / fgsm_num), 3)
    # p_per = round((p_per / pgd_num), 3)
    # b_per = round((b_per / bim_num), 3)
    # m_per = round((m_per / mim_num), 3)
    # print(" average perturbation on  fgsm adversarial samples:   " + str(f_per))
    # print(" average perturbation on pgd adversarial samples:   " + str(p_per))
    # print(" average perturbation onr bim adversarial samples:   " + str(b_per))
    # print(" average perturbation on mim adversarial samples:   " + str(m_per))

    # fgsm_res = np.array(fgsm_res)
    # pgd_res = np.array(pgd_res)
    # bim_res = np.array(bim_res)
    # mim_res = np.array(mim_res)

    # plt.rcParams['figure.figsize'] = (17.8, 7.2)

    # classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    #            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    #            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    #            'FM', 'GMSK', 'OQPSK']


    # sort_indices = np.argsort(pgd_res)
    # print("sort")
    # print(sort_indices)
    # fgsm_res = [fgsm_res[i] for i in sort_indices]
    # pgd_res = [pgd_res[i] for i in sort_indices]
    # bim_res = [bim_res[i] for i in sort_indices]
    # mim_res = [mim_res[i] for i in sort_indices]
    # classes = [classes[i] for i in sort_indices]
    # fgsm_perList = [fgsm_perList[i] for i in sort_indices]
    # pgd_perList = [pgd_perList[i] for i in sort_indices]
    # bim_perList = [bim_perList[i] for i in sort_indices]
    # mim_perList = [mim_perList[i] for i in sort_indices]

    # plt.xticks(size=8.5)

    # plt.xlabel("调制信号类型")
    # plt.ylabel("准确度")
    # plt.suptitle('图1 攻击成功率',
    #              x=0.5,  # x轴方向位置
    #              y=0.98,  # y轴方向位置
    #              size=9,  # 大小
    #              ha='center',  # 水平位置，相对于x,y，可选参数：{'center', 'left', right'}, default: 'center'
    #              va='top',  # 垂直位置，相对于x,y，可选参数：{'top', 'center', 'bottom', 'baseline'}, default: 'top'
    #              weight='bold',  # 字体粗细，以下参数可选
    #              rotation=1,  ##标题旋转，传入旋转度数，也可以传入vertical', 'horizontal'
    #              )
    
    # plt.plot(classes, fgsm_res, label='fgsm')
    # plt.plot(classes, pgd_res, label='pgd')
    # plt.plot(classes, bim_res, label='bim')
    # plt.plot(classes, mim_res, label='mim')
    
    # plt.legend()
    # plt.savefig(dir+'/picture/att_acc.png')


    # plt.legend()
    # plt.savefig(dir + '/picture/perturation.png')

    # plt.show()

def parseCmdArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='radioAttack/Models/ResNet_Model.h5',
                        help='model path')

    parser.add_argument('--data_path', type=str,
                        default='datas/testdatas',
                        help='data path')

    parser.add_argument('--methodList', type=str, nargs='+', default=["fgsm",
     "pgd", "bim","mim"], help='method namelist')
    
    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
           'FM', 'GMSK', 'OQPSK']


    args = parseCmdArgument()  # 解析命令行参数
    model_path = args.model_path
    data_folder = args.data_path
    augmentation_methods = args.methodList

    part_file_name = 'part0.h5'
    data_file = os.path.join(data_folder, part_file_name)
    f = h5py.File(data_file, 'r')

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 设置绘图字体
    matplotlib.rcParams['axes.unicode_minus'] = False

    ########读取模型#######
    datax = f['X'][:]
    f.close()
    print(datax.shape)
    in_shape = datax.shape
    model = Net(in_shape, len(classes))
    adam = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    model.load_weights(model_path)
    # model.predict(X_data)

    #====加载数据集
    i = random.randint(1, 23)

    datafile = data_folder+'/part'+str(i)+'.h5'
    f = h5py.File(datafile,'r')
    ########读取数据#######
    X_data = f['X'][:]
    Y_data = f['Y'][:]
    Z_data = f['Z'][:]
    f.close()
    # print("aaaaaaaaaaaaaaaa")
    # print(X_data[0])
    # print(X_data[1200])
    # print(X_data[2400])
    # print(X_data[3600])
    # print(X_data[4800])
    # print(X_data[6000])
    # print(X_data[7200])
    # print(X_data[8400])
    # print(X_data[9600])
    # print(X_data[10800])
    # print(X_data[17999])

    # if i==0:
    #     X_data = datax
    #     Y_data = datay
    #     Z_data = dataz
    # else:
    #     X_data = np.vstack((X_data,datax))
    #     Y_data = np.vstack((Y_data,datay))
    #     Z_data = np.vstack((Z_data,dataz))
    adv_dir = 'attResults/'
    
    eps = 0.037
    nb_iter = 50
    eps_iter = 0.006
    if (i == 0 or i == 1 or i == 2):
        eps = 0.07
        eps_iter = 0.008
        nb_iter = 70
    elif(i==17 or i==18):
        eps = 0.2
    elif(i==3 or i==9 or i==20 or i==21 or i==22 or i==23):
        eps = 0.045
        eps_iter = 0.007
        nb_iter = 70

    for method in augmentation_methods:
        if method == 'fgsm':
            # FGSM
            fgsm_result, fgsm_advx = fgsm(model, X_data,eps)
            # 攻击结果表
            fdir = adv_dir + 'fgsm_advx/fgsm_resultPart'+str(i)+'.csv'
            fgsm_result.to_csv(fdir, index=False)
            # 存储攻击样本
            advfile = adv_dir + 'fgsm_advx/fgsm_advxPart'+str(i)+'.h5'
            fw = h5py.File(advfile, 'w')
            fw['X'] = np.vstack(fgsm_advx)
            fw['Y'] = np.vstack(Y_data)
            fw['Z'] = np.vstack(Z_data)
            
            
            print('X shape:', fw['X'].shape)
            print('Y shape:',fw['Y'].shape)
            print('Z shape:',fw['Z'].shape)
            fw.close()
        
        elif method == 'pgd':
            # PGD
            pgd_result, pgd_advx = pgd(model, X_data ,eps,eps_iter,nb_iter)
            pdir = adv_dir + 'pgd_advx/pgd_resultPart'+str(i)+'.csv'
            pgd_result.to_csv(pdir, index=False)
            advfile = adv_dir + 'pgd_advx/pgd_advxPart'+str(i)+'.h5'
            fw = h5py.File(advfile, 'w')
            fw['X'] = np.vstack(pgd_advx)
            fw['Y'] = np.vstack(Y_data)
            fw['Z'] = np.vstack(Z_data)
            print('X shape:', fw['X'].shape)
            print('Y shape:', fw['Y'].shape)
            print('Z shape:', fw['Z'].shape)
            fw.close()
        
        elif method == 'bim':
            # BIM
            bim_result, bim_advx = BIM(model, X_data,eps,eps_iter,nb_iter)
            bdir = adv_dir + 'bim_advx/bim_resultPart'+str(i)+'.csv'
            bim_result.to_csv(bdir, index=False)
            advfile = adv_dir + 'bim_advx/bim_advxPart'+str(i)+'.h5'
            fw = h5py.File(advfile, 'w')
            fw['X'] = np.vstack(bim_advx)
            fw['Y'] = np.vstack(Y_data)
            fw['Z'] = np.vstack(Z_data)
            print('X shape:', fw['X'].shape)
            print('Y shape:', fw['Y'].shape)
            print('Z shape:', fw['Z'].shape)
            fw.close()

        elif method == 'mim':
            # MIM
            mim_result, mim_advx= MIM(model, X_data,eps,eps_iter,nb_iter)
            mdir = adv_dir + 'mim_advx/mim_resultPart'+str(i)+'.csv'
            mim_result.to_csv(mdir, index=False)
            advfile = adv_dir + 'mim_advx/mim_advxPart'+str(i)+'.h5'
            
            fw = h5py.File(advfile, 'w')
            fw['X'] = np.vstack(mim_advx)
            fw['Y'] = np.vstack(Y_data)
            fw['Z'] = np.vstack(Z_data)
            print('X shape:', fw['X'].shape)
            print('Y shape:', fw['Y'].shape)
            print('Z shape:', fw['Z'].shape)
            fw.close()


    attack_Result(augmentation_methods)



    for method in augmentation_methods:
        att_file = 'attResults/' + method + '_advx/' + method + '_advxPart' + str(i) + '.h5'
        plot_picture(datafile, att_file, method)


# fgsm_res = np.array(fgsm_res)
# pgd_res = np.array(pgd_res)
# bim_res = np.array(bim_res)
# mim_res = np.array(mim_res)
#
# x = np.arange(0,24)
# plt.plot(x,fgsm_res,label = 'fgsm')
# plt.plot(x,pgd_res,label = 'pgd')
# plt.plot(x,bim_res,label = 'bim')
# plt.plot(x,mim_res,label = 'mim')
#
# plt.legend()
# plt.savefig(dir+'/picture/att_succ.png')
# plt.show()
#
#
# print(fAcc,pAcc,bAcc,mAcc)
# with open(dirm,'r') as f:
#     data = csv.reader(f,delimiter=',')
#     next(f)
#     r_adv = sum(float(row[2]) for row in data)
#     r_adv = round(r_adv / sample_size, 3)
#     print(r_adv)
