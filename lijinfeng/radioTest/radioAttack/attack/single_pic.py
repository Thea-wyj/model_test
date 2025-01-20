import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_picture(original_file,att_file,method_name):
    f = h5py.File(original_file, 'r')
    datax = f['X'][:]
    datay = f['Y'][:]
    dataz = f['Z'][:]
    f.close()
    X_data = datax
    Y_data = datay
    Z_data = dataz

    f2 = h5py.File(att_file, 'r')
    X_adv = f2['X'][:]
    Y_adv = f2['Y'][:]
    Z_adv = f2['Z'][:]
    f2.close()

    sample_idx = 15
    print('snr:', Z_data[sample_idx])
    print('Y', Y_data[sample_idx])
    plt_data = X_data[sample_idx].T
    plt_advdata = X_adv[sample_idx].T

    # 绘制原始样本图
    fig, ax1 = plt.subplots(figsize=(30, 5))
    x = np.arange(0, 4096, 4)
    ax1.plot(x, plt_data[0], label='I signal', linewidth=0.5)
    ax1.plot(x, plt_data[1], color='red', label='Q signal', linewidth=0.5)
    ax1.set_title('natural sample')
    ax1.set_xlabel("x-point")
    ax1.set_ylabel("y-signal value")
    ax1.legend()

    # 保存原始样本图
    plt.savefig('attack_result/natural sample.jpg')

    # 绘制对抗样本图
    fig, ax2 = plt.subplots(figsize=(30, 5))
    ax2.plot(x, plt_advdata[0], label='I signal', linewidth=0.5)
    ax2.plot(x, plt_advdata[1], color='red', label='Q signal', linewidth=0.5)
    ax2.set_title(method_name+' attack sample')
    ax2.set_xlabel("x-point")
    ax2.set_ylabel("y-signal value")
    ax2.legend()

    # 保存对抗样本图
    plt.savefig('attack_result/'+method_name+' attack sample'+'.jpg')

    # 计算原始样本与对抗样本之间的差异
    noise = X_adv[sample_idx].T - X_data[sample_idx].T

    # 只绘制对抗样本和原始样本的比较图
    fig, ax3 = plt.subplots(figsize=(30, 10))

    ax3.plot(x, plt_data[0], label='I signal', linewidth=0.5)
    ax3.plot(x, plt_data[1], color='red', label='Q signal', linewidth=0.5)
    ax3.plot(x, plt_advdata[0], label='I signal (adversarial)', linewidth=0.5)
    ax3.plot(x, plt_advdata[1], color='red', label='Q signal (adversarial)', linewidth=0.5)
    ax3.set_title('Comparison between natural and ' + method_name +' attack samples')
    ax3.set_xlabel("x-point")
    ax3.set_ylabel("y-signal value")
    ax3.legend()

    # 保存图像为jpg格式
    plt.savefig('attack_result/'+method_name+' comparison.jpg')


    # 分别绘制I和Q方向上的噪声信号图
    fig, (ax4, ax5) = plt.subplots(2, 1, figsize=(30, 10))

    # 绘制I方向上的噪声
    ax4.plot(x, noise[0], label='I signal difference', linewidth=0.5, color='blue')
    ax4.set_title("Perturbation added by " + method_name + " (I signal)")
    ax4.set_xlabel("x-point")
    ax4.set_ylabel("y-signal value")  # 与原始样本一样的y坐标
    ax4.legend()
    ax4.set_ylim([min(X_data[sample_idx].T[0]), max(X_data[sample_idx].T[0])])  # 设置y轴刻度

    # 绘制Q方向上的噪声
    ax5.plot(x, noise[1], label='Q signal difference', linewidth=0.5, color='red')
    ax5.set_title("Perturbation added by " + method_name + " (Q signal)")
    ax5.set_xlabel("x-point")
    ax5.set_ylabel("y-signal value")  # 与原始样本一样的y坐标
    ax5.legend()
    ax5.set_ylim([min(X_data[sample_idx].T[1]), max(X_data[sample_idx].T[1])])  # 设置y轴刻度

    # 保存图像为jpg格式
    plt.savefig('attack_result/'+method_name+' perturbation.jpg')

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

att_file= 'attResults/fgsm_advx/fgsm_advxPart' + str(6) + '.h5'

original_file = 'datas/testdatas/part' + str(6) + '.h5'

method_name = 'fgsm'

plot_picture(original_file,att_file,method_name)

