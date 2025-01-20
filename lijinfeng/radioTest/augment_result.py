import h5py
import numpy as np
import os
import random
import json
import argparse
from keras.optimizers import adam_v2
from radioAttack.Models.resNet import Net


def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
    return X, Y, Z


# 定义数据增强方法
def add_gaussian_noise(x, mean, std_dev):
    """添加高斯白噪声"""
    noise = np.random.normal(mean, std_dev, x.shape)
    return x + noise

'''
def change_playback_speed(x, speed_factor):
    """改变播放速度"""
    num_samples = x.shape[1]
    num_channels = 2  # 由于每个样本有两个通道（IQ 信号），所以通道数为 2
    
    # 计算变速后的采样点数
    resampled_samples = int(num_samples / speed_factor)
    
    # 创建空白的输出数组
    interpolated_x = np.zeros((x.shape[0], resampled_samples, num_channels), dtype=x.dtype)

    # 对每条数据的每个通道进行重新采样
    for i in range(x.shape[0]):
        for channel in range(num_channels):
            # 原始数据
            original_signal = x[i, :, channel]
            
            # 生成变速后的时间轴
            resampled_time = np.arange(0, num_samples, speed_factor)[:resampled_samples]
            
            # 进行线性插值
            interpolated_x[i, :, channel] = np.interp(resampled_time, np.arange(num_samples), original_signal)
    
    return interpolated_x
'''

def time_shift(x, shift):
    """信号时移"""
    shifted_x = np.roll(x, shift)
    if shift < 0:
        shifted_x[shift:] = 0
    else:
        shifted_x[:shift] = 0
    return shifted_x

def blank_signal(x, blank_duration):
    """信号空白替换"""
    blank_samples = int(blank_duration * x.shape[0])
    x[:blank_samples] = 0
    return x

def parseCmdArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='radioAttack/Models/ResNet_Model.h5',
                        help='model path')

    parser.add_argument('--data_path', type=str,
                        default='datas/testdatas',
                        help='data path')

    parser.add_argument('--methodList', type=str, nargs='+', default=["gaussian_noise",
     "time_shift", "blank_signal"], help='method namelist')
    
    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parseCmdArgument()  # 解析命令行参数

    model_path = args.model_path
    data_folder = args.data_path
    augmentation_methods = args.methodList

    # 读取分割好的随机文件的数据
    num_files = 24

    # 创建保存增强数据的文件夹
    augmented_folders = ['gaussian_noise', 'time_shift', 'blank_signal']
    for folder in augmented_folders:
        os.makedirs(folder, exist_ok=True)

    # 创建结果字典
    result_dict = {'gaussian_noise': {'change_count': 0, 'total_count': 0},
                   'time_shift': {'change_count': 0, 'total_count': 0},
                   'blank_signal': {'change_count': 0, 'total_count': 0}}
    
    resulet_changes = {}

    # 定义预测集
    classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK']
    num_classes = len(classes)
    
    # 读取模型

    part_file_name = 'part0.h5'
    datafile = os.path.join(data_folder, part_file_name)

    f = h5py.File(datafile, 'r')
    datax = f['X'][:]
    f.close()
    print(datax.shape)
    in_shape = datax.shape
    model = Net(in_shape, len(classes))
    adam = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    model.load_weights(model_path)

    # 选择一个随机文件进行数据增强操作
    random_file_index = random.randint(0, num_files-1)
    datafile = os.path.join(data_folder, 'part' + str(random_file_index) + '.h5')
    X, Y, Z = load_hdf5(datafile)

    # 创建保存结果的文件夹
    result_folder = 'augment_result'
    os.makedirs(result_folder, exist_ok=True)

    # 创建保存增强数据的文件夹
    save_folder = os.path.join('augmented_data', 'part' + str(random_file_index))
    os.makedirs(save_folder, exist_ok=True)
    original_folder = os.path.join('augmented_data', 'original')
    os.makedirs(original_folder, exist_ok=True)

    # 预测原始样本类型
    original_predictions = model.predict(X)
    original_labels = np.argmax(original_predictions, axis=1)
    original_label_counts = np.bincount(original_labels, minlength=num_classes)
    

    for idx in range(X.shape[0]):
        # 获取选择的数据
        x = X[idx]
        y = Y[idx]
        z = Z[idx]

        # 数据增强方法的参数
        noise_mean = 0
        noise_std_dev = 0.1
        speed_factor = 1.1
        time_shift_amount = int(0.5 * x.shape[0])
        blank_duration = 0.4

        # 保存原始数据
        save_file = h5py.File(os.path.join(original_folder, f'original_data_{random_file_index}_{idx}.hdf5'), 'w')
        save_file.create_dataset('X_data', data=x)
        save_file.create_dataset('Y_data', data=y)
        save_file.create_dataset('Z_data', data=z)
        save_file.close()

        # 应用数据增强方法并保存增强后的数据
        for j, folder in enumerate(augmentation_methods):
            augmented_x = np.copy(x)
            if folder == 'gaussian_noise':
                augmented_x = add_gaussian_noise(augmented_x, noise_mean, noise_std_dev)
            elif folder == 'time_shift':
                augmented_x = time_shift(augmented_x, -time_shift_amount)
            elif folder == 'blank_signal':
                augmented_x = blank_signal(augmented_x, blank_duration)

            original_label = np.argmax(original_predictions[idx])

            augmented_predictions = model.predict(np.expand_dims(augmented_x, axis=0))
            augmented_label = np.argmax(augmented_predictions)

            # 判断是否改变结果
            if original_label != augmented_label:
                result_entry = result_dict[folder]
                change_count = result_entry['change_count']
                result_entry['change_count'] = change_count + 1

            result_entry = result_dict[folder]
            total_count = result_entry['total_count']
            result_entry['total_count'] = total_count + 1

            save_file = h5py.File(os.path.join(folder, f'augmented_data_{random_file_index}_{idx}.hdf5'), 'w')
            save_file.create_dataset('X_data', data=augmented_x)
            save_file.create_dataset('Y_data', data=y)
            save_file.create_dataset('Z_data', data=z)
            save_file.close()

        '''
        elif folder == 'playback_speed':
            augmented_x = change_playback_speed(augmented_x, speed_factor)
        '''

    for key in result_dict.keys():
        change_count = result_dict[key]['change_count']
        total_count = result_dict[key]['total_count']
        result_dict[key] = (change_count / total_count) * 100

    json_path = os.path.join(result_folder, 'results.json')
    with open(json_path, 'w') as json_file:
        json.dump(result_dict, json_file)
