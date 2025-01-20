# file_path = 'data_aishell/BAC009S0765W0130.wav'

import numpy as np
import os
import random
import librosa
from utils import read_wave_from_file, save_wav, tensor_to_img, get_feature, plot_spectrogram

def gaussian_white_noise_numpy(file_path, min_db=10, max_db=500):
    """
    高斯白噪声
    噪声音量db
        db = 10, 听不见
        db = 100,可以听见，很小
        db = 500,大
        人声都很清晰
    :param file_path:
    :param max_db:
    :param min_db:
    :return:
    """
    file_name = os.path.basename(file_path)
    samples, _ = read_wave_from_file(file_path)
    feature = get_feature(samples)
    #tensor_to_img(feature)
    
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    db = np.random.randint(low=min_db, high=max_db)
    noise = db * np.random.normal(0, 1, len(samples))  # 高斯分布
    samples = samples + noise
    samples = samples.astype(data_type)

    output_directory = "data_result"
    output_file_name = "gaussian_" + file_name
    out_file = os.path.join(output_directory, output_file_name)
    feature = get_feature(samples)
    save_wav(out_file, samples)
    #tensor_to_img(feature)

    return out_file


def speed_numpy(file_path, speed=None, min_speed=0.5, max_speed=1.5):
    """
    音频速度改变
    :param speed: 速度
    :param file_path: 音频文件位置
    :param max_speed: 最低拉伸倍速
    :param min_speed: 最高拉伸倍速
    :return:
    """
    file_name = os.path.basename(file_path)
    samples, frame_rate = read_wave_from_file(file_path)
    feature = get_feature(samples, feature_dim=80)
    x = feature.shape[0]
    #print('feature   ：', feature.shape)
    #plot_spectrogram(feature, 'before')

    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    if speed is None:
        speed = random.uniform(min_speed, max_speed)
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)

    output_directory = "data_result"
    output_file_name = "speed_" + file_name
    out_file = os.path.join(output_directory, output_file_name)
    save_wav(out_file, samples)
    feature = get_feature(samples, feature_dim=80)
    #print('feature   ：', feature.shape)
    #plot_spectrogram(feature, 'after')

    return out_file

def pitch_librosa(file_path, sr=16000, ratio=10):
    '''
    音调改变
    :param file_path: 音频文件位置
    :param sr: 采样率
    :param ratio: 音调变化幅度
    :return:
    '''
    file_name = os.path.basename(file_path)
    samples, frame_rate = read_wave_from_file(file_path)
    feature = get_feature(samples)
    #plot_spectrogram(feature, 'before')

    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    samples = samples.astype('float')
    ratio = random.uniform(-ratio, ratio)
    samples = librosa.effects.pitch_shift(samples, sr, n_steps=ratio)
    samples = samples.astype(data_type)

    output_directory = "data_result"
    output_file_name = "pitch_" + file_name
    out_file = os.path.join(output_directory, output_file_name)
    save_wav(out_file, samples)
    feature = get_feature(samples)
    #plot_spectrogram(feature, 'after')

    return out_file

def time_shift_numpy(file_path, max_ratio=0.05):
    """
    音频时间变化，在时间轴的±5％范围内的随机滚动。环绕式转换。
    :param file_path: 音频文件位置
    :param max_ratio:
    :return:
    """
    file_name = os.path.basename(file_path)
    samples, frame_rate = read_wave_from_file(file_path)
    feature = get_feature(file_path)
    #tensor_to_img(feature)

    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    frame_num = samples.shape[0]
    max_shifts = frame_num * max_ratio  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    samples = np.roll(samples, nb_shifts, axis=0)
    samples = samples.astype(data_type)

    output_directory = "data_result"
    output_file_name = "time_" + file_name
    out_file = os.path.join(output_directory, output_file_name)
    feature = get_feature(samples)
    save_wav(out_file, samples)
    #tensor_to_img(feature)

    return out_file


