import os
import Levenshtein
import argparse
import json
from audio_augment import gaussian_white_noise_numpy, speed_numpy, pitch_librosa, time_shift_numpy
from conv import GatedConv


def parseCmdArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='pretrained/model_cer0_19_l15_pytorch.pth',
                        help='model path')

    parser.add_argument('--data_path', type=str,
                        default='data_aishell/BAC009S0765W0130.wav',
                        help='data path')

    parser.add_argument('--methodList', type=str, nargs='+', default=["gaussian_white_noise_numpy",
     "speed_numpy", "pitch_librosa", "time_shift_numpy"], help='method namelist')
    
    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseCmdArgument()  # 解析命令行参数

    model_path = args.model_path
    audio_path = args.data_path
    
    model = GatedConv.load(os.path.join('..', model_path))
    text = model.predict(audio_path)

    filename = "result.json"

    with open(filename, "r") as file:
        data = json.load(file)

    if "gaussian_white_noise_numpy" in args.methodList:
        gaussian_path = gaussian_white_noise_numpy(audio_path)
        gaussian_text = model.predict(gaussian_path)
        #计算编辑距离
        edit_distance = Levenshtein.distance(text, gaussian_text)
        #计算改变程度
        change_percentage = edit_distance / len(text) * 100
        data["噪声添加后结果改变程度"] = change_percentage

    if "speed_numpy" in args.methodList:
        speed_path = speed_numpy(audio_path)
        speed_text = model.predict(speed_path)
        #计算编辑距离
        edit_distance = Levenshtein.distance(text, speed_text)
        #计算改变程度
        change_percentage = edit_distance / len(text) * 100
        data["音频速度变换后结果改变程度"] = change_percentage
    
    if "pitch_librosa" in args.methodList:
        pitch_path = pitch_librosa(audio_path)
        pitch_text = model.predict(pitch_path)
        #计算编辑距离
        edit_distance = Levenshtein.distance(text, pitch_text)
        #计算改变程度
        change_percentage = edit_distance / len(text) * 100
        data["音调变换后结果改变程度"] = change_percentage
    
    if "time_shift_numpy" in args.methodList:
        time_path = time_shift_numpy(audio_path)
        time_text = model.predict(time_path)
        #计算编辑距离
        edit_distance = Levenshtein.distance(text, time_text)
        #计算改变程度
        change_percentage = edit_distance / len(text) * 100
        data["音频时移后结果改变程度"] = change_percentage

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print("测评结束")

