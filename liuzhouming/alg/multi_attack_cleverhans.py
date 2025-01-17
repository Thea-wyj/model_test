import argparse
import json
import sys

import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from common_config import save_path, service, access_key, secret_key

from Data_Utils.Download_minio import download_model_minio, \
    download_dataset_minio
from Data_Utils.ModelLoader import load_model_pt
from Data_Utils.MyDataset import MyDataset


# str two bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# 计算平均失真度
def calculate_avg_norm_distortion_factor(raw_images, adv_images, is_success, eps_type='L2'):
    print(type(raw_images), type(adv_images), type(is_success))
    distortion_factor = 0.0
    n = 0
    for index in range(len(adv_images)):
        if is_success[index]:
            n = n + 1
            if eps_type == "L2":
                distortion_factor = distortion_factor + torch.norm(raw_images[index]
                                                                   - adv_images[index],
                                                                   2).item() \
                                    / torch.norm(raw_images[index], 2).item()
            elif eps_type == 'L1':
                distortion_factor = distortion_factor + torch.norm(raw_images[index] -
                                                                   adv_images[index],
                                                                   1).item() \
                                    / torch.norm(raw_images[index], 1).item()
            elif eps_type == 'Linf':
                distortion_factor = distortion_factor + torch.norm(raw_images[index] -
                                                                   adv_images[index],
                                                                   float('inf')).item() \
                                    / torch.norm(raw_images[index], float('inf')).item()
    return distortion_factor / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for Attack')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size, the size of sample to use generate '
                                                                   'adversarial examples')
    parser.add_argument('--model_url', type=str, default='', help='model url')
    parser.add_argument('--dataset_url', type=str, default='', help='dataset url')
    parser.add_argument('--methodList', type=str, nargs='+', default=[], help='method name list')
    parser.add_argument('--configFilePath', type=str, default='', help='method config file path')
    parser.add_argument('--debug', type=int, default=1, help='if debug 1 else 0')

    # 解析参数
    args = parser.parse_args()
    # print("args: ", args)
    device = args.device
    try:
        device = torch.device(device)
    except:
        device = torch.device('cpu')

    batch_size = args.batch_size
    model_url = args.model_url
    dataset_url = args.dataset_url
    methodList = args.methodList
    configFilePath = args.configFilePath
    debug = args.debug

    configList = []
    if configFilePath != "":
        with open(configFilePath, 'r', encoding='utf-8') as f:
            configList.extend(json.load(f))
        if debug:
            # 删除文件
            os.remove(configFilePath)
    else:
        sys.exit(0)

    begin_time = time.time()
    result = dict()  # result
    result["batch_size"] = batch_size

    download_model_minio(model_url, service = service, access_key = access_key, secret_key = secret_key, save_path=save_path)
    model = load_model_pt(model_url, save_path=save_path)
    # load datasets
    # load images
    mytransform = transforms.Compose([
        transforms.ToTensor()
    ]
    )
    download_dataset_minio(dataset_url,  service = service, access_key = access_key, secret_key = secret_key, save_path=save_path)
    test_loader = DataLoader(
        MyDataset(dataset_url,save_path=save_path, transform=mytransform),
        batch_size=batch_size,
        shuffle=True)
    images, labels = next(iter(test_loader)) # 取一次

    images = images.to(device)
    labels = labels.to(device)
    model = model.to(device)

    clean_acc = accuracy(fmodel, images, labels)
    result["clean_acc"] = round(clean_acc, 2)  # clean_acc 正确率

    result["attacks"] = []
    for i, attack in enumerate(configList):
        start = time.time()
        if attackMethod is None:
            attack_method = dict()
            attack_method["error"] = "方法错误"
            result["attacks"].append(attack_method)
            continue
        epsilon = configList[i].get("epsilon", 0.001)
        raw, clipped, success = attackMethod(fmodel, images, labels, epsilons=epsilon)
        # assert success.shape ==  len(images)  #  len(images))
        # print(success.shape)
        success_ = success.cpu().numpy()
        assert success_.dtype == np.bool_
        end = time.time()
        # print("cost %d seconds" % (end - start))
        attack_method = dict()
        attack_method["cost"] = round((end - start), 2)
        attack_method["epsilon"] = epsilon
        attack_method["method"] = str(attack)
        attack_method["methodName"] = configList[i].get("method_name")
        attack_method["success_attack_rate"] = success_.mean(axis=-1).round(2).tolist()
        attack_method["epsilon_type"] = epsilon_type
        # 计算失真率
        attack_method["distortion_factor"] = calculate_avg_norm_distortion_factor(images, clipped, success,
                                                                                  epsilon_type)
        result["attacks"].append(attack_method)

    end_time = time.time()
    result["cost"] = round(end_time - begin_time, 2)
    print(json.dumps(result))  # json格式数据



