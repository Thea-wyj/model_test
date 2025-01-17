import json
import os
import sys
import time

import numpy as np
import torch
from foolbox import PyTorchModel, accuracy
from torch.utils.data import DataLoader
from torchvision import transforms
from common_config import save_path, service, access_key, secret_key, get_result_file_path

from Data_Utils.Download_minio import download_model_minio, \
    download_dataset_minio
from Data_Utils.ModelLoader import load_model_pt
from Data_Utils.MyDataset import MyDataset
from skimage.metrics import structural_similarity as ssim  # 计算图片平均结构相似度
from skimage.metrics import peak_signal_noise_ratio as psnr  # 计算信噪比
from foolbox_common import parseCmdArgument, createMethodObj, method_name_map


def calculate_avg_norm_distortion_factor(raw_images, adv_images, is_success):
    """
        计算相应的指标
    """
    # 筛选出 真正的对抗样本
    select_raw_images = raw_images[is_success]
    select_adv_images = adv_images[is_success]

    # 没有对抗样本
    if select_raw_images.shape[0] == 0:
        return [0, 0, 0], 0, 0, [0, 0, 0]
    linf_avg_epsion = torch.norm(select_raw_images - select_adv_images, p=float('inf'), dim=(1, 2, 3))
    l1_avg_epsion = torch.norm(select_raw_images - select_adv_images, p=1, dim=(1, 2, 3))
    l2_avg_epsion = torch.norm(select_raw_images - select_adv_images, p=2, dim=(1, 2, 3))

    linf_distortion_factor = (linf_avg_epsion
                              / torch.norm(select_raw_images, p=float('inf'), dim=(1, 2, 3))).mean().item()
    l1_distortion_factor = (l1_avg_epsion
                            / torch.norm(select_raw_images, p=1, dim=(1, 2, 3))).mean().item()
    l2_distortion_factor = (l2_avg_epsion
                            / torch.norm(select_raw_images, p=2, dim=(1, 2, 3))).mean().item()

    psnr_item = psnr(select_raw_images.cpu().numpy(), select_adv_images.cpu().numpy(), data_range=1.0).item()

    ssim_item = 0
    # if select_raw_images.shape[0] <= 11:
    for index in range(select_raw_images.shape[0]):
        image_ndarray = np.array(raw_images[index].cpu())
        adv_ndarray = np.array(adv_images[index].cpu())
        ssim_item = ssim_item + ssim(image_ndarray, adv_ndarray, win_size=11, data_range=1.0,  channel_axis=0)
    ssim_item /= select_raw_images.shape[0]  # mean
    # else:
    #     # channel_first
    #     ssim_item = ssim(select_raw_images.cpu().numpy(), select_adv_images.cpu().numpy(),
    #                      win_size=11, data_range=1.0, channel_axis=1).item()
    return [l1_distortion_factor, l2_distortion_factor, linf_distortion_factor], ssim_item, psnr_item, \
        [l1_avg_epsion.mean().item(), l2_avg_epsion.mean().item(), linf_avg_epsion.mean().item()]


if __name__ == "__main__":
    # 解析参数
    # configList = [eval(cur_method_config) for cur_method_config in configList]
    args = parseCmdArgument()  # 解析命令行参数
    # print("args: ", args)
    device = args.device
    try:
        device = torch.device(device)
    except:
        device = torch.device('cpu')

    batch_size = args.batch_size
    # print("batch size".format(batch_size))
    model_url = args.model_url
    dataset_url = args.dataset_url
    # methodList = args.methodList
    targetedList = args.targeted
    configFilePath = args.configFilePath
    debug = args.debug

    resultFilePath = get_result_file_path(configFilePath)

    print('resultFilePath', resultFilePath)

    # configList
    configList = []
    if configFilePath != "":
        with open(configFilePath, 'r', encoding='utf-8') as f:
            configList.extend(json.load(f))
        # 删除文件
        if debug == 0:
            os.remove(configFilePath)
    else:
        sys.exit(0)

    begin_time = time.time()
    result = dict()  # result

    result["batch_size"] = batch_size

    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)
    model = load_model_pt(model_url, save_path=save_path)
    model.eval()

    # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    fmodel = PyTorchModel(model, bounds=(0, 1), device=device)

    # load datasets
    # load images
    # images, labels = samples(fmodel, dataset='imagenet', batchsize=batch_size)
    mytransform = transforms.Compose([
        transforms.ToTensor()
    ]
    )

    download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
                           save_path=save_path)
    test_loader = DataLoader(
        MyDataset(dataset_url, save_path=save_path, transform=mytransform),
        batch_size=batch_size,
        shuffle=True)

    # channel first
    images, labels = next(iter(test_loader))  # 取一次
    images = images.to(device)

    labels = labels.to(device)

    clean_acc = accuracy(fmodel, images, labels)
    result["clean_acc"] = round(clean_acc, 5)  # clean_acc 正确率
    # print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    # attack methods
    attacks = [
        createMethodObj(config) for config in configList
    ]

    result["attacks"] = []
    for i, attack in enumerate(attacks):
        if attack is None:
            continue

        try:
            epsilon_type, attackMethod = attack

            start = time.time()

            if attackMethod is None:
                attack_method = dict()
                attack_method["error"] = "方法错误"
                result["attacks"].append(attack_method)
                continue

            epsilon = configList[i].get("epsilon", 0.001)
            raw, clipped, success = attackMethod(fmodel, images, labels, epsilons=epsilon)

            # for j in range(success.shape[0]):
            #     if torch.isnan(clipped[j].any()):
            #         success[j] = False

            # print("cost %d seconds" % (end - start))
            attack_method = dict()
            attack_method["epsilon"] = epsilon
            attack_method["epsilon_type"] = epsilon_type
            method_key_name = configList[i].get("method_name")
            attack_method["methodName"] = method_name_map.get(method_key_name, method_key_name)
            attack_method["success_attack_rate"] = round(success.float().mean(axis=-1).item(), 5)

            # 计算失真率
            attack_method["distortion_factor"], attack_method["ssim"], \
            attack_method["psnr"], attack_method["avg_epsilon"] = calculate_avg_norm_distortion_factor(images, clipped, success)

            end = time.time()
            attack_method["cost"] = round((end - start), 5)
            result["attacks"].append(attack_method)

        except Exception as e:
            pass

    end_time = time.time()
    result["cost"] = round(end_time - begin_time, 5)

    print(json.dumps(result))
    # 评测结果记录 到文件
    with open(resultFilePath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result))

    print('resultFilePath', resultFilePath)
