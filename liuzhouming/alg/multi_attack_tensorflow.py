import json
import os
import sys
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from eagerpy.tensor.tensorflow import TensorFlowTensor
from foolbox import TensorFlowModel, accuracy
from Data_Utils.CustomTfDataset import CustomTfDataset
from Data_Utils.CustomTfDataset import load_model_h5, load_data_tf
from Data_Utils.Download_minio import download_model_minio, \
    download_dataset_minio
from foolbox_common import createMethodObj, parseCmdArgument, method_name_map
from common_config import service, access_key, secret_key, save_path, get_result_file_path


# 计算平均失真度, ssim , psnr
def calculate_avg_norm_distortion_factor(raw_images, adv_images, success):
    #  type eagerpy.tensor.tensorflow.TensorFlowTensor
    raw_select_images = raw_images[success]
    adv_select_images = adv_images[success]

    if raw_select_images.shape[0] == 0:
        return [0, 0, 0], 0, 0, [0, 0, 0]
    # (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
    linf_avg_epsion = round(np.mean((raw_select_images - adv_select_images).norms.linf(axis=(1, 2, 3)).numpy()).item(),
                            5)
    l1_avg_epsion = round(np.mean((raw_select_images - adv_select_images).norms.l1(axis=(1, 2, 3)).numpy()).item(), 5)
    l2_avg_epsion = round(np.mean((raw_select_images - adv_select_images).norms.l2(axis=(1, 2, 3)).numpy()).item(), 5)

    linf_distortion_factor = round(np.mean(
        (raw_select_images - adv_select_images).norms.linf(axis=(1, 2, 3)).numpy()
        / adv_select_images.norms.linf(axis=(1, 2, 3)).numpy()).item(), 5)

    l1_distortion_factor = round(np.mean(
        (raw_select_images - adv_select_images).norms.l1(axis=(1, 2, 3)).numpy()
        / adv_select_images.norms.l1(axis=(1, 2, 3)).numpy()).item(), 5)

    l2_distortion_factor = round(np.mean(
        (raw_select_images - adv_select_images).norms.l2(axis=(1, 2, 3)).numpy()
        / adv_select_images.norms.l2(axis=(1, 2, 3)).numpy()).item(), 5)

    # print(type(raw_images), raw_images.shape, isinstance(raw_images, tf.Tensor))
    # print(type(adv_images), adv_images.shape, isinstance(adv_images, tf.Tensor))
    # print(type(raw_images), isinstance(raw_images.raw, tf.Tensor))

    psnr_item = round(float(tf.reduce_mean(tf.image.psnr(raw_select_images.raw, adv_select_images.raw, 1.0))), 5)
    ssim_item = round(float(tf.reduce_mean(tf.image.ssim(raw_select_images.raw, adv_select_images.raw, 1.0))), 5)
    return [l1_distortion_factor, l2_distortion_factor, linf_distortion_factor], \
        ssim_item, psnr_item, \
        [l1_avg_epsion, l2_avg_epsion, linf_avg_epsion]


if __name__ == "__main__":

    program_start = time.time()

    args = parseCmdArgument()  # 解析命令行参数
    device = args.device
    batch_size = args.batch_size
    model_url = args.model_url
    dataset_url = args.dataset_url
    configFilePath = args.configFilePath
    debug = args.debug

    # resultFilePath
    resultFilePath = get_result_file_path(configFilePath)

    print('resultFilePath: ', resultFilePath)

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

    # 下载模型和数据集
    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)
    download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
                           save_path=save_path)

    if device.lower() != 'cpu' and tf.test.is_gpu_available():
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    # load model
    model = load_model_h5(model_url, save_path)
    # custom tf dataset
    custom_tf_dataset = CustomTfDataset(dataset_url, save_path, None)
    # images, labels
    images, labels = custom_tf_dataset[:len(custom_tf_dataset)]  # 所有数据集加载，需要优化
    test_iter = load_data_tf((images, labels), batch_size)  # 加载batch-size个

    for X, y in test_iter:
        images, labels = X, y
        if device == '/GPU:0':
            images = images.gpu()
            labels = labels.gpu()
        break

    # 加载模型和数据集
    # instantiate a model (could also be a TensorFlow or JAX model)
    # model = tf.keras.applications.ResNet50(weights="imagenet")
    # print("keras: ", type(model))

    # pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    # fmodel = TensorFlowModel(model, bounds=(0, 1))
    fmodel = TensorFlowModel(model, bounds=(0, 1), device=device)
    # fmodel = fmodel.transform_bounds((0, 1))
    if fmodel.data_format == 'channels_first':
        images = tf.transpose(images, (0, 3, 1, 2))  # channel first

    # print("fmodel: ", type(fmodel))
    #
    # # get data and test the model
    # # wrapping the tensors with ep.astensors is optional, but it allows
    # # us to work with EagerPy tensors in the following
    # # images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=10))
    images, labels = TensorFlowTensor(images), TensorFlowTensor(labels)

    clean_acc = accuracy(fmodel, images, labels)
    result["clean_acc"] = round(clean_acc, 5)  # clean_acc 正确率
    result["attacks"] = []

    # 加载模型和数据集
    # 攻击方法
    # attack methods
    attack_objs = [
        createMethodObj(config) for config in configList
    ]

    # apply the attack
    for i, attack in enumerate(attack_objs):
        if attack is None:
            continue
        # try:
        epsilon_type, fa_attack_obj = attack
        start = time.time()
        attack_method = dict()  # 每种方法的结果
        if fa_attack_obj is None:
            attack_method["error"] = "方法错误"
            result["attacks"].append(attack_method)
            continue

        epsilon = configList[i].get("epsilon", 0.05)

        # apply attack
        raw, clipped, success = fa_attack_obj(fmodel, images, labels, epsilons=epsilon)
        # print(type(success), success.shape)
        # # 计算失真率
        attack_method["distortion_factor"], attack_method["ssim"], \
            attack_method["psnr"], attack_method["avg_epsilon"] = calculate_avg_norm_distortion_factor(images,
                                                                                                       clipped,
                                                                                                       success)
        attack_method["epsilon"] = epsilon
        attack_method["epsilon_type"] = epsilon_type

        method_key_name = configList[i].get("method_name")
        attack_method["methodName"] = method_name_map.get(method_key_name, method_key_name)
        attack_method["success_attack_rate"] = round(float(tf.reduce_mean(tf.cast(success.raw, dtype=tf.float32),
                                                                          axis=-1)), 5)
        end = time.time()
        attack_method["cost"] = round((end - start), 5)
        result["attacks"].append(attack_method)

        # except Exception as e:
        #     print(e)

    end_time = time.time()
    result["cost"] = round(end_time - begin_time, 5)

    print('start write result to file: ', resultFilePath)

    print(json.dumps(result))
    # 评测结果记录 到文件
    with open(resultFilePath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(result))

    print('total cost cost: ', time.time() - program_start)
    print('resultFilePath: ', resultFilePath)
