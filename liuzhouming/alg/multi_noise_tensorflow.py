import argparse
import json
import random
import sys
import time
from math import fabs, sin, radians, cos
import os

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from foolbox import TensorFlowModel, accuracy

from Data_Utils.Download_minio import *
from Data_Utils.CustomTfDataset import CustomTfDataset
from Data_Utils.CustomTfDataset import load_model_h5, load_data_tf
from common_config import save_path, service, access_key, secret_key, get_result_file_path
from noise import gauss_noise, salt_noise, possion_noise, speckle_noise
from rain import get_noise, rain_blur, add_rain
from noise_common import noise_method_map

from sklearn.metrics import precision_score, \
    recall_score, f1_score, accuracy_score


# 随机旋转
class Random_Rotate(object):
    def __init__(self, degree=None):
        self.degree = degree
        if self.degree is None:
            self.degree = random.randint(0, 360)  # 取随机值

    def __call__(self, images):
        # images 为numpy(b, h, w, c)
        images = images.astype(np.uint8)
        new_images = []
        for img in images:
            height, width = img.shape[:2]
            M = cv2.getRotationMatrix2D((width / 2, height / 2), self.degree, 1)
            heightNew = int(width * fabs(sin(radians(self.degree))) + height * fabs(cos(radians(self.degree))))
            widthNew = int(height * fabs(sin(radians(self.degree))) + width * fabs(cos(radians(self.degree))))

            M[0, 2] += (widthNew - width) / 2
            M[1, 2] += (heightNew - height) / 2

            im_rotate = cv2.warpAffine(img, M, (widthNew, heightNew), borderValue=(255, 255, 255))
            im_rotate = cv2.resize(im_rotate, (height, width))
            new_images.append(im_rotate)
        return np.array(new_images)


# 模糊处理
class Blur(object):
    def __init__(self, blurType, kernel, k=None):
        self.blurType = blurType
        self.kernel = kernel
        self.k = k

    def __call__(self, images):
        images = images.astype(np.uint8)
        new_images = []
        for img in images:
            if self.blurType == "blur":
                new_image = cv2.blur(img, self.kernel)
            elif self.blurType == "guassianBlur":
                new_image = cv2.GaussianBlur(img, self.kernel, 0, 0)
            elif self.blurType == "medianBlur":
                new_image = cv2.medianBlur(img, self.k)
            new_images.append(new_image)
        return np.array(new_images)


# 随机裁剪
class Random_Cut(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):
        # h、w为想要截取的图片大小
        '''
        机器学习中随机产生负样本的
        '''
        # 随机产生x,y   此为像素内范围产生
        y = random.randint(1, images[0].shape[0])
        x = random.randint(1, images[0].shape[1])
        images = images.astype(np.uint8)
        new_images = []
        for img in images:
            # 随机截图
            shape = img.shape
            if self.h > shape[0] or self.w > shape[1]:
                # print("h or w is too big for this image!")
                # print("The image‘s shape is "+str(shape[0])+"*"+str(shape[1])+"*"+str(shape[2]))
                return
            # y_max 和 x_max保证裁剪范围不超过图片的范围
            y_max = y + self.h
            x_max = x + self.w
            if y + self.h > shape[0]:
                y_max = shape[0]
                y = y_max - self.h
            if x + self.w > shape[1]:
                x_max = shape[1]
                x = x_max - self.w
            cropImg = img[(y):(y_max), (x):(x_max)]
            cropImg = cv2.resize(cropImg, (shape[0], shape[1]))
            new_images.append(cropImg)
        return np.array(new_images)


# 雨天
class Rain_Add(object):
    def __init__(self, value, length, angle, w):
        self.value = value  # 雨滴数量
        self.length = length  # 控制雨水水痕长度
        self.angle = angle  # 控制雨下落角度
        self.w = w  # 控制雨点粗细程度

    def __call__(self, images):
        images = images.astype(np.uint8)
        new_images = []
        for img in images:
            # 设置value大小控制雨滴数量，length大小控制雨水水痕长度，angle大小来控制雨下落的角度，w来控制雨点粗细程度。
            noise = get_noise(img, value=self.value)
            rain = rain_blur(noise, length=self.length, angle=self.angle, w=self.w)
            # alpha_rain(rain, img, beta=0.6)  # 方法一，透明度赋值
            result = add_rain(rain, img)  # 方法二,加权后有玻璃外的效果
            new_images.append(result)
        return np.array(new_images)


class Gauss_Noise(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, images):
        images = images.astype(np.uint8)  # to ndarray
        new_images = []
        for img in images:
            new_images.append(gauss_noise(self.mean, self.sigma, img))
        return np.array(new_images)


class Salt_Noise(object):
    def __init__(self, s_vs_p, amount):
        self.s_vs_p = s_vs_p
        self.amount = amount

    def __call__(self, images):
        images = images.astype(np.uint8)  # to ndarray
        new_images = []
        for img in images:
            new_images.append(salt_noise(self.s_vs_p, self.amount, img))
        return np.array(new_images)


class Possion_Noise(object):
    def __init__(self):
        pass

    def __call__(self, images):
        images = images.astype(np.uint8)  # to ndarray
        new_images = []
        for img in images:
            new_images.append(possion_noise(img))
        return np.array(new_images)


class Speckle_Noise(object):
    def __init__(self):
        pass

    def __call__(self, images):
        images = images.astype(np.uint8)  # to ndarray
        new_images = []
        for img in images:
            new_images.append(speckle_noise(img))
        return np.array(new_images)


# if __name__ == "__main__":
#     images = torch.empty((4, 200, 300, 3), dtype=torch.float32).uniform_(0, 255)
#     print(type(images), images.shape, images.dtype)
#     # random_rotate = Random_Rotate()
#     # images = random_rotate(images)
#     # random_cut = Random_Cut(150, 250)
#     # images = random_cut(images)
#     rain_add = Rain_Add(500, 50, -30, 3)
#     rain_add(images)
#     print(type(images), images.shape, images.dtype)
#     pass


def createMethodObj(config_dict):
    method_name = config_dict.get("method_name")
    if method_name == "random_rotate":
        return Random_Rotate(config_dict.get("degree", None))
    elif method_name == "random_cut":
        return Random_Cut(config_dict.get("h"), config_dict.get("w"))
    elif method_name == "rain_add":
        return Rain_Add(config_dict.get("value"), config_dict.get("length"),
                        config_dict.get("angle"), config_dict.get("w"))
    elif method_name == "gauss_noise":
        return Gauss_Noise(config_dict.get("mean"), config_dict.get("sigma"))
    elif method_name == "salt_noise":
        return Salt_Noise(config_dict.get("s_vs_p"), config_dict.get("amount"))
    elif method_name == "possion_noise":
        return Possion_Noise()
    elif method_name == "speckle_noise":
        return Speckle_Noise()
    else:
        return None


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for Attack')
    parser.add_argument('--batch_size', type=int, default=500, help='batch_size, the size of sample to use')
    parser.add_argument('--model_url', type=str,
                        default='http://10.105.240.103:9000/model/20230401/213323-mlp_fashionmnist.h5',
                        help='http://localhost:9000/model/20230224/204945-mnist_mlp.pt')
    parser.add_argument('--dataset_url', type=str,
                        default='http://localhost:9000/dataset/20230331/11353-fashionMNIST.zip',
                        help='http://localhost:9000/dataset/20230224/11739-dataset.zip')
    parser.add_argument('--configFilePath', type=str,
                        default=r'E:\mycode\micro-test-ai\alg\config\425fc003-7009-44ff-aca9-b8f6ed2a1d77.json',
                        help='method config file path')

    # 解析参数
    args = parser.parse_args()

    # device = "/" + str(args.device).upper()

    batch_size = args.batch_size
    device = args.device
    model_url = args.model_url
    dataset_url = args.dataset_url
    configFilePath = args.configFilePath

    if device.lower() != 'cpu' and tf.test.is_gpu_available():
        device = '/GPU:0'
    else:
        device = '/CPU:0'

    # resultFilePath
    resultFilePath = get_result_file_path(configFilePath)

    print(resultFilePath)

    configList = []
    if configFilePath != "":
        with open(configFilePath, 'r', encoding='utf-8') as f:
            configList.extend(json.load(f))
        # 删除文件
        # os.remove(configFilePath)
        # print(configFilePath)
    else:
        sys.exit(0)

    begin_time = time.time()
    result = dict()  # result

    result["batch_size"] = batch_size
    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)
    model = load_model_h5(model_url, save_path=save_path)

    # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    fmodel = TensorFlowModel(model, bounds=(0, 1), device=device)
    # load datasets
    # load images
    # images, labels = samples(fmodel, dataset='imagenet', batchsize=batch_size)
    download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
                           save_path=save_path)

    # custom tf dataset
    custom_tf_dataset = CustomTfDataset(dataset_url, save_path, None)

    images, labels = custom_tf_dataset[:len(custom_tf_dataset)]  # 所有数据集加载，需要优化
    test_iter = load_data_tf((images, labels), batch_size)  # 加载batchsize个
    for X, y in test_iter:
        images, labels = X, y
        # if device == '/GPU:0':
        #     images = images.gpu()
        #     labels = labels.gpu()
        break

    # fmodel = PyTorchModel(model, bounds=(0, 1), device=device)
    # images, labels = load_dataset(dataset="fashionmnist", data_size=30,trans=None)
    images = images.numpy()

    attacks = [
        createMethodObj(config) for config in configList
    ]

    result["noiseMethodList"] = []
    new_images = None

    for i, attack in enumerate(attacks):
        start = time.time()
        noise_method = dict()
        if attack is None:
            noise_method["error"] = "方法错误"
            result["noiseMethodList"].append(noise_method)
            continue

        # attack
        # try:
        noise_method_key_name = configList[i].get("method_name", "")
        noise_method["methodName"] = noise_method_map.get(noise_method_key_name, noise_method_key_name)

        new_images = tf.constant(attack(images), dtype=tf.float32)  # device
        # if device == '/GPU:0':
        #     new_images = new_images.gpu()
        if len(new_images.shape) == 3:
            new_images = tf.expand_dims(new_images, -1)  # 拓展维度

        if fmodel.data_format == 'channels_first':
            new_images = tf.transpose(new_images, (0, 3, 1, 2))  # channel first

        change_clean_acc = accuracy(fmodel, new_images, labels)  # 变换后的准确率

        noise_method["change_clean_acc"] = round(change_clean_acc, 2)  # clean_acc 正确率

        # 预测值
        output = model.predict(new_images)  # 预测
        preds = tf.argmax(output, -1)  # 预测值

        f1_score_result = f1_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                   average='macro')  # 也可以指定micro模式
        noise_method["F1_SCORE"] = f1_score_result

        precision_score_result = precision_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                                 average='macro')
        noise_method["PRECISION"] = precision_score_result

        recall_score_result = recall_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                           average='macro')  # 也可以指定micro模
        noise_method["RECALL"] = recall_score_result

        end = time.time()
        noise_method["cost"] = round((end - start), 2)
        result["noiseMethodList"].append(noise_method)

        # except Exception as e:
        #     print(noise_method.get("method_name"))
        #     print(e)
        #     noise_method["error"] = "error"
        #     result["noiseMethodList"].append(noise_method)
        #     continue

    end_time = time.time()
    # print("images.shape: ", images.shape)
    images = tf.constant(images, dtype=tf.float32)
    # if device == '/GPU:0':
    #     images = images.gpu()
    if fmodel.data_format == 'channels_first':
        images = tf.transpose(images, (0, 3, 1, 2))  # channel first


    # 预测值
    output = model.predict(images)  # 预测
    preds = tf.argmax(output, -1)  # 预测值

    result["clean_acc"] = accuracy_score(y_true=labels.numpy(),
                                         y_pred=preds.numpy())  # clean_acc 正确率
    f1_score_result = f1_score(y_true=labels.numpy(),
                               y_pred=preds.numpy(),
                               average='macro')  # 也可以指定micro模式
    result["F1_SCORE"] = f1_score_result

    precision = precision_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                average='macro')
    result["PRECISION"] = precision

    recall_score = recall_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                average='macro')  # 也可以指定micro模
    result["RECALL"] = recall_score

    result["cost"] = round(end_time - begin_time, 2)
    print(json.dumps(result))  # json格式数据
    # 评测结果记录 到文件
    with open(resultFilePath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))
    print(resultFilePath)
