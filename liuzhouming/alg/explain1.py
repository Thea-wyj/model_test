import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from captum.attr import GuidedGradCam, GradientShap, Lime, Occlusion, IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch.autograd import Variable

from Data_Utils.Bucket import Bucket
from Data_Utils.Download_minio import download_model_minio, \
    download_pic_minio
from Data_Utils.ModelLoader import load_model_pt
from common_config import service, access_key, secret_key, save_path, get_result_file_path

import uuid


class MyIntegratedGradients(object):
    def __init__(self):
        pass

    def __call__(self, model, input, pred, config):
        ig = IntegratedGradients(model)
        attribution_ig = ig.attribute(input, target=pred.data.item(),
                                      n_steps=config.get("n_steps", 225))  # n_steps：近似法所用的步数
        return attribution_ig


# occlusion
class MyOcclusion(object):
    def __init__(self):
        pass

    def __call__(self, model, input, pred, config):
        config_sliding_window_shapes = config.get("sliding_window_shapes")
        if config_sliding_window_shapes is None:
            config_sliding_window_shapes = (3, 40, 40)
        else:
            config_sliding_window_shapes = tuple(
                int(num) for num in config_sliding_window_shapes.strip('() ').split(','))

        config_strides = config.get("strides")
        if config_strides is None:
            config_strides = (3, 30, 30)
        else:
            config_strides = tuple(int(num) for num in config_strides.strip('() ').split(','))
        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(input,
                                               strides=config_strides,  # 遮挡块的步长
                                               target=pred.data.item(),
                                               sliding_window_shapes=config_sliding_window_shapes,  # 遮挡块的大小
                                               baselines=config.get("baselines", 0))  # 遮挡的像素块的像素值，也可以是一个张量
        return attributions_occ


# grad-cam
class MyGradCam(object):
    def __init__(self):
        pass

    def __call__(self, model, input, pred, config):
        guided_gc = GuidedGradCam(model, layer=model.layer4)
        attribution_gc = guided_gc.attribute(input, pred.data.item())
        return attribution_gc


# lime
class MyLime(object):
    def __init__(self):
        pass

    def __call__(self, model, input, pred, config):
        lime = Lime(model)
        attributions_lime = lime.attribute(input, target=pred.data.item(),
                                           n_samples=config.get("n_samples", 1000))  # 样本数需要很多
        return attributions_lime


# GradientShap
class MyGradientShap(object):
    def __init__(self):
        pass

    def __call__(self, model, input, pred, config):
        gradient_shap = GradientShap(model)
        # 设置 baseline distribution
        rand_img_dist = torch.cat([input * 0, input * 1])
        # 获得输入图像每个像素的 GradientShap 值
        attributions_gs = gradient_shap.attribute(input,
                                                  n_samples=config.get("n_samples", 50),
                                                  # 输入batch中每个样本随机生成的样本数。随机样本是通过在每个样本中加入高斯随机噪声来生成的。默认值:5(如果未提供n_samples)。
                                                  stdevs=config.get("stdevs", 0.0001),
                                                  # 加到批处理中每个输入的均值为零的高斯噪声的标准差。如果stdevs是单个浮点值，那么所有输入都使用相同的值。
                                                  # 如果它是一个元组，那么它必须具有与输入元组相同的长度。在这种情况下，stdev元组中的每个stdev值对应于输入元组中具有相同索引的输入。默认值:0.0
                                                  baselines=rand_img_dist,
                                                  # 基线定义了计算期望的起点，可提供如下:
                                                  # 单个张量，如果输入是单个张量，第一个维度等于基线分布中的样例数量。剩余的维度必须与输入张量的维度匹配，从第二个维度开始。
                                                  # 张量的元组，如果输入是张量的元组，元组中任何张量的第一个维度等于基线分布中的示例数。剩余的维度必须与从第二维度开始的相应输入张量的维度相匹配。
                                                  target=pred.data.item())
        return attributions_gs


def create(config):
    method = config.get("method_name")
    if method == "GradientShap":
        return MyGradientShap()
    elif method == "Lime":
        return MyLime()
    elif method == "GuidedGradCam":
        return MyGradCam()
    elif method == "Occlusion":
        return MyOcclusion()
    elif method == "IntegratedGradients":
        return MyIntegratedGradients()
    else:
        return None


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # print(os.environ["KMP_DUPLICATE_LIB_OK"])
    # print(os.environ["KMP_DUPLICATE_LIB_OK"])
    torch.cuda.empty_cache()
    classes = ('cat', 'dog')
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    parser.add_argument('--model_url', type=str,
                        default='http://10.105.240.103:9000/model/pretrainedModel.pth',
                        help='model url')

    parser.add_argument('--pic_url_List', type=str, nargs='+',
                        default=["./data/picture/1.jpg", "./data/picture/3.jpg"],
                        help='pic url list')

    parser.add_argument('--configFilePath', type=str,
                        default="config.json",
                        help='method config file path')

    parser.add_argument('--debug', type=int, default=1, help='debug ? true for 1 false for 0')

    # 解析参数
    args = parser.parse_args()
    DEVICE = args.device
    try:
        DEVICE = torch.device(DEVICE)
    except:
        DEVICE = torch.device('cpu')
    print(DEVICE)

    model_url = args.model_url

    pic_url_List = args.pic_url_List

    debug = args.debug

    configFilePath = args.configFilePath

    # 结果保存的path
    resultFilePath = get_result_file_path(configFilePath)

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
    methods = [
        create(config) for config in configList
    ]

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_tmp = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)
    model = load_model_pt(model_url, save_path=save_path)

    # download pic List
    pic_url_List = download_pic_minio(pic_url_List, service=service,
                                      access_key=access_key, secret_key=secret_key, save_path=save_path)

    # model = torch.load('pretrainedModel.pth')
    # model = torch.load("pretrainedModel.pth", map_location='cpu')
    model.eval()  # test
    model.to(DEVICE)
    # download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
    #                        save_path=save_path)

    # path = 'data/test/59.jpg'
    res = {}

    minio_obj = Bucket(service, access_key, secret_key)

    for path in pic_url_List:
        img = Image.open(path)
        ori_img = transform_tmp(img)
        img = transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        input = img
        out = model(img)
        # Predict
        score, pred = torch.max(out.data, 1)

        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'), (0.25, '#2e752e'), (0.5, '#ffaf0f'),
                                                          (1, '#ca0019')],
                                                         N=256)

        for i, method in enumerate(methods):
            if method is None:
                continue
            config = configList[i]
            attribution = method(model, input, pred, config)
            result_img = viz.visualize_image_attr_multiple(
                np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                ["original_image", "heat_map"],
                ["all", "positive"],
                ["Original image, The prediction is: " + classes[pred.data.item()],
                 "Heat map"],
                cmap=default_cmap,
                show_colorbar=True,
                use_pyplot=False)
            now = datetime.datetime.now()
            # 年月日
            ymd = now.strftime("%Y%m%d")
            # 时分秒
            hms = str(uuid.uuid1())  # 防止命名重复

            # 以时分秒来命名图片
            filename = f"{hms}.png"
            # if not os.path.exists(save_path+"/"+ymd):
            # 如果路径不存在的话，在本地创建路径
            os.makedirs(save_path + "/" + ymd, exist_ok=True)
            # 将图片保存到本地
            result_img[0].savefig(save_path + "/" + ymd + "/" + filename)
            minio_obj.fput_file("images", ymd + "/" + filename, save_path + "/" + ymd + "/" + filename)
            # 获取res字典中以图片路径path为key的value，这个value也是一个字典（以方法名为key，结果图片路径为value）
            res_value = res.get(path, {})
            # 对本方法的结果图片的路径添加到res_value中，通过configList获取method_name
            res_value[configList[i].get("method_name")] = ymd + "/" + filename

        # 待解释图片的路径为key
        key_path = "/".join(path.split("/")[-2:])
        res[key_path] = res_value

    print(res)
    # 评测结果记录 到文件
    with open(resultFilePath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(res))

# {'./data/picture/1.jpg': {'IntegratedGradients': '20230507/205701.png', 'Occlusion': '20230507/205702.png',
# 'GuidedGradCam': '20230507/205702.png', 'Lime': '20230507/205724.png', 'GradientShap': '20230507/205746.png'},
# './data/picture/3.jpg': {'IntegratedGradients': '20230507/205701.png', 'Occlusion': '20230507/205702.png',
# 'GuidedGradCam': '20230507/205703.png', 'Lime': '20230507/205746.png', 'GradientShap': '20230507/205747.png'}}
