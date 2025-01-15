import argparse

import torch.utils.data.distributed
import torchvision.transforms as transforms
from captum.attr import GuidedGradCam, GradientShap, Lime, Occlusion, IntegratedGradients, LRP
from captum.metrics import infidelity, sensitivity_max
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
import numpy as np
from skimage.segmentation import slic

from torch.autograd import Variable
from PIL import Image
from torch.nn import CosineSimilarity
import json


# 均方差计算
def mse_diff(tensor1, tensor2):
    diff_sq = torch.pow(tensor1 - tensor2, 2)
    squared_diff = torch.sum(diff_sq)
    return squared_diff


# 绝对值差计算
def abs_diff(tensor1, tensor2):
    diff = torch.abs(tensor1 - tensor2)
    absolute_diff = torch.sum(diff)
    return absolute_diff


# 获取consistency指标值
def get_consistency():
    mask = attribution.clamp(min=threshold)
    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1
    mask = mask.to(torch.float32)
    mask_input = mask * input

    mask_out = model(mask_input)
    ccd = abs_diff(mask_out[0][pred.data.item()], score[0])
    return ccd


def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(DEVICE)
    return noise, inputs - noise


# 获取infidelity指标值
def get_infidelity():
    # Computes infidelity score for saliency maps
    infid = infidelity(model, perturb_fn, input, attribution, target=pred.data.item())

    return infid


# 获取sensitivity指标值
def get_sensitivity():
    cs = CosineSimilarity()  # 取值为-1,1,越大相似度越高，越小相似度越低
    sim = cs.forward(x1=attribution.view(attribution.shape[0], -1),
                     x2=anti_attribution.view(anti_attribution.shape[0], -1))
    return (sim + 1) / 2


# 获取stability指标值
def get_stability():
    sens = sensitivity_process
    return sens


# 获取validity指标值
def get_validity():
    mask = attribution.clamp(max=threshold)
    mask[mask >= threshold] = 0
    mask[mask < threshold] = 1
    mask = mask.to(torch.float32)
    mask_input = mask * input
    mask_out = model(mask_input)
    val_ig = abs_diff(mask_out[0][pred.data.item()], score[0])
    return val_ig


def explain_fn(explain, **kwargs):
    return explain.attribute(input, **kwargs)


if __name__ == "__main__":
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrainedModel.pth")
    parser.add_argument("--input_dir", type=str, default="data/test/DDT/989.jpg")
    parser.add_argument("--method_name", type=str, default="GuidedGradCam")
    parser.add_argument("--class_name", type=list, default=['DDT', 'HC', 'HM', 'QT', 'QZJ', 'YC'])
    parser.add_argument("--image_name", type=str, default="IntegratedGradients")

    # ig方法所需参数
    parser.add_argument("--n_steps", type=int, default=50)

    # occlusion方法所需参数
    parser.add_argument("--sliding_window_shapes", type=tuple, default=(3, 25, 25))
    parser.add_argument("--strides", type=tuple, default=(3, 20, 20))
    parser.add_argument("--baselines", type=int, default=0)

    # GuidedGradCam方法所需参数
    # parser.add_argument("--layer_index", type=int, default=4)

    # Lime方法所需参数
    parser.add_argument("--lime_n_samples", type=int, default=8000)

    # GradientShap方法所需参数
    parser.add_argument("--stdevs", type=float, default=0.0001)
    parser.add_argument("--gs_n_samples", type=int, default=50)

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图片处理
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_tmp = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # 加载模型
    model = torch.load(args.model_path)
    model.eval()
    model.to(DEVICE)
    # 加载图片，获取预测结果
    path = args.input_dir
    img = Image.open(path)
    ori_img = transform_tmp(img)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    input = img
    out = model(img)
    # Predict
    score, pred = torch.max(out.data, 1)
    min_val, min_idx = torch.min(out, dim=1)

    # 根据方法名,获取attribution
    attribution = None
    anti_attribution = None
    attribute_method = None
    sensitivity_process = None

    if args.method_name == "IntegratedGradients":
        # ig
        ig = IntegratedGradients(model)
        attribution = ig.attribute(input, target=pred.data.item(), n_steps=args.n_steps)  # n_steps：近似法所用的步数
        anti_attribution = ig.attribute(input, target=min_idx.data.item(), n_steps=args.n_steps)  # n_steps：近似法所用的步数
        sensitivity_process = sensitivity_max(ig.attribute, input, target=pred.data.item(), n_steps=50)

    elif args.method_name == "Occlusion":
        # occlusion
        occlusion = Occlusion(model)
        attribution = occlusion.attribute(input,
                                          strides=args.sliding_window_shapes,  # 遮挡块的步长
                                          target=pred.data.item(),
                                          sliding_window_shapes=args.sliding_window_shapes,  # 遮挡块的大小
                                          baselines=args.baselines)  # 遮挡的像素块的像素值，也可以是一个张量
        anti_attribution = occlusion.attribute(input,
                                               strides=args.sliding_window_shapes,  # 遮挡块的步长
                                               target=min_idx.data.item(),
                                               sliding_window_shapes=args.sliding_window_shapes,  # 遮挡块的大小
                                               baselines=args.baselines)  # 遮挡的像素块的像素值，也可以是一个张量
        sensitivity_process = sensitivity_max(occlusion.attribute, input,
                                              strides=args.sliding_window_shapes,  # 遮挡块的步长
                                              target=pred.data.item(),
                                              sliding_window_shapes=args.sliding_window_shapes,  # 遮挡块的大小
                                              baselines=args.baselines)
    elif args.method_name == "GuidedGradCam":
        layer = model.layer4
        guided_gc = GuidedGradCam(model, layer=layer)
        attribution = guided_gc.attribute(input, target=pred.data.item())
        anti_attribution = guided_gc.attribute(input, target=min_idx.data.item())
        sensitivity_process = sensitivity_max(guided_gc.attribute, input, target=pred.data.item())

    elif args.method_name == "Lime":
        test = input.permute([0, 2, 3, 1]).squeeze(0).cpu().numpy()
        segments = slic(test, n_segments=100, sigma=5)
        segments = torch.tensor(segments).to(DEVICE)
        lime = Lime(model)
        attribution = lime.attribute(input, target=pred.data.item(), n_samples=args.lime_n_samples,
                                     feature_mask=segments)
        anti_attribution = lime.attribute(input, target=min_idx.data.item(), n_samples=args.lime_n_samples,
                                          feature_mask=segments)
        sensitivity_process = sensitivity_max(lime.attribute, input, target=pred.data.item(),
                                              n_samples=args.lime_n_samples,
                                              feature_mask=segments)
    elif args.method_name == "GradientShap":
        gradient_shap = GradientShap(model)
        # 设置 baseline distribution
        rand_img_dist = torch.cat([input * 0, input * 1])
        # 获得输入图像每个像素的 GradientShap 值
        attribution = gradient_shap.attribute(input,
                                              n_samples=args.gs_n_samples,
                                              # 输入batch中每个样本随机生成的样本数。随机样本是通过在每个样本中加入高斯随机噪声来生成的。默认值:5(如果未提供n_samples)。
                                              stdevs=args.stdevs,
                                              # 高斯噪声的标准差。默认值:0.0(如果未提供stdevs)。
                                              baselines=rand_img_dist,
                                              # 用于计算期望的基线分布。默认值:None(如果未提供baselines)。
                                              target=pred.data.item())
        anti_attribution = gradient_shap.attribute(input,
                                                   n_samples=args.gs_n_samples,
                                                   # 输入batch中每个样本随机生成的样本数。随机样本是通过在每个样本中加入高斯随机噪声来生成的。默认值:5(如果未提供n_samples)。
                                                   stdevs=args.stdevs,
                                                   # 高斯噪声的标准差。默认值:0.0(如果未提供stdevs)。
                                                   baselines=rand_img_dist,
                                                   # 用于计算期望的基线分布。默认值:None(如果未提供baselines)。
                                                   target=min_idx.data.item())
        sensitivity_process = sensitivity_max(gradient_shap.attribute, input,
                                              n_samples=args.gs_n_samples,
                                              # 输入batch中每个样本随机生成的样本数。随机样本是通过在每个样本中加入高斯随机噪声来生成的。默认值:5(如果未提供n_samples)。
                                              stdevs=args.stdevs,
                                              # 高斯噪声的标准差。默认值:0.0(如果未提供stdevs)。
                                              baselines=rand_img_dist,
                                              # 用于计算期望的基线分布。默认值:None(如果未提供baselines)。
                                              target=pred.data.item())
    else:
        # 报错方法名未知，退出程序
        print("method name error!")
        exit(0)

    # 保存结果图片
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#2e752e'), (0.5, '#ffaf0f'), (1, '#ca0019')], N=256)

    result_img = viz.visualize_image_attr_multiple(
        np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(ori_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        ['Original Image predict:{},score:{:.4f}'.format(args.class_name[pred.data.item()], score.data.item()),
         args.method_name + ": Heat map"],
        cmap=default_cmap,
        show_colorbar=True)

    print("asd")
    result_img[0].savefig("D:\suanfa\ship_classfication_explain/output2/" + args.image_name + ".jpg")

    # 计算各指标
    max_val = torch.max(attribution)
    threshold = 0 * max_val.item()
    # infidelity
    infidelity_score = get_infidelity()
    # sensitivity
    sensitivity = get_sensitivity()
    # stability
    stability = get_stability()
    # validity
    validity = get_validity()
    # consistency
    consistency = get_consistency()
    # 将数据存储为json文件

    data = {
        "infidelity": infidelity_score.data.item(),
        "sensitivity": sensitivity.data.item(),
        "stability": stability.data.item(),
        "validity": validity.data.item(),
        "consistency": consistency.data.item()
    }
    print(data)
    with open("D:\suanfa\ship_classfication_explain/output2/" + args.image_name + ".json", "w") as f:
        json.dump(data, f)
