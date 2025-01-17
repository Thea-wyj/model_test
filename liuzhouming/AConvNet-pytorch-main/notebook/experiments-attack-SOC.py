import os
import sys

import foolbox as fa
import model
import torch
import torchvision
from data import loader
from data import preprocess
from foolbox import PyTorchModel, accuracy
from skimage.metrics import peak_signal_noise_ratio as psnr  # 计算信噪比
from utils import common

sys.path.append('../src')


def load_dataset(path, is_train, name, batch_size):
    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=torchvision.transforms.Compose([
            preprocess.CenterCrop(88), torchvision.transforms.ToTensor()
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=1
    )
    return data_loader


def evaluate(_m, ds):
    num_data = 0
    corrects = 0

    _m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(ds):
        images, labels, _ = data
        predictions = _m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


config = common.load_config(os.path.join(common.project_root, 'experiments/config/AConvNet-SOC.json'))
model_name = config['model_name']
test_set = load_dataset('dataset', False, 'soc', 100)

m = model.Model(
    classes=config['num_classes'], channels=config['channels'],
)

best_path = '/home/dzk/AConvNet-pytorch-main/experiments/model/AConvNet-SOC/model-055.pth'

m.load(best_path)
m.net.eval()

print(type(test_set))

min_value, max_value = 0, 0
for i, data in enumerate(test_set):
    images, labels, _ = data
    print(images.shape, labels.shape)
    print(images.min(), images.max())
    min_value = min(min_value, images.min().float())
    max_value = max(max_value, images.max().float())
print(min_value, max_value)

fmodel = PyTorchModel(m.net, bounds=(min_value, max_value), device=m.device)

test_set = load_dataset('dataset', False, 'soc', 100)

images, labels, _ = next(iter(test_set))

images = images.to(m.device)
labels = labels.to(m.device)

clean_acc = accuracy(fmodel, images, labels)

print('clean acc is:', clean_acc)

## LinfPGD

linfPGD = fa.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=0.03333333333333333, abs_stepsize=None, steps=40,
                                                        random_start=True)

# LinfPGD
raw, clipped, success = linfPGD(fmodel, images, labels, epsilons=0.01)

print(round(success.float().mean(axis=-1).item(), 5))


# 计算相关的指标

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

    psnr_item = psnr(select_raw_images.cpu().numpy(), select_adv_images.cpu().numpy(), data_range=12)

    # ssim_item = 0
    # # if select_raw_images.shape[0] <= 11:
    # for index in range(select_raw_images.shape[0]):
    #     image_ndarray = np.array(raw_images[index].cpu())
    #     adv_ndarray = np.array(adv_images[index].cpu())
    #     ssim_item = ssim_item + ssim(image_ndarray, adv_ndarray, win_size=11, channel_axis=0, multichannel=True)
    # ssim_item /= select_raw_images.shape[0]  # mean
    # else:
    #     # channel_first
    #     ssim_item = ssim(select_raw_images.cpu().numpy(), select_adv_images.cpu().numpy(),
    #                      win_size=11, data_range=1.0, channel_axis=1).item()
    return [l1_distortion_factor, l2_distortion_factor, linf_distortion_factor], psnr_item, \
        [l1_avg_epsion.mean().item(), l2_avg_epsion.mean().item(), linf_avg_epsion.mean().item()]


print(calculate_avg_norm_distortion_factor(images, clipped, success))

L0BrendelBethgeAttack = fa.attacks.L0BrendelBethgeAttack()
