import torch
import foolbox
import time
import numpy as np
import torchvision
from skimage.measure import compare_ssim
from skimage.metrics import peak_signal_noise_ratio

# 加载自定义的PyTorch模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 或者加载自定义的TensorFlow模型
# model = tf.keras.models.load_model('model.h5')

# 加载需要使用的数据集，并生成Foolbox数据加载器
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
fb_loader = foolbox.data.DataLoaderFromPyTorch(loader, bounds=(0, 1))

# 初始化Foolbox攻击器
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)

# 定义各种攻击方法和对应的参数
attacks = {
    'fgsm': foolbox.attacks.FGSM,
    'pgd': foolbox.attacks.PGD,
    'cw': foolbox.attacks.CarliniWagnerL2Attack
}

attack_params = {
    'fgsm': {'epsilons': [0.01, 0.02, 0.03]},
    'pgd': {'epsilon': 0.3, 'stepsize': 0.1, 'steps': 40},
    'cw': {'confidence': 0.1, 'learning_rate': 0.01, 'binsearch_steps': 10},
}

# 初始化结果变量
results = {}

# 对每种攻击方法进行评估
for attack_name, attack_fn in attacks.items():

    results[attack_name] = {'success_rate': [], 'distortion': [],
                            'ssim': [], 'psnr': [], 'execution_time': []}
    attack_param = attack_params[attack_name]

    for i, (inputs, labels) in enumerate(fb_loader):
        start = time.time()

        # 对样本进行攻击
        adv_inputs = attack_fn(fmodel, inputs.numpy(), labels.numpy(), **attack_param)

        end = time.time()

        # 计算攻击成功率
        success_rate = np.mean(fmodel.predict(adv_inputs).argmax(axis=-1) != labels.numpy())

        # 计算失真度
        distortion = np.mean(np.sqrt(np.sum((inputs.numpy() - adv_inputs) ** 2, axis=(1, 2, 3))))

        # 计算SSIM和PSNR
        ssim = np.mean(
            [compare_ssim(inputs.numpy()[j], adv_inputs[j], multichannel=True) for j in range(len(inputs.numpy()))])
        psnr = np.mean([peak_signal_noise_ratio(inputs.numpy()[j], adv_inputs[j], data_range=1) for j in
                        range(len(inputs.numpy()))])

        # 记录执行时间
        execution_time = end - start

        # 将结果添加到对应攻击方法的结果变量
        results[attack_name]['success_rate'].append(success_rate)
        results[attack_name]['distortion'].append(distortion)
        results[attack_name]['ssim'].append(ssim)
        results[attack_name]['psnr'].append(psnr)
        results[attack_name]['execution_time'].append(execution_time)

    # 计算各种指标的平均值
    results[attack_name]['mean_success_rate'] = np.mean(results[attack_name]['success_rate'])
    results[attack_name]['mean_distortion'] = np.mean(results[attack_name]['distortion'])
    results[attack_name]['mean_ssim'] = np.mean(results[attack_name]['ssim'])
    results[attack_name]['mean_psnr'] = np.mean(results[attack_name]['psnr'])
    results[attack_name]['mean_execution_time'] = np.mean(results[attack_name]['execution_time'])

# 输出每种攻击方法的结果
for attack_name, result in results.items():
    print('Attack: ', attack_name)
    print('Success Rate: ', result['mean_success_rate'])
    print('Distortion: ', result['mean_distortion'])
    print('SSIM: ', result['mean_ssim'])
    print('PSNR: ', result['mean_psnr'])
    print('Execution Time: ', result['mean_execution_time'])
