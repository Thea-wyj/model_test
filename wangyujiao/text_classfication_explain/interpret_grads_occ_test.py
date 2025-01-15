"""Script to serialize the saliency with gradient approaches and occlusion."""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.special import softmax

import torch
from captum.attr import DeepLift, GuidedBackprop, InputXGradient, Occlusion, \
    Saliency, configure_interpretable_embedding_layer, \
    remove_interpretable_embedding_layer, LayerIntegratedGradients, LayerDeepLift, LayerFeatureAblation, LayerLRP
from captum.attr._utils import visualization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, mean_absolute_error, max_error
from torch import nn
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelWrapper(nn.Module):
    def __init__(self, token):
        super(ModelWrapper, self).__init__()
        self.instance = token

    def forward(self, token):
        result_logits = model(token).logits
        return result_logits


def model_fn(instance):
    test_token = tokenizer(instance, return_tensors="pt").data['input_ids'].to(device)
    result_logits = model(test_token).logits
    return result_logits


def collate_threshold_solo(instance: torch.Tensor,
                           batch: list,
                           threshold=1.0, classes=None) -> torch.Tensor:
    if classes is None:
        classes = ["mainland China politics", "Hong Kong - Macau politics", "International news",
                   "financial news",
                   "culture", "entertainment", "sports"]
    # 获取输入的实例的tokenIds batch[0]

    # 通过 instances[i][-1] 获取每个实例的saliencies

    # 将每个实例的saliencies中每个token对每个标签的贡献相加，得到每个token对所有标签的贡献，是个list
    word_saliencies = [sum([_d[f'{_c}'] for _c in classes]) for _d
                       in instance]
    sorted_idx = np.array(word_saliencies).argsort()[::-1]
    # 计算instance中非填充tokens的数量，并将其赋值给n_tokens
    n_tokens = len([_t for _t in batch if _t != pad_token_id]) - (len(batch) - len(word_saliencies))
    # 计算需要mask的tokens的数量，并将其赋值给num_mask_tokens
    num_mask_tokens = int((threshold / 100) * n_tokens)

    num_masked = 0
    if num_mask_tokens > 0:
        for _id in sorted_idx:
            if _id < n_tokens and batch[_id] != pad_token_id:
                batch[_id] = mask_token_id
                num_masked += 1
            if num_masked == num_mask_tokens:
                break

    return batch


def generate_saliency(saliency_path, saliency):
    modelw = ModelWrapper(model).to(device)
    if saliency == 'deeplift':
        ablator = LayerDeepLift(modelw, model.bert.embeddings)
    elif saliency == 'ig':
        ablator = LayerIntegratedGradients(modelw, model.bert.embeddings)
    elif saliency == 'occlusion':
        ablator = Occlusion(modelw)

    class_attr_list = defaultdict(lambda: [])
    token_ids = tokenizer.encode(args.instance)
    text_list = tokenizer.convert_ids_to_tokens(token_ids)

    input_index = torch.tensor(token_ids, device=device).unsqueeze(0)
    baseline = torch.zeros_like(input_index)
    score, pred = torch.max(pre_logits.data, 1)

    additional = None
    # token_ids += batch[0].detach().cpu().numpy().tolist()
    # 收敛分数
    delta = torch.tensor(0.8, device=device)
    for i, cls_ in enumerate(args.class_name):
        if saliency == 'occlusion':
            attributions = ablator.attribute(input_index,
                                             sliding_window_shapes=(
                                                 args.sw,), target=i)
            attributions = attributions.squeeze(0)
        elif saliency == 'deeplift':
            attributions, delta = ablator.attribute(input_index, baseline, target=i, return_convergence_delta=True)
            attributions = attributions.squeeze(0)
            attributions = torch.mean(attributions, dim=1)
        else:
            attributions = ablator.attribute(input_index, target=i,
                                             additional_forward_args=additional)
            attributions = attributions.squeeze(0)
            attributions = torch.mean(attributions, dim=1)
        if i == pre_idx:
            attributions_explain = attributions
        if i == min_idx:
            anti_attributions_explain = attributions
        class_attr_list[cls_] += [_l for _l in
                                  attributions.cpu().detach().numpy()]

    attributions_explain = attributions_explain / torch.norm(attributions_explain)
    anti_attributions_explain = anti_attributions_explain / torch.norm(anti_attributions_explain)
    attributions_explain = attributions_explain.cpu().detach().numpy()
    anti_attributions_explain = anti_attributions_explain.cpu().detach().numpy()

    viz = [visualization.VisualizationDataRecord(
        attributions_explain.tolist(),
        score.data.item(),
        args.class_name[pre_idx],
        args.class_name[pre_idx],
        args.class_name[pre_idx],
        attributions_explain.sum(),
        text_list,
        delta)]

    print('Visualize attributions based on' + args.saliency)
    table = visualization.visualize_text(viz)
    with open(os.path.join(args.output_dir,
                           f'{args.saliency}_result.html'), "w", encoding='utf-8') as file:
        file.write('<meta charset="UTF-8">\n' + table.data)

    # SERIALIZE
    print('Serializing...', flush=True)
    with open(saliency_path + '_saliency', 'w') as out:
        saliencies = []
        for token_i, token_id in enumerate(token_ids):
            token_sal = {'token': tokenizer.convert_ids_to_tokens(token_id)}
            for cls_ in args.class_name:
                token_sal[cls_] = float(class_attr_list[cls_][token_i])
            saliencies.append(token_sal)

        out.write(json.dumps({'tokens': saliencies}) + '\n')
        out.flush()
    # 获取忠实度()
    data = {
        "Faithfulness": get_faithfulness(saliencies),
        "Class Sensitivity": get_class_sensitivity(attributions_explain, anti_attributions_explain),
        "Consistency": get_consistency(attributions_explain, percent=50, window=1),
    }
    with open(saliency_path + "_metrics.json", "w") as f:
        json.dump(data, f)


# 获取忠实度
def get_faithfulness(instance):
    thresholds = list(range(0, 110, 10))
    model_scores = []
    for threshold in thresholds:
        token_ids = tokenizer.encode(args.instance)
        masked_token_ids = collate_threshold_solo(instance,
                                                  batch=token_ids,
                                                  threshold=threshold,
                                                  classes=args.class_name)
        results = model(torch.tensor(masked_token_ids, device=device).unsqueeze(0)).logits
        model_scores.append(abs_diff(pre_logits[0][pre_idx].item(), results[0][pre_idx].item()))
    print(thresholds, model_scores)
    return auc(thresholds, model_scores)


def get_class_sensitivity(exp_max, exp_min):
    """
       consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
       数值范围[-1,1],数值越大越相似。
       :param tensor1:
       :param tensor2:
       :return:
       """

    tensor1 = torch.tensor(exp_max)
    tensor2 = torch.tensor(exp_min)

    # 求模长
    tensor1_norm = torch.norm(tensor1)
    tensor2_norm = torch.norm(tensor2)
    # 内积
    tensor1_tensor2 = torch.dot(tensor1, tensor2)
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    # 转化为float
    cosin = cosin.numpy()

    return (cosin + 1) / 2


def abs_diff(x, y):
    diff = abs(x - y)
    return diff


def get_consistency(exp_max, percent=50, window=1):
    token_list = tokenizer.encode(args.instance)
    # 计算前n%贡献度的阈值
    threshold = sorted(exp_max)[int(len(exp_max) * (100 - percent) / 100)]

    # 找到所有高于阈值的元素的索引
    high_contribution_indices = [i for i, e in enumerate(exp_max) if e > threshold]
    if len(high_contribution_indices) == 0:
        return 0
    # 根据这些索引，在item中提取相应的元素以及其前后5个元素
    selected_indices = set()
    for i in high_contribution_indices:
        selected_indices.update(range(max(0, i - window), min(len(token_list), i + window + 1)))

    # 生成并返回结果列表
    result = [token_list[i] for i in sorted(selected_indices)]
    prediction = pre_logits[0][pre_idx].item()
    new_prediction = model(torch.tensor(result, device=device).unsqueeze(0)).logits[0][pre_idx].item()
    ccd = abs_diff(new_prediction, prediction)
    return ccd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        help="Path where the models can be found, "
                             "with a common prefix, without _1",
                        default='model/roberta-base-finetuned-chinanews-chinese', type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--output_dir",
                        help="Path where the saliencies will be serialized",
                        default='output/',
                        type=str)
    parser.add_argument("--sw", help="Sliding window", type=int, default=1)
    parser.add_argument("--saliency", help="Saliency type", type=str, default='occlusion', )
    parser.add_argument("--batch_size",
                        help="Batch size for explanation generation", type=int,
                        default=300)
    parser.add_argument("--instance", help="",
                        default="《007》首周票房2.15亿 ，挤走《一代宗师》，丹尼尔·克雷格第三次出演007。", type=str)
    
    parser.add_argument("--instance_file", help="",
                        default="C:\\Users\\42102\Desktop\\文本可解释.txt",
                        type=str)



    parser.add_argument("--class_name", help="",
                        default=["mainland China politics", "Hong Kong - Macau politics", "International news",
                                 "financial news",
                                 "culture", "entertainment", "sports"], type=list)

    args = parser.parse_args()



    with open(args.instance_file, "r", encoding="utf-8") as file:
        instance = file.read()

    args.instance=instance
    print(instance)


    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pre_logits = model_fn(args.instance)

    pre_idx = torch.argmax(pre_logits).item()
    min_idx = torch.argmin(pre_logits).item()

    saliency = args.saliency
    if saliency in ['guided', 'sal', 'inputx', 'deeplift']:
        aggregations = ['mean']  #
    else:  # occlusion
        aggregations = ['none']
    for aggregation in aggregations:
        flops = []
        print('Running aggregation ', aggregation, flush=True)

        model_path = args.model_path
        generate_saliency(
            os.path.join(args.output_dir,
                         f'{saliency}'),
            saliency)
