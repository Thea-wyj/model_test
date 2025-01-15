"""Script to serialize the saliencies from the LIME method."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from lime.lime_text import LimeTextExplainer
from sklearn.metrics import auc
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, batch_size):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.batch_size = batch_size

    def forward(self, instance):
        results = []
        token_ids = [tokenizer.encode(i) for i in instance]
        for i in tqdm(range(0, len(token_ids), self.batch_size),
                      'Building a local approximation...'):
            batch_ids = token_ids[i:i + self.batch_size]
            max_batch_id = max([len(_l) for _l in batch_ids])
            padded_batch_ids = [
                _l + [pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(tokens_tensor).logits
            results += logits.detach().cpu().numpy().tolist()
        return np.array(results)


def model_fn(instance):
    test_token = tokenizer(instance, return_tensors="pt").data['input_ids']
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


def generate_saliency(saliency_path):
    modelw = ModelWrapper(model, device, tokenizer, args.batch_size)

    explainer = LimeTextExplainer(class_names=args.class_name)
    instance = args.instance

    with open(saliency_path, 'w') as out:
        token_ids = tokenizer.encode(instance)
        if len(token_ids) < 6:
            token_ids = token_ids + [pad_token_id] * (
                    6 - len(token_ids))
        text_list = tokenizer.convert_ids_to_tokens(token_ids)
        result_str = ' '.join(text_list)

        exp = explainer.explain_instance(
            result_str, modelw,
            num_features=len(text_list),
            top_labels=args.labels)

        explanation = {}
        saliencies = []
        for i, cls_ in enumerate(args.class_name):
            cls_expl = [None] * len(exp.as_list(label=i))
            for j, item in enumerate(exp.as_list(label=i)):
                cls_expl[exp.as_map()[i][j][0]] = item
            explanation[cls_] = cls_expl

        for index, item in enumerate(cls_expl):
            token_saliency = {'token': item[0]}
            for cls_ in args.class_name:
                token_saliency[cls_] = explanation[cls_][index][1]
            saliencies.append(token_saliency)

        out.write(json.dumps({'tokens': saliencies}) + '\n')
        out.flush()
    # 获取忠实度()
    data = {
        "Faithfulness": get_faithfulness(saliencies),
        "Class Sensitivity": get_class_sensitivity(exp),
        "Consistency": get_consistency(exp, percent=50, window=1),
    }
    with open(saliency_path + "_metrics.json", "w") as f:
        json.dump(data, f)
    return exp


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


def get_class_sensitivity(exp):
    """
       consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
       数值范围[-1,1],数值越大越相似。
       :param tensor1:
       :param tensor2:
       :return:
       """
    exp_max = exp.as_list(label=pre_idx)
    exp_min = exp.as_list(label=min_idx)

    exp_max = sorted(exp_max, key=lambda x: x[0])
    exp_min = sorted(exp_min, key=lambda x: x[0])

    tensor1 = torch.tensor([i[1] for i in exp_max])
    tensor2 = torch.tensor([i[1] for i in exp_min])

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


def get_consistency(exp, percent=50, window=1):
    exp_max = exp.as_list(label=pre_idx)
    token_list = tokenizer.encode(args.instance)
    text_list = tokenizer.convert_ids_to_tokens(token_list)
    filtered_exp_max = [e for e in exp_max if e[0] in text_list]

    filtered_exp_max = sorted(filtered_exp_max, key=lambda x: x[1], reverse=True)
    exp_len = int(len(filtered_exp_max) * percent / 100)
    exp_cut = filtered_exp_max[:exp_len]

    high_contribution_indices = [text_list.index(e[0]) for e in exp_cut]

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
                        help="Path to the model",
                        default='model/roberta-base-finetuned-chinanews-chinese', type=str)
    parser.add_argument("--output_dir",
                        help="Path where the saliency will be serialized",
                        default='output/', type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=True)
    parser.add_argument("--labels", help="Number of target labels", type=int,
                        default=7)
    parser.add_argument("--batch_size", help="Size of batch_size", type=int,
                        default=300)

    parser.add_argument("--instance", help="",
                        default="《007》首周票房2.15亿 ，挤走《一代宗师》，丹尼尔·克雷格第三次出演007。",
                        type=str)


    parser.add_argument("--instance_file", help="",
                        default="test.txt",
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pre_logits = model_fn(args.instance)

    pre_idx = torch.argmax(pre_logits).item()
    min_idx = torch.argmin(pre_logits).item()

    exp = generate_saliency(os.path.join(args.output_dir, 'lime_saliency'))
    exp.save_to_file(os.path.join(args.output_dir, 'lime_result.html'), labels=[pre_idx],
                     predict_proba=False)
    # plt = exp.as_pyplot_figure(label=pre_idx)
    # plt.show()
