import re
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib


def operation_extract(rule_item):
    # 提取>,<,=,>=,<=
    # rule_item为一个规则项，str
    # operation为要返回的运算符 list
    operation = []
    i = 0
    while i < len(rule_item):
        if rule_item[i] == '=':
            operation.append('=')
            i += 1
        elif rule_item[i] == '>':
            if rule_item[i + 1] == '=':
                operation.append('>=')
                i += 2
            else:
                operation.append('>')
                i += 1
        elif rule_item[i] == '<':
            if rule_item[i + 1] == '=':
                operation.append('<=')
                i += 2
            else:
                operation.append('<')
                i += 1
        else:
            i += 1
    return operation


def feature_name_extract(rule_item, feature_list):
    # 提取属性名
    # rule_item为一个规则项，str
    # str为要返回的运算符 str
    for feature in feature_list:
        feature = feature + ' '
        if feature in rule_item:
            return feature.strip()
    return None


def compare_operation(op_str, value1, value2):
    # op_str：比较运算符 str
    # value1：样本中的属性值
    # value2：规则的属性值
    # 返回0表示不匹配，1表示匹配
    if op_str == '=':
        if value1 != value2:
            return 0
    elif op_str == '>':
        if value1 <= value2:
            return 0
    elif op_str == '<':
        if value1 >= value2:
            return 0
    elif op_str == '<=':
        if value1 > value2:
            return 0
    else:  # >=
        if value1 < value2:
            return 0
    return 1


def match_onerule_conti(rule, sample, feature_list):
    # rule是从文本中提取的一行规则
    # sample是字典列表
    rule_items = re.findall("'([^']*)'", rule)  # 提取规则项
    if rule_items:
        for item in rule_items:
            operation = operation_extract(item)  # 提取比较运算符
            feature = feature_name_extract(item, feature_list)  # 提取属性名
            if feature is None:
                print("提取特征错误")
            values = re.findall("(\d+\.?\d*)", item)  # 提取属性值（最后一项为概率值）
            # print(operation)
            # print(values)
            if len(operation) == 1:
                if compare_operation(operation[0], sample[feature], float(values[0])) == 0:
                    return 0
            if len(operation) == 2:
                if ' or ' in item:
                    temp1 = compare_operation(operation[0], sample[feature], float(values[0]))
                    temp2 = compare_operation(operation[1], sample[feature], float(values[1]))
                    if (temp1 or temp2) == 0:
                        return 0
                else:  # and
                    temp1 = compare_operation(operation[0], float(values[0]), sample[feature])
                    temp2 = compare_operation(operation[1], sample[feature], float(values[1]))
                    if (temp1 and temp2) == 0:
                        return 0
        return 1
    else:
        return 0


def match_rules(rule_list, sample, prob_list, feature_list):
    n = 0
    n_conflict = 0
    class_ = 0
    class_temp = 0
    for i in range(len(rule_list)):
        if match_onerule_conti(rule_list[i], sample, feature_list):
            n += 1
            if i <= len(prob_list) - 1:
                if prob_list[i] > 0.5:
                    class_temp = 1
                else:
                    class_temp = 0
            if n > 1:  # 匹配多个规则
                if class_temp != class_:
                    n_conflict += 1
            class_ = class_temp
    return n, n_conflict  # 匹配的规则数，矛盾的次数


def pred(x_dict, rule_list, feature_list):
    """预测结果计算"""
    for i, rule in enumerate(rule_list):
        if match_onerule_conti(rule, x_dict, feature_list):  # 第一条匹配的规则
            if float(re.search("\),(\d+\.?\d*)", rule).group(1)) > 0.5:
                return 1
            else:
                return 0
    if float(rule_list[-1]) > 0.5:
        return 1
    else:
        return 0


def definiteness(data_list, rule_list, prob, type, feature_list):
    """计算明确性指标，其中重叠率和矛盾率在type为list时，均为0"""
    n = len(data_list)
    # 类覆盖率
    class_type = set([1 if x >= 0.5 else 0 for x in prob])
    class_overlap_rate = len(class_type) / 2.0
    n_matched_samples = 0
    n_overlap_samples = 0
    n_conflict_samples = 0
    for sample in data_list:
        for rule in rule_list:
            if match_onerule_conti(rule, sample, feature_list):
                n_matched_samples += 1
                break
        if type == "set":
            overlap, conflict = match_rules(rule_list, sample, prob, feature_list)
            if overlap > 1:
                n_overlap_samples += 1
            if conflict > 0:
                n_conflict_samples += 1
        # if match_rules(rule_list,sample,prob):
        #   n_overlap_samples += 1
    # print(n_matched_samples,n_overlap_samples,n_conflict_samples)
    coverage_rate = n_matched_samples / n  # 样本覆盖率
    overlap_rate = n_overlap_samples / n  # 重叠率
    conflict_rate = n_conflict_samples / n  # 矛盾率
    return coverage_rate, overlap_rate, conflict_rate, class_overlap_rate


def rule_extract(path_dict):
    """从dataset_test加载样本、特征列表"""
    dataset_test_path = path_dict["dataset_test"]
    dataset_train_path = path_dict["dataset_train"]
    csv_data = pd.read_csv(dataset_test_path, low_memory=False)  # 防止弹出警告
    feature_list = csv_data.columns.values.tolist()[:-1]
    x_test = np.array(csv_data)[:, :-1]
    data_list = []  # 组合成数据集列表，列表的每一个项是一个样本，每个项是dict
    for item in x_test:
        data_list_item = dict(zip(feature_list, item))  # 字典类型
        data_list.append(data_list_item)
    """遍历全部XAI提取指标"""
    index_dict = {}
    for i in range(len(path_dict) - 2):
        xai = "XAI" + str(i + 1)
        blackbox_path = path_dict[xai]["blackbox"]
        proxy_path = path_dict[xai]["proxy"]

        """ 加载黑盒、规则 """
        # 从blackbox_path加载黑盒，获取黑盒预测结果y_bb
        blackbox = joblib.load(blackbox_path)
        y_bb = blackbox.predict(x_test)
        # 从proxy_path加载规则
        with open(proxy_path, 'r') as f:
            content = f.read()
            rule_list = content.splitlines()  # 规则列表
            rule_type = rule_list[0]  # 规则类型
            rule_list = rule_list[1:]
            prob = []  # 预测概率
            for i in rule_list:
                str_prob = re.findall("\),(\d+\.?\d*)", i)  # 提取概率 list
                if str_prob:
                    float_prob = float(str_prob[0])
                    prob.append(float_prob)
            """复杂性"""
            model_size = len(rule_list)  # 规则数量
            max_length = 0  # 最大规则长度
            total_rule_length = 0  # 规则总长度
            for i in rule_list:
                rule_items = re.findall("'([^']*)'", i)  # 提取规则项
                if not rule_items:  # 是default规则
                    model_size -= 1
                total_rule_length += len(rule_items)
                if len(rule_items) > max_length:
                    max_length = len(rule_items)

            """明确性"""
            coverage_rate, overlap_rate, conflict_rate, class_overlap_rate = definiteness(data_list, rule_list, prob,
                                                                                          rule_type, feature_list)

            """一致性"""
            y_rule = np.array([pred(x, rule_list, feature_list) for x in data_list])  # 规则预测结果
            consistency = roc_auc_score(y_rule, y_bb)

            index_dict[xai] = {"model_size": model_size, "max_length": max_length,
                               "total_rule_length": total_rule_length,
                               "coverage_rate": coverage_rate, "overlap_rate": overlap_rate,
                               "class_overlap_rate": class_overlap_rate,
                               "conflict_rate": conflict_rate, "consistency": consistency}
    # print(index_dict)
    return index_dict
