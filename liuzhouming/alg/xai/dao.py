import json
import os
import re
import pydot
import tree_extraction


def read_json(path):
    # 将json文件转化为一个字典
    if os.path.exists(path):
        with open(path, 'r') as load_f:
            load_dict = json.load(load_f)
            return load_dict
    else:
        with open(path, 'w') as load_f:
            save_json(path, {})
            return {}


def save_json(path, dict):
    # 将字典转化为json文件
    with open(path, "w") as f:
        json.dump(dict, f)


def update_json(path, dict):
    # 向json中添加新项
    old_dict = read_json(path)
    old_dict.update(dict)
    save_json(path, old_dict)


def add_json(path, tuple):
    # 向json中添加新项（黑盒和代理的json）
    old_dict = read_json(path)
    if len(tuple) == 2:  # 黑盒
        if not old_dict.get(tuple[0]):
            old_dict[tuple[0]] = tuple[1]
        else:
            old_dict[tuple[0]].update(tuple[1])
    else:  # 代理
        if not old_dict.get(tuple[0]):
            old_dict[tuple[0]] = {tuple[1]: tuple[2]}
        else:
            if not old_dict[tuple[0]].get(tuple[1]):
                old_dict[tuple[0]][tuple[1]] = tuple[2]
            else:
                old_dict[tuple[0]][tuple[1]].update(tuple[2])
    save_json(path, old_dict)


def test_train_read(filename):
    # 获取数据集的名字和类型
    group = re.match("(.+)_(.+).csv", filename)
    if not group:
        return {"error": "请检查文件命名格式是否有误"}, None, None
    else:
        name = group.group(1)
        type = group.group(2)
        if type != "test" and type != "train":
            return {"error": "请检查数据集后缀是否为test或train"}, None, None
        # if not session.get('test_train'):
        #     session['test_train'] = "{}_{}".format(name, type)
        # else:
        #     name_1, type_1 = session.get('test_train').split("_")
        #     print("name1{}type1{}name{}type{}".format(name_1, type_1, name, type))
        #     del session["test_train"]
        #     if name_1 != name:
        #         return {"error": "请检查测试集和训练集是否属于同一数据集"}, None, None
        #     elif (type == "test" and type_1 != "train") or (type == "train" and type_1 != "test"):
        #         return {"error": "请上传一个测试集一个训练集"}, None, None
        return {}, name, type


# 判断train和test是否同时存在,有问题的直接删了
def detect_train_test():
    # 做一个判断，确保每个数据集都同时有test和train
    test_json = read_json("static/dataset/test.json")
    train_json = read_json("static/dataset/train.json")
    test_keys = list(test_json.keys())
    train_keys = list(train_json.keys())
    same = [x for x in test_keys if x in train_keys]
    different_test = [x for x in test_keys if x not in same]
    different_train = [x for x in train_keys if x not in same]
    # 不符合的删掉
    for x in different_test:
        file = test_json.pop(x, '')
        if file != '':
            os.remove('static/Data/dataset/' + file)
    for x in different_train:
        file = train_json.pop(x, '')
        if file != '':
            os.remove('static/Data/dataset/' + file)
    save_json("static/dataset/test.json", test_json)
    save_json("static/dataset/train.json", train_json)
    different_test.extend(different_train)
    return same, different_test


# 划分代理模型输入用
def proxy_dataset_read(filename):
    # 获取数据集的名字和类型
    group = re.match("(.+)_(.+)_(.+)\..+", filename)
    if not group:
        return {"error": "请检查文件命名格式是否有误"}, None, None
    else:
        model = group.group(1)
        blackbox = group.group(2)
        dataset = group.group(3)
        return {}, model, blackbox, dataset


# 划分黑盒模型输入用
def blackbox_dataset_read(filename):
    # 获取数据集的名字和类型
    group = re.match("(.+)_(.+)\..+", filename)
    if not group:
        return {"error": "请检查文件命名格式是否有误"}, None, None
    else:
        model = group.group(1)
        dataset = group.group(2)
        return {}, model, dataset


# 读取规则
def read_rules(path):
    """读入规则"""
    with open(path, 'r') as f:
        content = f.read()
        rule_list = content.splitlines()[1:]  # 规则列表
        rule_type = content.splitlines()[0]
        prob = []  # 预测概率
        for i in rule_list:
            str_prob = re.findall("\),(\d+\.?\d*)", i)  # 提取概率 list
            if str_prob:
                float_prob = float(str_prob[0])
                prob.append(float_prob)
        Rules = []
        for i, rule in enumerate(rule_list):
            ifs = re.findall("\'([^\']+)\'", rule)
            ifs = " AND ".join(ifs)
            str_prob = re.findall("\),(\d+\.?\d*)", rule)  # 提取概率 list
            if str_prob:
                float_prob = float(str_prob[0])
                Rules.append({"IF": ifs, "THEN": float_prob})
            else:
                Rules.append({"IF": "default", "THEN": float(rule)})
    return Rules, rule_type


# 获取文件路径
def get_path(session):
    path_dict = {}
    # 数据集路径
    dataset = session["dataset"]
    test_json = read_json("static/dataset/test.json")
    train_json = read_json("static/dataset/train.json")
    path_dict["dataset_test"] = "static/Data/dataset/" + test_json[dataset]
    path_dict["dataset_train"] = "static/Data/dataset/" + train_json[dataset]
    # 黑盒代理模型
    xai = session["XAI"]
    proxy_type = session["proxy_type"]
    bb_json = read_json("static/blackbox/index.json")
    pro_json = read_json("static/Data/proxy_" + proxy_type + "/index.json")
    for key in xai.keys():
        bb, pro = xai[key].split('_')
        path_dict[key] = {"blackbox": "static/Data/blackbox/" + bb_json[dataset][bb],
                          "proxy": "static/Data/proxy_" + proxy_type + "/" + pro_json[dataset][bb][pro]}
    return path_dict


# 预览决策树用
def read_tree(proxy_path, pic_path):
    # 读取决策树文件
    decision_info_path = proxy_path  # 决策树信息文件存储地址
    tree_ = tree_extraction.getTree(decision_info_path)
    feature_text = tree_extraction.getFeatureText(decision_info_path)  # 获取特征文本（画图）
    lable_text = tree_extraction.getLabelText(decision_info_path)  # 获取标签文本（画图）
    # 构建树结构 返回根节点
    treeNode = tree_extraction.buildTree(tree_)
    # 输出决策树画图用的dot文件
    dot_path = 'static/img/tmp_tree.dot'  # 决策树用于画图的dot文件
    tree_extraction.visualize(treeNode, feature_text, lable_text, dot_path)
    (graph,) = pydot.graph_from_dot_file(dot_path)
    graph.write_png(pic_path);


if __name__ == "__main__":
    pass
