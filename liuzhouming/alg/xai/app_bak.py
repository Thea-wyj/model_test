import argparse
import json
import os
import sys
import predict
import rule_extraction
import tree_extraction
from predict import getIndexList
import dao

# 获取文件路径
def get_path(config):
    path_dict = {}
    # 数据集路径
    datasetName = config.get("dataSetName")
    root = os.path.dirname(__file__)

    test_json = dao.read_json(os.path.join(root, 'static/Data/dataset/test.json'))
    train_json = dao.read_json(os.path.join(root, "static/Data/dataset/train.json"))
    path_dict["dataset_test"] = os.path.join(os.path.join(root, "static/Data/dataset"), test_json[datasetName])
    path_dict["dataset_train"] = os.path.join(os.path.join(root, "static/Data/dataset"),  train_json[datasetName])
    # 黑盒代理模型
    xaiList = config.get("blackBoxAndProxyModelConfigList")
    proxy_type = config.get("proxyType")
    bb_json = dao.read_json(os.path.join(root, "static/Data/blackbox/index.json"))
    pro_json = dao.read_json(os.path.join(os.path.join(root, "static/Data/proxy_" + proxy_type),"index.json"))

    index = 1
    for xai in xaiList:
        bb, pro = xai['blackBoxModelName'], xai['proxyModelName']
        path_dict["XAI" + str(index)] = {
            "blackbox": os.path.join(os.path.join(root, "static/Data/blackbox/") , bb_json[datasetName][bb]),
            "proxy": os.path.join(os.path.join(root, "static/Data/proxy_" + proxy_type), pro_json[datasetName][bb][pro])
        }
        index = index + 1
    return path_dict

def change_minio_url_to_local_path(config):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configFilePath',
                        type=str,
                        default='E:\\MyJavaCode\\micro-test-ai\\micro-evaluate-service\\src\\main\\java\\com\\macro\\cloud\\evaluate\\controller\\alg\\2100082b-8043-43ba-8ec1-60e715a5a2b5.json',
                        help='可解释性测评配置')

    parser.add_argument('--proxy_type', type=str, default='tree', help='代理模型类型')
    parser.add_argument('--debug', type=int, default=1)

    # 解析参数
    args = parser.parse_args()
    configFilePath = args.configFilePath
    proxy_type = args.proxy_type

    config = {}
    if configFilePath != "":
        with open(configFilePath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 删除文件
        if args.debug != 1:
            os.remove(configFilePath)

    # do something
    if len(config) == 0:
        print(json.dumps(dict()))
        sys.exit(0)

    proxy_type = config.get("proxyType")  # 代理模型类型
    evaModel = config.get("evaModel")
    evaModelConfig = eval(config.get("evaModelConfig"))
    if proxy_type == "":
        print(json.dumps(dict()))
        sys.exit(0)

    root = os.path.dirname(__file__)
    path_dict = dict()

    # rule
    # print(__file__)

    indicator_dict = dict()
    # 解析参数
    path_dict = get_path(config)

    # 指标提取
    # 基于规则
    if proxy_type == "rule":
        # path_dict_1 = {'dataset_test': os.path.join(root, 'static/Data/dataset/breastcancer_test.csv'),
        #              'dataset_train': os.path.join(root, 'static/Data/dataset/breastcancer_train.csv'),
        #              'XAI1': {'blackbox': os.path.join(root, 'static/Data/blackbox/randomforest_breastcancer.pkl'),
        #                       'proxy': os.path.join(root, 'static/Data/proxy_rule/brs_randomforest_breastcancer.txt')},
        #              'XAI2': {'blackbox': os.path.join(root, 'static/Data/blackbox/svm_breastcancer.pkl'),
        #                       'proxy': os.path.join(root, 'static/Data/proxy_rule/mdl_svm_breastcancer.txt')},
        #              'XAI3': {'blackbox': os.path.join(root, 'static/Data/blackbox/randomforest_breastcancer.pkl'),
        #                       'proxy': os.path.join(root, 'static/Data/proxy_rule/sbrl_randomforest_breastcancer.txt')}}

        indicator_dict = rule_extraction.rule_extract(path_dict)
        for key, value in indicator_dict.items():
            value["consistency"] = round(value["consistency"], 3)
            value["coverage_rate"] = round(value["coverage_rate"], 3)
            value["class_overlap_rate"] = round(value["class_overlap_rate"], 3)
            value["overlap_rate"] = round(value["overlap_rate"], 3)
            value["conflict_rate"] = round(value["conflict_rate"], 3)
            value["model_size"] = round(value["model_size"], 3)
            value["total_rule_length"] = round(value["total_rule_length"], 3)
            value["max_length"] = round(value["max_length"], 3)

    elif proxy_type == "tree":
        # tree
        # path_dict_1 = {'dataset_test': os.path.join(root, 'static/Data/dataset/breastcancer_test.csv'),
        #              'dataset_train': os.path.join(root, 'static/Data/dataset/breastcancer_train.csv'),
        #              'XAI1': {'blackbox': os.path.join(root, 'static/Data/blackbox/randomforest_breastcancer.pkl'),
        #                       'proxy': os.path.join(root,
        #                                             'static/Data/proxy_tree/tree1_randomforest_breastcancer.txt')},
        #              'XAI2': {'blackbox': os.path.join(root, 'static/Data/blackbox/svm_breastcancer.pkl'),
        #                       'proxy': os.path.join(root, 'static/Data/proxy_tree/tree2_svm_breastcancer.txt')}}

        # 指标提取
        indicator_dict = tree_extraction.tree_extract(path_dict)
        for key, value in indicator_dict.items():
            value["AUC"] = round(value["AUC"], 3)
            value["APL"] = round(value["APL"], 3)
            value["node count"] = round(value["node count"], 3)
            value["duplicate_subtree"] = round(value["duplicate_subtree"], 3)
            value["duplicate_attr"] = round(value["duplicate_attr"], 3)

    # print indicator_dict
    sum_indicator_dict = getIndexList(proxy_type, indicator_dict)
    # print("sum_indicator_dict", sum_indicator_dict)

    # topsis + 层次分析法
    list, w1_param, w2_param, w3_param, text_param = None, None, None, None, None

    if evaModel == "topsis_ahp":
        list, w1_param, w2_param, w3_param, text_param = predict.predict_topsis_ahp(proxy_type,
                                                                                    indicator_dict,
                                                                                    evaModelConfig=evaModelConfig)
    elif evaModel == "topsis_ahp_entropy":
        list, w1_param, w2_param, w3_param, text_param = predict.predict_topsis_ahp_entropy(proxy_type,
                                                                                            indicator_dict,
                                                                                            evaModelConfig=evaModelConfig)
    elif evaModel == "fuzzy_ahp":
        list, w1_param, w2_param, w3_param, text_param = predict.predict_fuzzy_ahp(proxy_type, indicator_dict,
                                                                                            evaModelConfig=evaModelConfig)
    elif evaModel == "rsr_ahp":
        list, w1_param, w2_param, w3_param, text_param = predict.predict_rsr_ahp(proxy_type, indicator_dict,
                                                                                 evaModelConfig=evaModelConfig)
    elif evaModel == "grey_ahp":
        list, w1_param, w2_param, w3_param, text_param = predict.predict_grey_ahp(proxy_type, indicator_dict,
                                                                                 evaModelConfig=evaModelConfig)
    # tree
    if proxy_type == "tree":
        sum_indicator_dict["proxy_type"] = "tree"
    # rule
    if proxy_type == "rule":
        sum_indicator_dict["proxy_type"] = "rule"
    sum_indicator_dict["method_name"] = evaModel # 方法名
    result = [sum_indicator_dict, list, w1_param, w2_param, w3_param, text_param]
    print(json.dumps(result))
