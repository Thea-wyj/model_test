import json
import numpy as np
import pandas as pd
from fractions import Fraction

from model_fuzzy_ahp import AHP
from model_grey_ahp import grs_count
from model_rsr_ahp import rsr
from model_topsis_ahp import TOPSIS
from model_topsis_ahp_entropy import TOPSIS_AHP_Entropy

EVAMODEL_NAME_MAP = {"fuzzy_ahp": "模糊综合分析法 + 层次分析法", "topsis_ahp": "topsis + 层次分析法", "rsr_ahp": "秩和比法 + 层次分析法",
                     "grey_ahp": "灰色关联分析法 + 层次分析法", "topsis_ahp_entropy": "topsis + 层次分析法 + 熵权法"}


def getIndexList(proxy_type, index_session):
    if proxy_type == "rule":
        result = {"model_name": [], "class_overlap_rate": [], "conflict_rate": [], "consistency": [],
                  "coverage_rate": [],
                  "max_length": [], "model_size": [], "overlap_rate": [], "total_rule_length": []}
        for key, item in index_session.items():
            result["model_name"].append(key)
            result["class_overlap_rate"].append(item["class_overlap_rate"])
            result["conflict_rate"].append(item["conflict_rate"])
            result["consistency"].append(item["consistency"])
            result["coverage_rate"].append(item["coverage_rate"])
            result["max_length"].append(item["max_length"])
            result["model_size"].append(item["model_size"])
            result["overlap_rate"].append(item["overlap_rate"])
            result["total_rule_length"].append(item["total_rule_length"])
        return result
    elif proxy_type == "tree":
        result = {"model_name": [], "AUC": [], "APL": [], "node count": [], "duplicate_subtree": [],
                  "duplicate_attr": []}
        for key, item in index_session.items():
            result["model_name"].append(key)
            result["AUC"].append(item["AUC"])
            result["APL"].append(item["APL"])
            result["node count"].append(item["node count"])
            result["duplicate_subtree"].append(item["duplicate_subtree"])
            result["duplicate_attr"].append(item["duplicate_attr"])
        return result


def ahp_tree(criteria, index1_2, index1_3, index2_3, c1, c2, c3, complexity1_2, clarity1_2):
    try:
        index1_2 = float(Fraction(index1_2))
        index1_3 = float(Fraction(index1_3))
        index2_3 = float(Fraction(index2_3))
        criteria[0][1] = index1_2
        criteria[1][0] = 1 / index1_2
        criteria[0][2] = index1_3
        criteria[2][0] = 1 / index1_3
        criteria[1][2] = index2_3
        criteria[2][1] = 1 / index2_3

        c = None
        # 方案层
        if c1 != None:
            complexity1_2 = float(Fraction(complexity1_2))
            clarity1_2 = float(Fraction(clarity1_2))
            c2[0][1] = complexity1_2
            c2[1][0] = 1 / complexity1_2
            c3[0][1] = clarity1_2
            c3[1][0] = 1 / clarity1_2
            c = [c1, c2, c3]
        return criteria, c
    except Exception:
        return "numbererror", "请输入正确的数字，请勿填入其他字符！"


def ahp_rule(criteria, index1_2, index1_3, index2_3, c1, c2, c3, complexity1_2, complexity1_3, complexity1_4,
             complexity2_3, complexity2_4, complexity3_4, clarity1_2, clarity1_3, clarity2_3):
    # AHP 评价矩阵
    # 准则层重要性矩阵
    try:
        index1_2 = float(Fraction(index1_2))
        index1_3 = float(Fraction(index1_3))
        index2_3 = float(Fraction(index2_3))
        criteria[0][1] = index1_2
        criteria[1][0] = 1 / index1_2
        criteria[0][2] = index1_3
        criteria[2][0] = 1 / index1_3
        criteria[1][2] = index2_3
        criteria[2][1] = 1 / index2_3

        c = None
        # 方案层
        if c1 != None:
            # 方案层
            complexity1_2 = float(Fraction(complexity1_2))
            complexity1_3 = float(Fraction(complexity1_3))
            complexity1_4 = float(Fraction(complexity1_4))
            complexity2_3 = float(Fraction(complexity2_3))
            complexity2_4 = float(Fraction(complexity2_4))
            complexity3_4 = float(Fraction(complexity3_4))
            clarity1_2 = float(Fraction(clarity1_2))
            clarity1_3 = float(Fraction(clarity1_3))
            clarity2_3 = float(Fraction(clarity2_3))
            c2[0][1] = complexity1_2
            c2[1][0] = 1 / complexity1_2
            c2[0][2] = complexity1_3
            c2[2][0] = 1 / complexity1_3
            c2[0][3] = complexity1_4
            c2[3][0] = 1 / complexity1_4
            c2[1][2] = complexity2_3
            c2[2][1] = 1 / complexity2_3
            c2[1][3] = complexity2_4
            c2[3][1] = 1 / complexity2_4
            c2[2][3] = complexity3_4
            c2[3][2] = 1 / complexity3_4
            c3[0][1] = clarity1_2
            c3[1][0] = 1 / clarity1_2
            c3[0][2] = clarity1_3
            c3[2][0] = 1 / clarity1_3
            c3[1][2] = clarity2_3
            c3[2][1] = 1 / clarity2_3
            c = [c1, c2, c3]
        return criteria, c

    except Exception:
        return "numbererror", "请输入正确的数字，请勿填入其他字符！"


# @app.route('/predict_topsis_ahp', methods=['POST', 'GET'])
def predict_topsis_ahp(proxy_type, indicator_dict, evaModelConfig):
    if proxy_type == "tree":
        b1 = np.array([])
        b2 = np.array([])
        b3 = np.array([])
        model_name = np.array([])

        for key, item in indicator_dict.items():
            if b1.shape[0] == 0:
                b1 = np.hstack((b1, item['AUC']))
                b2 = np.hstack((b2, [item['APL'], item['node count']]))
                b3 = np.hstack((b3, [item['duplicate_subtree'], item['duplicate_attr']]))
                model_name = np.hstack((model_name, key))
            else:
                b1 = np.vstack((b1, item['AUC']))
                b2 = np.vstack((b2, [item['APL'], item['node count']]))
                b3 = np.vstack((b3, [item['duplicate_subtree'], item['duplicate_attr']]))
                model_name = np.vstack((model_name, key))

        b = [b1, b2, b3]
        # 指标类型，1为收益型，-1为损失型
        s1 = np.array([1])
        s2 = np.array([-1, -1])
        s3 = np.array([-1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])
        c3 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])

        # criteria, c = ahp_tree(criteria, index1_2, index1_3, index2_3, c1, c2, c3, complexity1_2, clarity1_2)
        criteria, c = ahp_tree(criteria, evaModelConfig.get("index1_2"),
                               evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2,
                               c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("clarity1_2"))

        if criteria == "numbererror":
            return c
        topsis = TOPSIS(b, s, criteria, c)
        try:
            result, w2, w1 = topsis.run()
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["APL", "节点数量"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["重复子树比例", "重复子树比例"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = json.loads(result.to_json(orient='records'))

        XAIname_para = ""
        for i in range(len(list)):
            list[i]["model_name"] = model_name[i][0]
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            num_in_eva = -1
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            if num_in_eva != -1:
                list[num_in_eva]["AUC"] = indexOriginList["AUC"][i]
                list[num_in_eva]["APL"] = indexOriginList["APL"][i]
                list[num_in_eva]["node count"] = indexOriginList["node count"][i]
                list[num_in_eva]["duplicate_subtree"] = indexOriginList["duplicate_subtree"][i]
                list[num_in_eva]["duplicate_attr"] = indexOriginList["duplicate_attr"][i]

        list.sort(key=lambda k: k.get('C'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1
        text_param.append(XAIname_para)  # 6
        text_param.append(list[0]["model_name"])  # 7
        text_param.append(format(list[0]["C"], '.3f'))  # 8

        return list, w1_param, w2_param, w3_param, text_param,
        # session['result'] = json.loads(result.to_json())

        # return render_template("predict_topsis_ahp.html", list=list, w2=w2_param, w1=w1_param, w3=w3_param,
        #                        text_param=text_param, evaname=session["evamodel"])
    elif proxy_type == 'rule':
        # 二级指标矩阵 顺序mdrl brs sbrl
        b1 = np.array([])
        b2 = np.array([])
        b3 = np.array([])
        model_name = np.array([])

        for key, item in indicator_dict.items():
            if b1.shape[0] == 0:
                b1 = np.hstack((b1, item['consistency']))
                b2 = np.hstack((b2, [item['coverage_rate'], item['class_overlap_rate'], item['overlap_rate'],
                                     item['conflict_rate']]))
                b3 = np.hstack((b3, [item['model_size'], item['total_rule_length'], item['max_length']]))
                model_name = np.hstack((model_name, key))
            else:
                b1 = np.vstack((b1, item['consistency']))
                b2 = np.vstack((b2, [item['coverage_rate'], item['class_overlap_rate'], item['overlap_rate'],
                                     item['conflict_rate']]))
                b3 = np.vstack((b3, [item['model_size'], item['total_rule_length'], item['max_length']]))
                model_name = np.vstack((model_name, key))

        b = [b1, b2, b3]
        # 指标类型，1为收益型，-1为损失型
        s1 = np.array([1])
        s2 = np.array([1, 1, -1, -1])
        s3 = np.array([-1, -1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0]])
        c3 = np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

        criteria, c = ahp_rule(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("complexity1_3"), evaModelConfig.get("complexity1_4"),
                               evaModelConfig.get("complexity2_3"), evaModelConfig.get("complexity2_4"),
                               evaModelConfig.get("complexity3_4"), evaModelConfig.get("clarity1_2"),
                               evaModelConfig.get("clarity1_3"), evaModelConfig.get("clarity2_3"))
        if criteria == "numbererror":
            return c
        topsis = TOPSIS(b, s, criteria, c)
        try:
            result, w2, w1 = topsis.run()
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["覆盖率", "类覆盖率", "重叠率", "矛盾率"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["模型大小", "规则模型总长度", "规则最大长度"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = json.loads(result.to_json(orient='records'))
        XAIname_para = ""
        for i in range(len(list)):
            list[i]["model_name"] = model_name[i][0]
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["consistency"] = indexOriginList["consistency"][i]
            list[num_in_eva]["coverage_rate"] = indexOriginList["coverage_rate"][i]
            list[num_in_eva]["class_overlap_rate"] = indexOriginList["class_overlap_rate"][i]
            list[num_in_eva]["overlap_rate"] = indexOriginList["overlap_rate"][i]
            list[num_in_eva]["conflict_rate"] = indexOriginList["conflict_rate"][i]
            list[num_in_eva]["model_size"] = indexOriginList["model_size"][i]
            list[num_in_eva]["total_rule_length"] = indexOriginList["total_rule_length"][i]
            list[num_in_eva]["max_length"] = indexOriginList["max_length"][i]

        list.sort(key=lambda k: k.get('C'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1
        text_param.append(XAIname_para)  # 6
        text_param.append(list[0]["model_name"])  # 7
        text_param.append(format(list[0]["C"], '.3f'))  # 8

        # session['result'] = json.loads(result.to_json())
        return list, w1_param, w2_param, w3_param, text_param
        # return render_template("predict_topsis_ahp_RULE.html", list=list, w2=w2_param, w1=w1_param, w3=w3_param,
        #                        text_param=text_param, evaname=session["evamodel"])


# @app.route('/predict_topsis_ahp_entropy', methods=['POST', 'GET'])
def predict_topsis_ahp_entropy(proxy_type, indicator_dict, evaModelConfig):
    if proxy_type == "tree":
        b1 = np.array([])
        b2 = np.array([])
        b3 = np.array([])
        model_name = np.array([])
        for key, item in indicator_dict.items():
            if b1.shape[0] == 0:
                b1 = np.hstack((b1, item['AUC']))
                b2 = np.hstack((b2, [item['APL'], item['node count']]))
                b3 = np.hstack((b3, [item['duplicate_subtree'], item['duplicate_attr']]))
                model_name = np.hstack((model_name, key))
            else:
                b1 = np.vstack((b1, item['AUC']))
                b2 = np.vstack((b2, [item['APL'], item['node count']]))
                b3 = np.vstack((b3, [item['duplicate_subtree'], item['duplicate_attr']]))
                model_name = np.vstack((model_name, key))
        b = [b1, b2, b3]
        s1 = np.array([1])
        s2 = np.array([-1, -1])
        s3 = np.array([-1, -1])
        s = [s1, s2, s3]
        # 一级指标的AHP评价矩阵
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = None
        c2 = None
        c3 = None

        criteria, c = ahp_tree(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("clarity1_2"))
        if criteria == "numbererror":
            return c
        topsis = TOPSIS_AHP_Entropy(b, s, criteria)

        try:
            result, w2, w1 = topsis.run()
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"

        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["APL", "节点数量"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["重复子树比例", "重复子树比例"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = json.loads(result.to_json(orient='records'))
        XAIname_para = ""
        for i in range(len(list)):
            list[i]["model_name"] = model_name[i][0]
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            num_in_eva = -1
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            if num_in_eva != -1:
                list[num_in_eva]["AUC"] = indexOriginList["AUC"][i]
                list[num_in_eva]["APL"] = indexOriginList["APL"][i]
                list[num_in_eva]["node count"] = indexOriginList["node count"][i]
                list[num_in_eva]["duplicate_subtree"] = indexOriginList["duplicate_subtree"][i]
                list[num_in_eva]["duplicate_attr"] = indexOriginList["duplicate_attr"][i]

        list.sort(key=lambda k: k.get('C'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1
        text_param.append(XAIname_para)  # 6
        text_param.append(list[0]["model_name"])  # 7
        text_param.append(format(list[0]["C"], '.3f'))  # 8

        # session['result'] = json.loads(result.to_json())
        return list, w1_param, w2_param, w3_param, text_param

    elif proxy_type == "rule":
        b1 = np.array([])
        b2 = np.array([])
        b3 = np.array([])
        model_name = np.array([])

        for key, item in indicator_dict.items():
            if b1.shape[0] == 0:
                b1 = np.hstack((b1, item['consistency']))
                b2 = np.hstack((b2, [item['coverage_rate'], item['class_overlap_rate'], item['overlap_rate'],
                                     item['conflict_rate']]))
                b3 = np.hstack((b3, [item['model_size'], item['total_rule_length'], item['max_length']]))
                model_name = np.hstack((model_name, key))
            else:
                b1 = np.vstack((b1, item['consistency']))
                b2 = np.vstack((b2, [item['coverage_rate'], item['class_overlap_rate'], item['overlap_rate'],
                                     item['conflict_rate']]))
                b3 = np.vstack((b3, [item['model_size'], item['total_rule_length'], item['max_length']]))
                model_name = np.vstack((model_name, key))
        # 二级指标矩阵 顺序mdrl brs sbrl
        # b1 = np.array([[0.96,2.528, 7,0.00001, 0.00001], [0.948,3.093, 17,0.3529, 0.0645], [0.919,4.347, 31,0.00001, 0.1508]])
        # b = [b1]
        # 指标类型，1为收益型，-1为损失型
        # b1 = np.array([[0.96], [0.948], [0.919]])
        # b2 = np.array([[2.528, 7],
        #                [3.093, 17],
        #                [4.347, 31]])
        # b3 = np.array([[0, 0],
        #                [0.3529, 0.0645],
        #                [0, 0.1508]])
        b = [b1, b2, b3]
        # s1 = np.array([1,-1,-1,-1,-1])
        # s = [s1]
        s1 = np.array([1])
        s2 = np.array([1, 1, -1, -1])
        s3 = np.array([-1, -1, -1])
        s = [s1, s2, s3]

        # 一级指标的AHP评价矩阵
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = None
        c2 = None
        c3 = None
        criteria, c = ahp_rule(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("complexity1_3"), evaModelConfig.get("complexity1_4"),
                               evaModelConfig.get("complexity2_3"), evaModelConfig.get("complexity2_4"),
                               evaModelConfig.get("complexity3_4"), evaModelConfig.get("clarity1_2"),
                               evaModelConfig.get("clarity1_3"), evaModelConfig.get("clarity2_3"))

        if criteria == "numbererror":
            return c

        topsis = TOPSIS_AHP_Entropy(b, s, criteria)

        try:
            result, w2, w1 = topsis.run()
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"

        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["覆盖率", "类覆盖率", "重叠率", "矛盾率"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["模型大小", "规则模型总长度", "规则最大长度"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = json.loads(result.to_json(orient='records'))
        XAIname_para = ""
        for i in range(len(list)):
            list[i]["model_name"] = model_name[i][0]
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            num_in_eva = -1
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            if num_in_eva != -1:
                list[num_in_eva]["consistency"] = indexOriginList["consistency"][i]
                list[num_in_eva]["coverage_rate"] = indexOriginList["coverage_rate"][i]
                list[num_in_eva]["class_overlap_rate"] = indexOriginList["class_overlap_rate"][i]
                list[num_in_eva]["overlap_rate"] = indexOriginList["overlap_rate"][i]
                list[num_in_eva]["conflict_rate"] = indexOriginList["conflict_rate"][i]
                list[num_in_eva]["model_size"] = indexOriginList["model_size"][i]
                list[num_in_eva]["total_rule_length"] = indexOriginList["total_rule_length"][i]
                list[num_in_eva]["max_length"] = indexOriginList["max_length"][i]

        list.sort(key=lambda k: k.get('C'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1
        text_param.append(XAIname_para)  # 6
        text_param.append(list[0]["model_name"])  # 7
        text_param.append(format(list[0]["C"], '.3f'))  # 8

        return list, w1_param, w2_param, w3_param, text_param
    else:
        return "error"


# @app.route('/predict_fuzzy_ahp', methods=['POST', 'GET'])
def predict_fuzzy_ahp(proxy_type, indicator_dict, evaModelConfig):
    if proxy_type == "tree":
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])
        c3 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])

        criteria, c = ahp_tree(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("clarity1_2"))

        if criteria == "numbererror":
            return c
        # 专家打分
        t1 = [1]  # t for 指标类型，1越大越好，0越小越好
        s1 = [[1, 0.7, 0.5, 0.3]]

        t2 = [0, 0]
        s2 = [[1, 2, 4, 6],
              [6, 12, 24, 30]]

        t3 = [0, 0]
        s3 = [[0.1, 0.2, 0.5, 0.7],
              [0.1, 0.2, 0.5, 0.7]]

        t = [t1, t2, t3]
        s = [s1, s2, s3]

        # 模糊评价等级
        scores = np.array([90, 80, 65, 30])

        result = []
        F = []
        for key, item in indicator_dict.items():
            i1 = []
            i2 = []
            i3 = []
            i1.append(item['AUC'])
            i2.append(item['APL'])
            i2.append(item['node count'])
            i3.append(item['duplicate_subtree'])
            i3.append(item['duplicate_attr'])
            i = [i1, i2, i3]
            try:
                a, w2, w1, f = AHP(criteria, c, t, s, i, scores, key).run()
            except Exception as e:
                return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
            result.append({"model_name": key, "score": round(a, 3), "F": str([round(i, 3) for i in f.tolist()])})
            F.append(f)
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["APL", "节点数量"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["重复子树比例", "重复子树比例"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = result

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["AUC"] = indexOriginList["AUC"][i]
            list[num_in_eva]["APL"] = indexOriginList["APL"][i]
            list[num_in_eva]["node count"] = indexOriginList["node count"][i]
            list[num_in_eva]["duplicate_subtree"] = indexOriginList["duplicate_subtree"][i]
            list[num_in_eva]["duplicate_attr"] = indexOriginList["duplicate_attr"][i]

        XAIname_para = ""
        for i in range(len(list)):
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","
        list.sort(key=lambda k: k.get('score'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1

        text_param.append(str(list[0]["F"]))  # 6
        text_param.append(list[0]["score"])  # 7

        return_str = ""
        if len(list) > 1:
            return_str += "同理得到"
            for i in range(1, len(list)):
                return_str += "{}的综合得分为{},".format(list[i]["model_name"], list[i]["score"])
            return_str += "{}的解释性最好。".format(list[0]["model_name"])

        text_param.append(return_str)  # 8
        return list, w1_param, w2_param, w3_param, text_param

    elif proxy_type == "rule":
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0]])
        c3 = np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

        criteria, c = ahp_rule(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("complexity1_3"), evaModelConfig.get("complexity1_4"),
                               evaModelConfig.get("complexity2_3"), evaModelConfig.get("complexity2_4"),
                               evaModelConfig.get("complexity3_4"), evaModelConfig.get("clarity1_2"),
                               evaModelConfig.get("clarity1_3"), evaModelConfig.get("clarity2_3"))

        if criteria == "numbererror":
            return c
        # 专家打分
        t1 = [1]  # t for 指标类型，1越大越好，0越小越好
        s1 = [[1, 0.9, 0.7, 0.5]]  # s 是专家打分，确定隶属函数的区间

        t2 = [1, 1, 0, 0]
        s2 = [[1, 0.7, 0.5, 0.3],
              [1, 0.7, 0.5, 0.3],
              [0, 0.3, 0.5, 0.7],
              [0, 0.1, 0.2, 0.3]]

        t3 = [0, 0, 0]
        s3 = [[3, 5, 7, 10],
              [3, 10, 20, 40],
              [3, 5, 7, 10]]

        t = [t1, t2, t3]
        s = [s1, s2, s3]
        # 模糊评价等级
        scores = np.array([90, 80, 65, 30])

        result = []
        F = []

        for key, item in indicator_dict.items():
            i1 = []
            i2 = []
            i3 = []
            i1.append(item['consistency'])
            i2.append(item['coverage_rate'])
            i2.append(item['class_overlap_rate'])
            i2.append(item['overlap_rate'])
            i2.append(item['conflict_rate'])
            i3.append(item['model_size'])
            i3.append(item['total_rule_length'])
            i3.append(item['max_length'])
            i = [i1, i2, i3]
            try:
                a, w2, w1, f = AHP(criteria, c, t, s, i, scores, key).run()
            except Exception as e:
                return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
            result.append({"model_name": key, "score": round(a, 3), "F": str([round(i, 3) for i in f.tolist()])})
            F.append(f)
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["覆盖率", "类覆盖率", "重叠率", "矛盾率"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["模型大小", "规则模型总长度", "规则最大长度"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = result

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["consistency"] = indexOriginList["consistency"][i]
            list[num_in_eva]["coverage_rate"] = indexOriginList["coverage_rate"][i]
            list[num_in_eva]["class_overlap_rate"] = indexOriginList["class_overlap_rate"][i]
            list[num_in_eva]["overlap_rate"] = indexOriginList["overlap_rate"][i]
            list[num_in_eva]["conflict_rate"] = indexOriginList["conflict_rate"][i]
            list[num_in_eva]["model_size"] = indexOriginList["model_size"][i]
            list[num_in_eva]["total_rule_length"] = indexOriginList["total_rule_length"][i]
            list[num_in_eva]["max_length"] = indexOriginList["max_length"][i]

        XAIname_para = ""
        for i in range(len(list)):
            XAIname_para += list[i]["model_name"]
            if i != len(list) - 1:
                XAIname_para += ","
        list.sort(key=lambda k: k.get('score'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1

        text_param.append(str(list[0]["F"]))  # 6
        text_param.append(list[0]["score"])  # 7
        text_param.append(list[0]["model_name"])  # 8
        return_str = ""
        if len(list) > 1:
            return_str += "同理得到"
            for i in range(1, len(list)):
                return_str += "{}的综合得分为{},".format(list[i]["model_name"], list[i]["score"])
            return_str += "{}的解释性最好。".format(list[0]["model_name"])

        text_param.append(return_str)  # 9
        return list, w1_param, w2_param, w3_param, text_param
        # return render_template("predict_fuzzy_ahp_RULE.html", list=list, w2=w2_param, w1=w1_param, w3=w3_param,
        #                        text_param=text_param, evaname=session["evamodel"])
    else:
        return "error"


# @app.route('/predict_rsr_ahp', methods=['POST', 'GET'])
def predict_rsr_ahp(proxy_type, indicator_dict, evaModelConfig):
    if proxy_type == "tree":
        AUCList = []
        APLList = []
        nodeList = []
        DsubtreeList = []
        DattrList = []
        indexList = []
        for key, item in indicator_dict.items():
            AUCList.append(item['AUC'])
            APLList.append(item['APL'])
            nodeList.append(item['node count'])
            DsubtreeList.append(item['duplicate_subtree'])
            DattrList.append(item['duplicate_attr'])
            indexList.append(key)
        data = pd.DataFrame({'AUC': AUCList,
                             'APL': APLList,
                             'node count': nodeList,
                             'duplicate_subtree': DsubtreeList,
                             'duplicate_attr': DattrList},
                            index=indexList,
                            columns=['AUC', 'APL', 'node count', 'duplicate_subtree', 'duplicate_attr'])
        data["APL"] = 1 / data["APL"]
        data["node count"] = 1 / data["node count"]
        data["duplicate_subtree"] = 1 / data["duplicate_subtree"]
        data["duplicate_attr"] = 1 / data["duplicate_attr"]

        s1 = np.array([1])
        s2 = np.array([-1, -1])
        s3 = np.array([-1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])
        c3 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])

        criteria, c = ahp_tree(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("clarity1_2"))
        if criteria == "numbererror":
            return c
        try:
            result, distribution, w2, w1, w3 = rsr(data, s, criteria, c)
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["APL", "节点数量"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["重复子树比例", "重复子树比例"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        list = result.to_dict("records")
        index_list = result.index.tolist()
        for key, item in enumerate(index_list):
            list[key]["model_name"] = item
        for item in list:
            item["RSR"] = round(item["RSR"], 4)
            item["Probit"] = round(item["Probit"], 4)
            item["RSR Regression"] = round(item["RSR Regression"], 4)

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["AUC"] = indexOriginList["AUC"][i]
            list[num_in_eva]["APL"] = indexOriginList["APL"][i]
            list[num_in_eva]["node count"] = indexOriginList["node count"][i]
            list[num_in_eva]["duplicate_subtree"] = indexOriginList["duplicate_subtree"][i]
            list[num_in_eva]["duplicate_attr"] = indexOriginList["duplicate_attr"][i]

        # 所有的模型名字
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["model_name"])
            if i != len(list) - 1:
                return_str += "、"
        text_param.append(return_str)  # 6
        # list之后为依据rsr regression排序的
        list.sort(key=lambda k: k.get('RSR Regression'), reverse=True)
        # 所有的模型排名
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["model_name"])
            if i != len(list) - 1:
                return_str += ">"
        text_param.append(return_str)  # 7
        # 所有的分档
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["Level"])
            if i != len(list) - 1:
                return_str += "、"
        text_param.append(return_str)  # 8
        # 最好的模型
        text_param.append(list[0]["model_name"])  # 8
        return list, w1_param, w2_param, w3_param, text_param

    elif proxy_type == "rule":
        consistencyList = []
        coverage_rateList = []
        class_overlap_rateList = []
        overlap_rateList = []
        conflict_rateList = []
        model_sizeList = []
        total_rule_lengthList = []
        max_lengthList = []
        indexList = []
        for key, item in indicator_dict.items():
            consistencyList.append(item['consistency'])
            coverage_rateList.append(item['coverage_rate'])
            class_overlap_rateList.append(item['class_overlap_rate'])
            overlap_rateList.append(item['overlap_rate'])
            conflict_rateList.append(item['conflict_rate'])
            model_sizeList.append(item['model_size'])
            total_rule_lengthList.append(item['total_rule_length'])
            max_lengthList.append(item['max_length'])
            indexList.append(key)
        data = pd.DataFrame({'consistency': consistencyList,
                             'coverage_rate': coverage_rateList,
                             'class_overlap_rate': class_overlap_rateList,
                             'overlap_rate': overlap_rateList,
                             'conflict_rate': conflict_rateList,
                             'model_size': model_sizeList,
                             'total_rule_length': total_rule_lengthList,
                             'max_length': max_lengthList},
                            index=indexList,
                            columns=['consistency', 'coverage_rate', 'class_overlap_rate', 'overlap_rate',
                                     'conflict_rate', 'model_size', 'total_rule_length', 'max_length'])
        data["overlap_rate"] = 1 / data["overlap_rate"]
        data["conflict_rate"] = 1 / data["conflict_rate"]
        data["model_size"] = 1 / data["model_size"]
        data["total_rule_length"] = 1 / data["total_rule_length"]
        data["max_length"] = 1 / data["max_length"]

        # 指标类型，1为收益型，-1为损失型
        s1 = np.array([1])
        s2 = np.array([1, 1, -1, -1])
        s3 = np.array([-1, -1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0]])
        c3 = np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

        criteria, c = ahp_rule(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("complexity1_3"), evaModelConfig.get("complexity1_4"),
                               evaModelConfig.get("complexity2_3"), evaModelConfig.get("complexity2_4"),
                               evaModelConfig.get("complexity3_4"), evaModelConfig.get("clarity1_2"),
                               evaModelConfig.get("clarity1_3"), evaModelConfig.get("clarity2_3"))

        if criteria == "numbererror":
            return c
        try:
            result, distribution, w2, w1, w3 = rsr(data, s, criteria, c)
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["覆盖率", "类覆盖率", "重叠率", "矛盾率"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["模型大小", "规则模型总长度", "规则最大长度"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        ####
        list = result.to_dict("records")
        index_list = result.index.tolist()
        for key, item in enumerate(index_list):
            list[key]["model_name"] = item
        for item in list:
            item["RSR"] = round(item["RSR"], 4)
            item["Probit"] = round(item["Probit"], 4)
            item["RSR Regression"] = round(item["RSR Regression"], 4)

        indexOriginList = getIndexList(proxy_type, indicator_dict)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["X1：consistency"] = indexOriginList["consistency"][i]
            list[num_in_eva]["X2：coverage_rate"] = indexOriginList["coverage_rate"][i]
            list[num_in_eva]["X3：class_overlap_rate"] = indexOriginList["class_overlap_rate"][i]
            list[num_in_eva]["X4：overlap_rate"] = indexOriginList["overlap_rate"][i]
            list[num_in_eva]["X5：conflict_rate"] = indexOriginList["conflict_rate"][i]
            list[num_in_eva]["X6：model_size"] = indexOriginList["model_size"][i]
            list[num_in_eva]["X7：total_rule_length"] = indexOriginList["total_rule_length"][i]
            list[num_in_eva]["X8：max_length"] = indexOriginList["max_length"][i]

        # 所有的模型名字
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["model_name"])
            if i != len(list) - 1:
                return_str += "、"
        text_param.append(return_str)  # 6

        # list之后为依据rsr regression排序的
        list.sort(key=lambda k: k.get('RSR Regression'), reverse=True)

        # 所有的模型排名
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["model_name"])
            if i != len(list) - 1:
                return_str += ">"
        text_param.append(return_str)  # 7
        # 所有的分档
        return_str = ""
        for i in range(0, len(list)):
            return_str += "{}".format(list[i]["Level"])
            if i != len(list) - 1:
                return_str += "、"
        text_param.append(return_str)  # 8
        # 最好的模型
        text_param.append(list[0]["model_name"])  # 8

        return list, w1_param, w2_param, w3_param, text_param
    else:
        return "error"


# @app.route('/predict_grey_ahp', methods=['POST', 'GET'])
def predict_grey_ahp(proxy_type, indicator_dict, evaModelConfig):
    if proxy_type == "tree":
        # 0、数据初始化
        index = []
        data = []
        for key, item in indicator_dict.items():
            data_tmp = []
            data_tmp.append(item['AUC'])
            data_tmp.append(item['APL'])
            data_tmp.append(item['node count'])
            data_tmp.append(item['duplicate_subtree'])
            data_tmp.append(item['duplicate_attr'])
            data.append(data_tmp)
            index.append(key)
        columns = ['AUC', 'APL', 'node count', 'duplicate_subtree', 'duplicate_attr']
        # duplicate_subtree
        # ":0,"
        # duplicate_attr
        # data = [[0.7,9,8,0.8,0.4],[0.7,9,8,0.8,0.4],[0.6,8,9,0.7,0.3],[0.8,10,7,0.9,0.5]]
        # columns=['m1','m2','m3','m4','m5']
        # index=["model1","model2","model3","model4"]

        X = pd.DataFrame(data=data, index=index, columns=columns)

        s1 = np.array([1])
        s2 = np.array([-1, -1])
        s3 = np.array([-1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])
        c3 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])

        criteria, c = ahp_tree(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("clarity1_2"))
        if criteria == "numbererror":
            return c
        try:
            result, result_origin, ksi, w2, w1, w3 = grs_count(X, s, criteria, c, columns)
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["APL", "节点数量"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["重复子树比例", "重复子树比例"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        ####
        indexOriginList = getIndexList(proxy_type, indicator_dict)
        list = ksi.to_dict("records")
        index_list = ksi.index.tolist()
        for key, item in enumerate(index_list):
            list[key]["model_name"] = item
            list[key]["related_score"] = round(result_origin[key], 4)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["APL"] = indexOriginList["APL"][i]
            list[num_in_eva]["AUC"] = indexOriginList["AUC"][i]
            list[num_in_eva]["node count"] = indexOriginList["node count"][i]
            list[num_in_eva]["duplicate_subtree"] = indexOriginList["duplicate_subtree"][i]
            list[num_in_eva]["duplicate_attr"] = indexOriginList["duplicate_attr"][i]

        list.sort(key=lambda k: k.get('related_score'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1

        text_param.append(len(list))  # 6
        return_str = ""
        for key, item in enumerate(list):
            return_str += item["model_name"] + "（关联度为：" + str(item["related_score"]) + ")"
            if key != len(list) - 1:
                return_str += ","
        text_param.append(return_str)  # 7

        return list, w1_param, w2_param, w3_param, text_param

    elif proxy_type == "rule":
        index = []
        data = []
        for key, item in indicator_dict.items():
            data_tmp = []
            data_tmp.append(item['consistency'])
            data_tmp.append(item['coverage_rate'])
            data_tmp.append(item['class_overlap_rate'])
            data_tmp.append(item['overlap_rate'])
            data_tmp.append(item['conflict_rate'])
            data_tmp.append(item['model_size'])
            data_tmp.append(item['total_rule_length'])
            data_tmp.append(item['max_length'])
            data.append(data_tmp)
            index.append(key)
        columns = ['consistency', 'coverage_rate', 'class_overlap_rate', 'overlap_rate', 'conflict_rate', 'model_size',
                   'total_rule_length', 'max_length']

        # data = [[0.7,9,8,0.8,0.4],[0.7,9,8,0.8,0.4],[0.6,8,9,0.7,0.3],[0.8,10,7,0.9,0.5]]
        # columns=['m1','m2','m3','m4','m5']
        # index=["model1","model2","model3","model4"]

        X = pd.DataFrame(data=data, index=index, columns=columns)

        # 指标类型，1为收益型，-1为损失型
        s1 = np.array([1])
        s2 = np.array([1, 1, -1, -1])
        s3 = np.array([-1, -1, -1])
        # s4 = np.array([1, 1])
        s = [s1, s2, s3]
        # AHP 评价矩阵
        # 准则层重要性矩阵
        criteria = np.array([[1, 3, 4],
                             [1 / 3, 1, 2],
                             [1 / 4, 1 / 2, 1]])
        # 方案层
        c1 = np.array([[1.0]])
        c2 = np.array([[1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0]])
        c3 = np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0]])

        criteria, c = ahp_rule(criteria, evaModelConfig.get("index1_2"), evaModelConfig.get("index1_3"),
                               evaModelConfig.get("index2_3"), c1, c2, c3, evaModelConfig.get("complexity1_2"),
                               evaModelConfig.get("complexity1_3"), evaModelConfig.get("complexity1_4"),
                               evaModelConfig.get("complexity2_3"), evaModelConfig.get("complexity2_4"),
                               evaModelConfig.get("complexity3_4"), evaModelConfig.get("clarity1_2"),
                               evaModelConfig.get("clarity1_3"), evaModelConfig.get("clarity2_3"))

        if criteria == "numbererror":
            return c
        try:
            result, result_origin, ksi, w2, w1, w3 = grs_count(X, s, criteria, c, columns)
        except Exception as e:
            return "CR>0.1，一致性检验不通过，请检查判断矩阵的填写是否合理。"
        w2_param = w2.copy()
        w1_param = w1.copy()
        w3_param = w2.copy()

        w3_param[0] = w3_param[0] * w1_param[0]
        w3_param[1] = w3_param[1] * w1_param[1]
        w3_param[2] = w3_param[2] * w1_param[2]

        w1_param = ["%s%%" % (format(i * 100, '.2f')) for i in w1_param]
        w2_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[0]]
        w2_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[1]]
        w2_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w2_param[2]]
        w3_param[0] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[0]]
        w3_param[1] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[1]]
        w3_param[2] = ["%s%%" % (format(i * 100, '.2f')) for i in w3_param[2]]

        text_param = []

        w1_list = w1.tolist()
        w1_list_max = max(w1_list)
        w1_list_max_index = w1_list.index(max(w1_list))
        LAYER1_INDEX_MAP = ["一致性", "复杂性", "明确性"]

        text_param.append(LAYER1_INDEX_MAP[w1_list_max_index])  # 0
        text_param.append("%s%%" % (format(w1_list_max * 100, '.2f')))  # 1

        w2_complexity_list = w2[1].tolist()
        w2_complexity_list_max = max(w2_complexity_list)
        w2_complexity_list_max_index = w2_complexity_list.index(max(w2_complexity_list))
        LAYER2_COMPLEXITY_INDEX_MAP = ["覆盖率", "类覆盖率", "重叠率", "矛盾率"]

        text_param.append(LAYER2_COMPLEXITY_INDEX_MAP[w2_complexity_list_max_index])  # 2
        text_param.append("%s%%" % (format(w2_complexity_list_max * 100, '.2f')))  # 3

        w2_clarity_list = w2[2].tolist()
        w2_clarity_list_max = max(w2_clarity_list)
        w2_clarity_list_max_index = w2_clarity_list.index(max(w2_clarity_list))
        LAYER2_CLARITY_INDEX_MAP = ["模型大小", "规则模型总长度", "规则最大长度"]

        text_param.append(LAYER2_CLARITY_INDEX_MAP[w2_clarity_list_max_index])  # 4
        text_param.append("%s%%" % (format(w2_clarity_list_max * 100, '.2f')))  # 5

        ####
        indexOriginList = getIndexList(proxy_type, indicator_dict)
        list = ksi.to_dict("records")
        index_list = ksi.index.tolist()
        for key, item in enumerate(index_list):
            list[key]["model_name"] = item
            list[key]["related_score"] = round(result_origin[key], 4)
        for i in range(len(indicator_dict)):
            # 获取origin顺序的指标
            model_name = indexOriginList['model_name'][i]
            # 获取对应XAI model评估指标在评估模型返回的结果中的序号
            for j in range(len(indicator_dict)):
                if list[j]["model_name"] == model_name:
                    num_in_eva = j
                    break
            list[num_in_eva]["consistency"] = indexOriginList["consistency"][i]
            list[num_in_eva]["coverage_rate"] = indexOriginList["coverage_rate"][i]
            list[num_in_eva]["class_overlap_rate"] = indexOriginList["class_overlap_rate"][i]
            list[num_in_eva]["overlap_rate"] = indexOriginList["overlap_rate"][i]
            list[num_in_eva]["conflict_rate"] = indexOriginList["conflict_rate"][i]
            list[num_in_eva]["model_size"] = indexOriginList["model_size"][i]
            list[num_in_eva]["total_rule_length"] = indexOriginList["total_rule_length"][i]
            list[num_in_eva]["max_length"] = indexOriginList["max_length"][i]
        list.sort(key=lambda k: k.get('related_score'), reverse=True)
        for i in range(len(list)):
            list[i]["rank"] = i + 1

        text_param.append(len(list))  # 6
        return_str = ""
        for key, item in enumerate(list):
            return_str += item["model_name"] + "（关联度为：" + str(item["related_score"]) + ")"
            if key != len(list) - 1:
                return_str += ","
        text_param.append(return_str)  # 7
        return list, w1_param, w2_param, w3_param, text_param
    else:
        pass
