import warnings

import pandas as pd
import numpy as np
# import statsmodels.api as sm
from scipy.stats import norm

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)


def cal_weights(input_matrix):
    input_matrix = np.array(input_matrix)
    n, n1 = input_matrix.shape
    assert n == n1, '不是一个方阵'
    for i in range(n):
        for j in range(n):
            if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                raise ValueError('不是反互对称矩阵')

    eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

    max_idx = np.argmax(eigenvalues)
    max_eigen = eigenvalues[max_idx].real
    eigen = eigenvectors[:, max_idx].real
    eigen = eigen / eigen.sum()

    if n > 9:
        CR = None
        warnings.warn('无法判断一致性')
    elif n == 1:
        CR = 0
    else:
        CI = (max_eigen - n) / (n - 1)
        CR = CI / RI[n - 1]
        if CR > 0.1:
            raise Exception("CR>0.1，一致性检验不通过")
    return max_eigen, CR, eigen


def rsr(data, s, criteria, c, weight=None, threshold=None, full_rank=True):
    # 0、计算权重
    W = []
    for i in range(len(c)):
        max_eigen, CR, w = cal_weights(c[i])  # 权重
        W.append(w)
        # print(i)
    # print(W)
    max_eigen, CR, weight = cal_weights(criteria)  # AHP计算一级指标的权重
    # print(weight)


    # 1、由原始数据进行计算秩值（整次/非整次）
    Result = pd.DataFrame()
    n, m = data.shape
    if full_rank:
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = data.iloc[:, i].rank(method="dense")
    else:
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = 1 + (n - 1) * (data.iloc[:, i].max() - data.iloc[:, i]) / (
                        data.iloc[:, i].max() - data.iloc[:, i].min())
    # print("***由原始数据进行计算秩值******")
    # print(Result)

    # 2、计算得到RSR值和RSR值排名
    weight_tmp = W.copy()
    weight_tmp[0] = W[0] * weight[0]
    weight_tmp[1] = W[1] * weight[1]
    weight_tmp[2] = W[2] * weight[2]
    weight_combine = []
    for item in weight_tmp:
        weight_combine.extend(item.tolist())
    # print(weight_combine)
    Result['RSR'] = (Result.iloc[:, 1::2] * weight_combine).sum(axis=1) / n
    Result['RSR_Rank'] = Result['RSR'].rank(ascending=False)
    # print("****计算得到RSR值和RSR值排名*****")
    # print(Result)


    # 3、列出RSR的分布表格情况并且得到Probit值
    RSR = Result['RSR']
    RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))
    Distribution = pd.DataFrame(index=sorted(RSR.unique()))
    Distribution['f'] = RSR.value_counts().sort_index()
    Distribution['Σ f'] = Distribution['f'].cumsum()
    Distribution[r'\bar{R} f'] = [RSR_RANK_DICT[i] for i in Distribution.index]
    Distribution[r'\bar{R}/n*100%'] = Distribution[r'\bar{R} f'] / n
    Distribution.iat[-1, -1] = 1 - 1 / (4 * n)
    Distribution['Probit'] = 5 - norm.isf(Distribution.iloc[:, -1])
    # print("****列出RSR的分布表格情况并且得到Probit值*****")
    # print(Distribution)


    # 4、计算回归方差并进行回归分析
    # print("****计算回归方差并进行回归分析*****")
    r0 = np.polyfit(Distribution['Probit'], Distribution.index, deg=1)
    # print(sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit'])).fit().summary())
    # if r0[1] > 0:
    #     print(f"回归直线方程为：y = {r0[0]} Probit + {r0[1]}")
    # else:
    #     print(f"回归直线方程为：y = {r0[0]} Probit - {abs(r0[1])}")
    # print("r0=",r0)


    # 5、代入回归方程并分档排序
    Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit'])
    Result['RSR Regression'] = np.polyval(r0, Result['Probit'])
    threshold = np.polyval(r0, [2, 4, 6, 8]) if threshold is None else np.polyval(r0, threshold)
    Result['Level'] = pd.cut(Result['RSR Regression'], threshold, labels=range(len(threshold) - 1, 0, -1))
    # print("***代入回归方程并分档排序******")
    # print("threshold=",threshold)
    # print(Result)

    return Result, Distribution, W, weight, weight_combine


def rsrAnalysis(data, file_name=None, **kwargs):
    Result, Distribution = rsr(data, **kwargs)
    file_name = 'RSR 分析结果报告.xlsx' if file_name is None else file_name + '.xlsx'
    Excel_Writer = pd.ExcelWriter(file_name)
    Result.to_excel(Excel_Writer, '综合评价结果')
    Result.sort_values(by='Level', ascending=False).to_excel(Excel_Writer, '分档排序结果')
    Distribution.to_excel(Excel_Writer, 'RSR分布表')
    Excel_Writer.save()
    return Result, Distribution


if __name__ == '__main__':
    columns = ['AUC', 'APL', 'nodeNumber', 'duplicateSubTree', 'duplicateSubAttr']
    index = ["tree1", "tree2", "tree3"]
    data = pd.DataFrame({'AUC': [0.96, 0.948, 0.919],
                         'APL': [2.528, 3.093, 4.347],
                         'node': [7, 17, 31],
                         'dTree': [0, 0.3529, 0],
                         'dAttr': [0, 0.0645, 0.1508]},
                        index=["tree1", "tree2", "tree3"], columns=['AUC', 'APL', 'node', 'dTree', 'dAttr'])
    data["APL"] = 1 / data["APL"]
    data["node"] = 1 / data["node"]
    data["dTree"] = 1 / data["dTree"]
    data["dAttr"] = 1 / data["dAttr"]

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
    c1 = np.array([[1]])
    c2 = np.array([[1, 1],
                   [1, 1]])
    c3 = np.array([[1, 1],
                   [1, 1]])
    # c4 = np.array([[1, 3],
    #    [1/3, 1]])
    c = [c1, c2, c3]

    result, distribution, w2, w1, w3 = rsr(data, s, criteria, c)
    # print(type(result))
    # print("*****")
    # print(result)
    # print("*****")
    # print(distribution)
    # print("*****")
    # print(result['Level'])
    # print("*****")
    # print(result['X1：AUC'])
    # print(type(result['X1：AUC']))
    # print("*****")
    # print(result['R1：AUC'])

    # rsrAnalysis(data)
