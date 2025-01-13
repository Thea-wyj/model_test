import argparse
import json

import pandas as pd
import numpy as np
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from keras.models import load_model
from sklearn.metrics import auc
import tensorflow as tf
import h5py
from tensorflow.keras.optimizers import legacy
from model.res_net.resNet import Net


# 计算不忠度
def get_faithfulness(model, sequences, exp, label):
    # 用前一时刻的值替换一些贡献突出的时刻的值，并观察模型性能的变化，计算两者差的AUC值
    prediction_or = model.predict(sequences)[0][label]
    item = sequences[0]
    auc_scores = []
    for i in range(exp.shape[1]):
        item_masked = item.copy()
        exp_i = exp[:, i]
        item_i = item_masked[:, i]
        thresholds = list(range(0, 110, 10))
        model_scores = []
        for threshold in thresholds:
            # 计算前n%的元素数量
            num_elements_top_n = int(threshold / 100 * len(exp_i))

            # 获取exp内前n%贡献度的值的索引
            top_n_indices = sorted(range(len(exp_i)), key=lambda j: exp_i[j], reverse=True)[:num_elements_top_n]

            # 在item中用前一个值代替这些位置的值，对于第一个元素使用None
            for idx in top_n_indices:
                item_i[idx] = item_i[idx - 1] if idx > 0 else item_i[0]

            for row in range(len(item_masked)):
                item_masked[row][i] = item_i[row]

            results = model.predict(item_masked[None, :])[0][label]
            model_scores.append(np.abs(prediction_or - results))
        print(thresholds, model_scores)
        auc_scores.append(auc(thresholds, model_scores))
    return np.mean(auc_scores)


def get_class_sensitivity(exp, anti_exp):
    """
       consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
       数值范围[-1,1],数值越大越相似。
       :param tensor1:
       :param tensor2:
       :return:
       """
    tensor1 = tf.convert_to_tensor(exp)
    tensor2 = tf.convert_to_tensor(anti_exp)
    # 求模长
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))

    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
    cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
    # 转化为float
    cosin = cosin.numpy()

    return (cosin + 1) / 2


# 绝对值差异
def abs_diff(x, y):
    diff = abs(x - y)
    return diff


def extend_2d_ndarray(input_array, n, num_features):
    m = input_array.shape[0]
    repeats = n // m + 1  # 计算重复次数
    output_array = np.resize(input_array, (repeats * m, num_features))[:n, :]  # 重塑并切片以得到正确的长度
    return output_array


def get_consistency(model, item, exp, label, prediction, percent=50, window=2):
    # Test with some data
    assert len(item) == len(exp), "The size of the 'items' and 'exp' lists should be the same."
    if exp.shape[-1] > 1:
        exp = exp.sum(axis=1, keepdims=True)
    # 计算前n%贡献度的阈值
    threshold = sorted(exp)[int(len(exp) * (100 - percent) / 100)]

    # 找到所有高于阈值的元素的索引
    high_contribution_indices = [i for i, e in enumerate(exp) if e > threshold]
    if len(high_contribution_indices) == 0:
        return 0
    # 根据这些索引，在item中提取相应的元素以及其前后5个元素
    selected_indices = set()
    for i in high_contribution_indices:
        selected_indices.update(range(max(0, i - window), min(len(item), i + window + 1)))

    # 生成并返回结果列表
    result = [item[i] for i in sorted(selected_indices)]
    new_items = np.array(result)
    if args.data_type == "h5":
        if len(new_items) < len(item):
            new_items = extend_2d_ndarray(new_items, len(item), new_items.shape[-1])
    new_predictions = model.predict(new_items[None, :])[0][label]
    ccd = abs_diff(new_predictions, prediction)
    return ccd.astype(np.float64)


if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/ecg_model.hdf5")
    parser.add_argument("--input_dir", type=str, default="data/ECG200/ECG200_TEST.CSV")
    parser.add_argument("--data_type", type=str, default="csv")
    parser.add_argument("--method_name", type=str, default="IG")
    parser.add_argument("--class_num", type=int, default=2)

    args = parser.parse_args()
    predictions = None
    sequences = None
    method_name = args.method_name

    if args.data_type == "csv":
        model = load_model(args.model_path)
        test = pd.read_csv(args.input_dir, header=0)
        test_value = test.values[:, 1:]
        test_value = np.expand_dims(test_value, axis=-1)
        sequences = test_value[1][None, :]  # 1*96*1

        predictions = model.predict(sequences)
    elif args.data_type == "h5":
        # 读取数据 + 模型
        f = h5py.File(args.input_dir, 'r')
        datax = f['X'][:]
        f.close()
        print(datax.shape)
        in_shape = datax.shape
        model = Net(in_shape, args.class_num)
        adam = legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        model.load_weights(args.model_path)

        sequences = datax[2][None, :]
        result = model.predict(sequences)
        predictions = tf.nn.softmax(result, axis=-1).numpy()

    test_label = int(np.argmax(predictions[0]))
    anti_label = int(np.argmin(predictions[0]))

    item = sequences[0]
    label = test_label
    # 解释结果
    """Initialization
           Arguments:
               model [torch.nn.Module, tf.keras.Model]: model to be explained
               NumTimeSteps int : Number of Time Step
               NumFeatures int : Number Features
               method str: Saliency Methode to be used
                + Gradients (GRAD)
                + Integrated Gradients (IG)
                + SmoothGrad (SG)
               mode str: Second dimension 'time'->`(1,time,feat)`  or 'feat'->`(1,feat,time)`
    """
    int_mod = TSR(model, len(item), sequences.shape[-1], method=method_name, mode='time')

    exp = int_mod.explain(sequences, labels=label, TSR=True)
    anti_exp = int_mod.explain(sequences, labels=anti_label, TSR=True)

    # %matplotlib inline
    result_img = int_mod.plot(sequences, exp, figsize=(8, 6),
                              save='output/' + args.data_type + '/' + method_name + '_result.png')

    faithfulness = get_faithfulness(model, sequences, exp, label)
    class_sensitivity = get_class_sensitivity(exp, anti_exp)
    consistency = get_consistency(model, item, exp, label, predictions[0][label])
    data = {
        "faithfulness": faithfulness,
        "class_sensitivity": class_sensitivity,
        "consistency": consistency
    }
    with open("output/" + args.data_type + '/' + method_name + "_metrics.json", "w") as f:
        json.dump(data, f)
