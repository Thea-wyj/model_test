import argparse
import json
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import precision_score, \
    recall_score, f1_score, accuracy_score

import tensorflow as tf
from Data_Utils.CustomTfDataset import CustomTfDataset
from Data_Utils.CustomTfDataset import load_model_h5, load_data_tf
from Data_Utils.Download_minio import download_model_minio, \
    download_dataset_minio
from common_config import save_path, service, access_key, secret_key, get_result_file_path


if __name__ == "__main__":
    logging.getLogger('tensorflow').disabled = True
    BATCH_SIZE = "BATCH_SIZE"
    ACC = "ACC"
    PRECISION = "PRECISION"
    F1_SCORE = "F1_SCORE"
    RECALL = "RECALL"

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for Attack')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size, the size of sample to use calculate')
    parser.add_argument('--model_url', type=str,
                        default='http://localhost:9090/model/20231026/103442-nwpu.h5',
                        help='model url')
    parser.add_argument('--dataset_url', type=str,
                        default='http://localhost:9090/dataset/20231026/104418-NWPU.zip', help='dataset url')
    parser.add_argument('--config', type=str, nargs='+', default=["ACC", "PRECISION", "F1_SCORE", "RECALL"],
                        help='method name list')

    parser.add_argument('--configFilePath', type=str,
                        default=r'D:\pythonDemo\alg\config\af957a4a-9e94-461e-bad6-9909b99ba953.json',
                        help='method config file path')

    # 解析参数
    args = parser.parse_args()
    device = args.device

    batch_size = args.batch_size
    model_url = args.model_url
    dataset_url = args.dataset_url
    config = args.config

    configFilePath = args.configFilePath

    # 评测结果存放文件
    resultFilePath = get_result_file_path(configFilePath)
    print(resultFilePath)

    result = dict()
    # load model
    # model = models.resnet18(pretrained=True).eval()
    # torch.save(model, "model.pt")
    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)

    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[
    #     0.229, 0.224, 0.225], axis=-3)

    # load images
    download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
                           save_path=save_path)

    # custom tf dataset
    model = load_model_h5(model_url, save_path=save_path)
    custom_tf_dataset = CustomTfDataset(dataset_url, save_path, None)

    # channel_last
    global images
    imgs, labels = custom_tf_dataset[:len(custom_tf_dataset)]  # 所有数据集加载，需要优化
    test_iter = load_data_tf((imgs, labels), batch_size)  # 加载batchsize个
    for X, y in test_iter:
        images, labels = X, y
        break

    output = model.predict(images)  # 预测
    preds = tf.argmax(output, -1)  # 预测值

    result[BATCH_SIZE] = batch_size

    if ACC in config:
        acc_score = accuracy_score(y_true=labels.numpy(), y_pred=preds.numpy())
        result[ACC] = acc_score

    if F1_SCORE in config:
        f1_score = f1_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                            average='macro')  # 也可以指定micro模式
        result[F1_SCORE] = f1_score

    if PRECISION in config:
        precision = precision_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                    average='macro')
        result[PRECISION] = precision

    if RECALL in config:
        recall_score = recall_score(y_true=labels.numpy(), y_pred=preds.numpy(),
                                    average='macro')  # 也可以指定micro模
        result[RECALL] = recall_score

    # 评测结果记录 到文件
    f = open(resultFilePath, 'w', encoding='utf-8')
    f.write(json.dumps(result))
    f.close()
    print(resultFilePath)

