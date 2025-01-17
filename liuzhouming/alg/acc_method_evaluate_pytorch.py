import argparse
import json

from sklearn.metrics import precision_score, \
    recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from common_config import save_path, service, access_key, secret_key, get_result_file_path

from Data_Utils.Download_minio import *
from Data_Utils.ModelLoader import *
from Data_Utils.MyDataset import MyDataset

if __name__ == "__main__":
    BATCH_SIZE = "BATCH_SIZE"
    ACC = "ACC"
    PRECISION = "PRECISION"
    F1_SCORE = "F1_SCORE"
    RECALL = "RECALL"

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for Attack')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size, the size of sample to use calculate')
    parser.add_argument('--model_url', type=str, default='http://127.0.0.1:9000/model/20230913/172152-mnist_mlp.pt', help='model url')
    parser.add_argument('--dataset_url', type=str, default='http://localhost:9000/dataset/20230913/172116-fashion-mnist.zip', help='dataset url')
    parser.add_argument('--config', type=str, nargs='+', default=["ACC", "PRECISION", "F1_SCORE", "RECALL"],
                        help='method name list')
    parser.add_argument('--configFilePath', type=str,
                        default=r'D:\pythonDemo\alg\config\342001b4-6b3a-40f2-a2b1-48a941f11c2e.json',
                        help='method config file path')

    # 解析参数
    args = parser.parse_args()
    device = args.device
    try:
        device = torch.device(device)
    except:
        device = torch.device('cpu')

    batch_size = args.batch_size
    model_url = args.model_url
    dataset_url = args.dataset_url
    config = args.config
    configFilePath = args.configFilePath

    # 评测结果存放文件
    resultFilePath = get_result_file_path(configFilePath)

    print('resultFilePath :', resultFilePath)
    result = dict()
    # load model
    # model = models.resnet18(pretrained=True).eval()
    # torch.save(model, "model.pt")
    download_model_minio(model_url, service=service, access_key=access_key, secret_key=secret_key, save_path=save_path)
    model = load_model_pt(model_url, save_path=save_path)
    model.eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[
    #     0.229, 0.224, 0.225], axis=-3)

    # load images
    # images, labels = samples(fmodel, dataset='imagenet', batchsize=batch_size)
    mytransform = transforms.Compose([
        transforms.ToTensor()
    ]
    )
    download_dataset_minio(dataset_url, service=service, access_key=access_key, secret_key=secret_key,
                           save_path=save_path)
    test_loader = DataLoader(
        MyDataset(dataset_url, save_path=save_path, transform=mytransform),
        batch_size=batch_size,
        shuffle=True)

    # 取图像和标签
    images, labels = next(iter(test_loader))
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)

    preds = torch.argmax(output, -1)  # 预测值

    result[BATCH_SIZE] = batch_size

    if ACC in config:
        acc_score = accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy())
        result[ACC] = acc_score

    if F1_SCORE in config:
        f1_score = f1_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(),
                            average='macro')  # 也可以指定micro模式
        result[F1_SCORE] = f1_score

    if PRECISION in config:
        precision = precision_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(),
                                    average='macro')
        result[PRECISION] = precision

    # accuracy = (preds == labels).float().mean()
    # print("acc: ", accuracy.item())

    if RECALL in config:
        recall_score = recall_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(),
                                    average='macro')  # 也可以指定micro模
        result[RECALL] = recall_score

    # 评测结果记录 到文件
    with open(resultFilePath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(result))
    print('resultFilePath :', resultFilePath)
