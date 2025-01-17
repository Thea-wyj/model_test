from Data_Utils.Bucket import Bucket
import zipfile
import os

def download_dataset_minio(dataset_url, service, access_key, secret_key, save_path):
    directoryName = dataset_url.split('/')[-1].split('.')[0] # 取url最后一段 作为路径名称
    dataset_dir = save_path + "/download/dataset/" + directoryName
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) != 0:
        return
    minio_obj = Bucket(service, access_key, secret_key)
    path_split = dataset_url.split('/')
    obj_name = '/'.join(path_split[-2:]) # 对象名称
    minio_obj.fget_file("dataset", obj_name, save_path+ "/download/dataset/" + directoryName + "/dataset.zip")
    f = zipfile.ZipFile(save_path+ "/download/dataset/" + directoryName + "/dataset.zip", 'r')  # 压缩文件位置
    for file in f.namelist():
        f.extract(file, save_path+ "/download/dataset/" + directoryName + "/")  # 解压位置
    f.close()


def download_pic_minio(picUrls, service, access_key, secret_key, save_path):
    minio_pic_Urls = []
    minio_obj = Bucket(service, access_key, secret_key)
    for pic_url in picUrls:
        pic_path = save_path + "/" + pic_url;
        if not os.path.exists(pic_path):
            minio_obj.fget_file("images", pic_url, pic_path)
        minio_pic_Urls.append(pic_url)
    return minio_pic_Urls


def download_model_minio(model_url, service, access_key, secret_key, save_path):
    path_split = model_url.split('/')
    directoryPathName = model_url.split('/')[-1] # 取url最后一段 作为路径名称
    model_dir = save_path+"/download/model/"+directoryPathName
    if os.path.exists(model_dir):
        return
    minio_obj = Bucket(service, access_key, secret_key)
    obj_name = '/'.join(path_split[-2:])
    # print(save_path + "/download/model/" + obj_name)
    # print('obj_name', obj_name)
    minio_obj.fget_file("model", obj_name, save_path + "/download/model/" + directoryPathName)


if __name__ == '__main__':
    pass
    # minio_obj = Bucket(service="localhost:9000", access_key="minioadmin", secret_key="minioadmin")

    # minio_obj.fget_file("model", "20221213/152633-1796_.zip", "download/model/1/try.zip")

    # mytransform = transforms.Compose([
    #     transforms.ToTensor()
    # ]
    # )
    # dataset_url='http://localhost:9000/dataset/20230221/155930-minist.zip'
    # model_url = 'http://localhost:9000/model/20230221/204945-mnist_mlp.pt'
    # # minio 配置
    # save_path = "E:/data_2"
    # service = "localhost:9000"
    # access_key = "admin"
    # secret_key = "bupt_Minio_admin"
    #
    # download_model_minio(model_url, service, access_key, secret_key, save_path)
    # #
    # # model_pt = load_model_pt(model_url, save_path)
    # download_dataset_minio(dataset_url,  service = service, access_key = access_key, secret_key = secret_key, save_path=save_path)
    # test_loader = DataLoader(
    #     MyDataset(dataset_url,save_path=save_path, transform=mytransform),
    #     batch_size=20,
    #     shuffle=True)
    # images, labels = next(iter(test_loader))
    # print(images[0],labels[0])
    # print(type(images), images.shape, type(labels), labels.shape) # (b,h,w,c)

