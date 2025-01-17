import torch
from PIL import Image
from torch.utils import data
import numpy as np

#定义对数据集的加载方式
def filePlace_loader(filename):
    place_list = []
    label_list = []
    with open(filename, 'r') as f:
        for line in f:
            # print(line)
            place_label = line.split()
            place_list.append(place_label[0])
            label_list.append(place_label[1])
    return place_list, label_list


def default_loader(path):
    return Image.open(path)


#继承自data.Dataset,完成对方法的重写
class MyDataset(data.Dataset):
    def __init__(self, dataset_url, save_path, transform = None, loader = default_loader):
        # 获得对应的数据位置和标签　　　　
        directoryName = dataset_url.split('/')[-1].split('.')[0] # 取url最后一段 作为路径名称
        self.filePath = save_path + "/download/dataset/" + directoryName

        # print("MyDataset file path is {}".format(self.filePath))
        FilePlaces, LabelSet = filePlace_loader(self.filePath + "/test.txt")
        # 对标签进行修改，因为读进来变成了str类型，要修改成long类型
        LabelSet = [np.int64(i) for i in LabelSet]
        if transform != None:
            LabelSet = torch.Tensor(LabelSet).long()
        self.imgs_place = FilePlaces
        self.LabelSet = LabelSet
        self.transform = transform
        self.loader = loader

# 这里是对数据进行读取 使用之前定义的loader方法来执行
    def __getitem__(self, item):
        img_place = self.imgs_place[item]
        label = self.LabelSet[item]
        # print("img_place is {}".format(self.filePath+"/"+img_place))
        img = self.loader(self.filePath+"/"+img_place)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img) # to ndarray
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
        return img,label


    def __len__(self):
        return len(self.imgs_place)


