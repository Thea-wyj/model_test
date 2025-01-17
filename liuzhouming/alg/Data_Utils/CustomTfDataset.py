import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# 定义对数据集的加载方式
def filePlace_loader(filename):
    place_list = []
    label_list = []
    with open(filename, 'r') as f:
        for line in f:
            place_label = line.split()
            place_list.append(place_label[0])
            label_list.append(place_label[1])
    return place_list, label_list


def default_loader(path):
    return Image.open(path)


# 继承自data.Dataset,完成对方法的重写
class CustomTfDataset(object):
    def __init__(self, dataset_url, save_path, transform=None, loader=default_loader):
        # 获得对应的数据位置和标签　　　　
        directoryName = dataset_url.split('/')[-1].split('.')[0]  # 取url最后一段 作为路径名称
        self.filePath = save_path + "/download/dataset/" + directoryName

        # print("MyDataset file path is {}".format(self.filePath))
        FilePlaces, LabelSet = filePlace_loader(self.filePath + "/test.txt")
        # 对标签进行修改，因为读进来变成了str类型，要修改成long类型
        LabelSet = [np.int64(i) for i in LabelSet]
        if transform is not None:
            LabelSet = tf.convert_to_tensor(LabelSet, dtype=tf.int64)  # convert to tensor
        self.imgs_place = FilePlaces
        self.LabelSet = LabelSet
        self.transform = transform
        self.loader = loader

    # 这里是对数据进行读取 使用之前定义的loader方法来执行
    def __getitem__(self, item):
        img_place = self.imgs_place[item]
        label = self.LabelSet[item]
        # print("img_place is {}".format(self.filePath+"/"+img_place))

        imgs_place = []
        if isinstance(img_place, str):
            imgs_place = [img_place]
        else:
            imgs_place = img_place

        img_np_list = []
        for img_path in imgs_place:
            img = self.loader(self.filePath + "/" + img_path)
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = np.array(img)  # to ndarray
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
            img_np_list.append(img)

        if isinstance(img_place, str):
            return img_np_list[0], label
        return img_np_list, label

    def __len__(self):
        return len(self.imgs_place)


def mytransform(x):
    x = np.array(x)
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=-1)
    return x.transpose(2, 0, 1)
    # x 为numpy(b, h, w, c)
    # tf_tensor = tf.convert_to_tensor(np.array(x))
    # return tf.transpose(tf_tensor, perm=[2, 0, 1])  # c,h,w


def load_model_h5(model_url, save_path):
    path_split = model_url.split('/')
    filepath = save_path + "/download/model/" + path_split[-1]
    return keras.models.load_model(filepath)


def load_data_tf(mnist_test, batch_size, resize=None):
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.cast(X, dtype='float32') / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)

    return tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
        batch_size).map(resize_fn)


if __name__ == "__main__":
    # dataset = CustomTfDataset("http://127.0.0.1:9000/dataset/20230224/11739-dataset.zip", "E:/data_2", mytransform)
    # print(dataset[0][0].shape, type(dataset[0][0]))
    # print("test")
    modeL_path = r"E:/densenet121_weights_tf_dim_ordering_tf_kernels.h5"
    model = keras.models.load_model(modeL_path)
    print(model.summary())
