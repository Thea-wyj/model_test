import numpy as np
import torchvision
from torchvision import transforms
from torch.utils import data
import random

def load_dataset(dataset, data_size=20, trans=transforms.ToTensor(), load_test=True, is_shuffle=False):
    # trans = transforms.ToTensor()  # PIL -> 32位浮点数 /255 -》 0~1
    mnist_data = None
    if dataset.upper() == "MNIST":
        mnist_data = torchvision.datasets.MNIST(root='./data', train=(load_test!=True), download=True, transform=trans)
    elif dataset.upper() == "FASHIONMNIST":
        mnist_data = torchvision.datasets.FashionMNIST(root='./data', train=(load_test != True), download=True, transform=trans)
    elif dataset.upper() == "CIFAR10":
        mnist_data = torchvision.datasets.CIFAR10(root='./data', train=(load_test != True), download=True, transform=trans)
    else:
        return mnist_data
    if trans is None:
        images = []
        labels = []
        indexArray = random.sample(range(0, len(mnist_data)), data_size)
        for index in indexArray:
            image_ndarray = np.array(mnist_data[index][0])
            if len(image_ndarray.shape) == 2:
                image_ndarray = np.expand_dims(image_ndarray, axis=-1)
            images.append(image_ndarray)
            labels.append(mnist_data[index][1])
        return np.array(images), np.array(labels)
    else:
        images, labels = next(iter(data.DataLoader(mnist_data, batch_size=data_size, shuffle=is_shuffle)))
        return images, labels


if __name__ == "__main__":
    # numpy
    images, labels = load_dataset(dataset="mnist", trans=None)
    print(len(images), len(labels))
    print(type(images), type(labels))
    print(images.shape, labels.shape)

    # tensor
    images_tensor, labels_tensor = load_dataset(dataset='mnist', trans=transforms.ToTensor())
    print(len(images_tensor), len(labels_tensor))
    print(type(images_tensor), type(labels_tensor))
    print(images_tensor.shape, labels_tensor.shape)
