import cv2
import numpy as np

def gauss_noise(mean, sigma, img):
    # 参考博客： https://blog.csdn.net/sinat_29957455/article/details/123977298
    # 读取图片
    #img = cv2.imread("img.png")
    shape = img.shape
    # 设置高斯分布的均值和方差
    #mean = 0
    # 设置高斯分布的标准差
    #sigma = 10
    # 根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean, sigma, (shape[0], shape[1], shape[2]))
    # 给图片添加高斯噪声
    noisy_img = img + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    # 保存图片
    return noisy_img
    # cv2.imwrite("noisy_img.png",noisy_img)
    # cv2.imshow('noise_effct_result', noisy_img)
    # cv2.waitKey()
    # cv2.destroyWindow('rain_effct')

def salt_pepper_noise(image, prob):
    """
    添加椒盐噪声
    :param image: 输入图像
    :param prob: 噪声比
    :return: 带有椒盐噪声的图像
    """
    salt = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                salt[i][j] = 0
            elif rdn > thres:
                salt[i][j] = 255
            else:
                salt[i][j] = image[i][j]
    return salt

def salt_noise(s_vs_p, amount, img):
    # 椒盐噪声就是给图片添加黑白噪点，椒指的是黑色的噪点(0,0,0)盐指的是白色的噪点(255,255,255)，通过设置amount来控制添加噪声的比例，值越大添加的噪声越多，图像损坏的更加严重
    # 参考博客： https://blog.csdn.net/sinat_29957455/article/details/123977298
    # 读取图片
    #img = cv2.imread("demo.png")
    # 设置添加椒盐噪声的数目比例
    #s_vs_p = 0.5
    # 设置添加噪声图像像素的数目
    #amount = 0.04
    noisy_img = np.copy(img)
    # 添加salt噪声
    num_salt = np.ceil(amount * img.size * s_vs_p)
    # 设置添加噪声的坐标位置
    # print(img.shape)
    shape = img.shape[:-1]
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in shape]
    noisy_img[coords[0], coords[1], :] = 255
    # 添加pepper噪声
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape]
    noisy_img[coords[0], coords[1], :] = 0
    return noisy_img
    # 保存图片
    #cv2.imwrite("noisy_img.png", noise_img)
    # cv2.imshow('salt_effct_result', noisy_img)
    # cv2.waitKey()
    # cv2.destroyWindow('rain_effct')

def possion_noise(img):
    # 参考博客： https://blog.csdn.net/sinat_29957455/article/details/123977298
    # 读取图片
    #img = cv2.imread("demo.png")
    # 计算图像像素的分布范围
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    # 给图片添加泊松噪声
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img
    # 保存图片
    #cv2.imwrite("noisy_img.png", noisy_img)
    # cv2.imshow('possion_effct_result', noisy_img)
    # cv2.waitKey()
    # cv2.destroyWindow('rain_effct')

    # shape = img.shape
    # # 添加噪声
    # noise_type = np.random.poisson(lam=0.3, size=(shape[0], shape[1], shape[2])).astype(dtype='uint8')  # lam>=0 值越小，噪声频率就越少，size为图像尺寸
    # noise_image = noise_type + img  # 将原图与噪声叠加
    # cv2.imshow('possion_effct_result', noise_image)
    # cv2.waitKey(0)
    # cv2.destroyWindow()

def speckle_noise(img):
    # 读取图片
    #img = cv2.imread("demo.png")
    shape = img.shape
    # 随机生成一个服从分布的噪声
    gauss = np.random.randn(shape[0], shape[1], shape[2])
    # 给图片添加speckle噪声
    noisy_img = img + img * gauss
    # 归一化图像的像素值
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img
    # 保存图片
    # cv2.imshow('speckle_effct_result', noisy_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow()


if __name__ == '__main__':
    img = cv2.imread("img.png")
    #gauss_noise(0,25, img)
    #salt_noise(0.5, 0.04, img)
    #possion_noise(img)
    speckle_noise(img)