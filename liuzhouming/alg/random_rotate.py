import cv2
import numpy as np
import os
from math import *
import random
import numpy as np

#参考博客：https://blog.csdn.net/qq_40037127/article/details/125206182
def random_rotate(img):
    x = random.randint(0, 360)
    degree = x
    #img = cv2.imread('img.png')
    height, width = img.shape[:2]

    M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    M[0, 2] += (widthNew - width) / 2
    M[1, 2] += (heightNew - height) / 2

    im_rotate = cv2.warpAffine(img, M, (widthNew, heightNew), borderValue=(255, 255, 255))
    im_rotate = cv2.resize(im_rotate, (width, height))
    cv2.imshow("test", im_rotate)
    #cv2.imwrite("test_new4.jpg", im_rotate)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('img.png')
    print(img.shape, type(img))
    random_rotate(img)