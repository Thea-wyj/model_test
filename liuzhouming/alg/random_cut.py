import cv2
import random

def random_cut(img, h, w):
    # h、w为想要截取的图片大小
    '''
    机器学习中随机产生负样本的
    '''


# 随机产生x,y   此为像素内范围产生
    y = random.randint(1, 890)
    x = random.randint(1, 1480)
    # 随机截图
    shape = img.shape
    if h > shape[0] or w > shape[1]:
        print("h or w is too big for this image!")
        print("The image‘s shape is "+str(shape[0])+"*"+str(shape[1])+"*"+str(shape[2]))
        return
    # y_max 和 x_max保证裁剪范围不超过图片的范围
    y_max = y+h
    x_max = x+w
    if y+h > shape[0]:
        y_max = shape[0]
        y = y_max-h
    if x+w > shape[1]:
        x_max = shape[1]
        x = x_max-w
    cropImg = img[(y):(y_max), (x):(x_max)]
    cropImg = cv2.resize(cropImg, (shape[0], shape[1] ))
    #cv2.imwrite('pic/' + str(count) + '.png', cropImg)
    cv2.imshow('rain_effct', cropImg)
    cv2.waitKey()
    cv2.destroyWindow('rain_effct')


if __name__ == '__main__':
    img = cv2.imread('img.png')
    random_cut(img, h=300, w= 400)