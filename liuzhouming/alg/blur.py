import cv2
import numpy as np

img = cv2.imread("img.png")
kernel = np.ones((3,3),np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),0,0)
imgMedianBlur = cv2.blur(img,(5,5))
imgMBlur = cv2.medianBlur(img,3)

imgCanny = cv2.Canny(img,150,200)
imgDilation = cv2.dilate(img,kernel,iterations=1)
imgErode = cv2.erode(img,kernel,iterations=1)

cv2.imshow("Original Image",img)
cv2.imshow("Gray Image",imgGray)
cv2.imshow("Gaussian Blur",imgBlur)
cv2.imshow("Median Blur",imgMedianBlur)
cv2.imshow("mBlur", imgMBlur)
cv2.imshow("Canny Edge",imgCanny)
cv2.imshow("Image Dilation",imgDilation)
cv2.imshow("Image Erode",imgErode)
cv2.waitKey(0)


