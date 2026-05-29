import cv2
import numpy as np
#读取转换
img_bgr = cv2.imread('lab2.png')
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#通道分离
img_h,img_s,img_v = cv2.split(img_hsv)
#生成二值图
_, binary_v = cv2.threshold(img_v, 200, 255, cv2.THRESH_BINARY)
#框选轮廓
contours, _ = cv2.findContours(binary_v, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#矩形模拟
for contour in contours:
    #限制面积
    area = cv2.contourArea(contour)
    if area < 500 or area > 2000:
        continue
    # 多边形拟合
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 6:
        continue
    #画出绿色的拟合六边形
    #cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 2)
    #画出边框
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('lab1_out.png', img_bgr)
cv2.imwrite('lab2_out.png', img_bgr)
cv2.waitKey(0)


