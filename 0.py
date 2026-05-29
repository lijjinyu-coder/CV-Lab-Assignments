import cv2 # 导入视觉库
img = cv2.imread('0.png') # 把引号内的文件名修改为你的图片名
cv2.imshow('test', img)
cv2.waitKey(0)