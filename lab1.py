import cv2
#读取图像
img_bgr = cv2.imread('lab1.png')
#转换hsv
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#通道分离
img_h,img_s,img_v = cv2.split(img_hsv)
#H,S,V通道筛选
mask_h = cv2.inRange(img_h,130, 179)
mask_s = cv2.inRange(img_s,30, 255)
mask_v = cv2.inRange(img_v,30,255)
#与运算
mask_h_and_s = cv2.bitwise_and(mask_h, mask_s)
mask = cv2.bitwise_and(mask_h_and_s, mask_v)
#提取
img_out = cv2.bitwise_and(img_bgr,img_bgr,mask=mask)
#显示保存
cv2.imshow('Original', img_bgr)
cv2.imshow('Result', img_out)
cv2.imwrite('lab1_out.png',img_out)

cv2.waitKey(0)
cv2.destroyAllWindows()