import cv2
import numpy as np
#打开视频
video_path = 'lab2(plus).mp4'
cap = cv2.VideoCapture(video_path)
#检查是否打开
if not cap.isOpened():
    print("无法打开视频文件！请检查路径。")
    exit()
#视频数据
fps = cap.get(cv2.CAP_PROP_FPS)      # 帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频信息：{width}x{height}, {fps:.2f} fps")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('lab2(plus)_out.mp4', fourcc, fps, (width, height))
# 定义红色的 HSV 阈值
lower_red1 = np.array([0, 160, 50])      # 红色低区间
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 160, 20])    # 红色高区间
upper_red2 = np.array([180, 255, 255])
#逐帧读取
frame_count = 0
while True:
    success, frame = cap.read()
    processed_frame = frame

    if not success:
        print("失败")
        break

    frame_count += 1
    #处理图像
    hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
    # 生成二值图
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 2. 定义结构元素 (可以根据噪点大小调整，这里用 3x3 或 5x5 矩形)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    clean_img = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    #dilated_img = cv2.dilate(red_mask, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    #eroded_img = cv2.erode(dilated_img, kernel, iterations=1)


    # 框选轮廓
    contours, _ = cv2.findContours(clean_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 矩形模拟
    for contour in contours:
        # 限制面积
        area = cv2.contourArea(contour)
        if area < 300 or area > 6000:
            continue
        # 多边形拟合
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        #长宽比筛选
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        long_side = max(w, h)
        short_side = min(w, h)

        if short_side > 0:
            aspect_ratio = long_side / short_side
            if 1 < aspect_ratio < 4.0:
                # if len(approx) != 4:
                # q    continue
                # 画出绿色的拟合六边形
                # cv2.drawContours(img_bgr, [approx], -1, (0, 255, 0), 2)
                # 画出边框
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Mask", red_mask)
    cv2.imshow("Detection Result", frame)

    #out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    # 按 'q' 或 'ESC' 键退出
    if key == ord('q') or key == 27:
        print("用户终止程序。")
        break
cap.release()                  # 关闭视频文件
cv2.destroyAllWindows()        # 销毁所有窗口
print("程序结束。")