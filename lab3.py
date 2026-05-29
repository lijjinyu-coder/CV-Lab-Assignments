import cv2
import numpy as np

prev_ordered_pts = None


#  核心排序与追踪函数
def get_clockwise_corners(pts):
    center = np.mean(pts, axis=0)
    dx, dy = pts[:, 0] - center[0], pts[:, 1] - center[1]
    angles = np.where(np.arctan2(dy, dx) < 0, np.arctan2(dy, dx) + 2 * np.pi, np.arctan2(dy, dx))
    return pts[np.argsort(-angles)]


def track_corners(new_pts, prev_pts):
    new_ordered = get_clockwise_corners(new_pts)

    if prev_pts is None:
        return np.roll(new_ordered, -np.argmin(new_ordered[:, 0] + new_ordered[:, 1]), axis=0)

    min_dist, best_shift = float('inf'), 0
    for i in range(4):
        dist = np.sum(np.linalg.norm(np.roll(new_ordered, i, axis=0) - prev_pts, axis=1))
        if dist < min_dist:
            min_dist, best_shift = dist, i

    # 如果移动距离过大（超过200像素），重新初始化原点
    if min_dist > 200:
        return np.roll(new_ordered, -np.argmin(new_ordered[:, 0] + new_ordered[:, 1]), axis=0)

    return np.roll(new_ordered, best_shift, axis=0)


fx = 772.6899  # X轴焦距
fy = 773.8098  # Y轴焦距
cx = 647.9536  # 光心 X 坐标
cy = 362.0083  # 光心 Y 坐标
k1 = 0.0395  # 径向畸变系数 1
k2 = -0.1149  # 径向畸变系数 2

# 3x3 相机内参矩阵
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

# 畸变系数矩阵
dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

# 真实尺寸（mm）
W, H = 100.0, 100.0
# 物体点：原点在左上角，Z=0
objectPoints = np.array([
    [0, 0, 0],
    [W, 0, 0],
    [W, H, 0],
    [0, H, 0]
], dtype=np.float32)

# 坐标轴 3D 端点 (长度50mm)
axis_points = np.array([
    [0.0, 0.0, 0.0],  # 索引 0: 原点
    [50.0, 0.0, 0.0],  # 索引 1: X轴终点
    [0.0, 50.0, 0.0],  # 索引 2: Y轴终点
    [0.0, 0.0, 50.0]  # 索引 3: Z轴终点
], dtype=np.float64)

cap = cv2.VideoCapture(0)

# 初始化视频导出设置 (导出为 MP4)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('lab3_output.mp4', fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 图像处理（灰度、模糊、边缘检测）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 40, 150)

    # 获取轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 识别到 4 边形
        if len(approx) == 4:
            pts = approx.reshape(4, 2)

            # 使用追踪逻辑对齐角点并保存为历史参考帧
            ordered_pts = track_corners(pts, prev_ordered_pts)
            prev_ordered_pts = ordered_pts.copy()
            ordered_pts = ordered_pts.astype(np.float32)

            # 绘制 4 个角点及标号
            for i, pt in enumerate(ordered_pts):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(pt[0]) - 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 解算 3D 姿态 (solvePnP)
            success, rvec, tvec = cv2.solvePnP(objectPoints, ordered_pts, camera_matrix, dist_coeffs)

            if success:
                # 投影 3D 坐标轴到 2D 画面上
                projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                projected_axis_points = np.int32(projected_axis_points)

                # 绘制坐标轴线条
                for i in range(1, len(projected_axis_points)):
                    cv2.line(frame,
                             tuple(projected_axis_points[0].ravel()),
                             tuple(projected_axis_points[i].ravel()),
                             (255, 0, 0), 2)

    # 写入视频文件
    out.write(frame)

    # 显示当前画面
    cv2.imshow('frame', frame)

    # 监听键盘，按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # 这一步很关键，否则视频文件会损坏
cv2.destroyAllWindows()