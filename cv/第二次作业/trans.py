import cv2
import numpy as np

input_file = "1.png"
output_file_1 = "affine.png"
output_file_2 = "perspective.png"

# 读取输入图像
image = cv2.imread(input_file)

# 定义仿射变换矩阵
rows, cols, ch = image.shape
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
dst_points = np.float32([[50, 50], [cols - 100, 50], [50, rows - 100]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

# 应用仿射变换
affine_image = cv2.warpAffine(image, affine_matrix, (cols, rows))
cv2.imwrite(output_file_1, affine_image)

# 定义透视变换矩阵的源点和目标点
src_points = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
dst_points = np.float32([[50, 50], [cols - 100, 50], [cols - 100, rows - 100], [50, rows - 100]])

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 应用透视变换
perspective_image = cv2.warpPerspective(image, perspective_matrix, (cols, rows))
cv2.imwrite(output_file_2, perspective_image)



