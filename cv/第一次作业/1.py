import cv2
import numpy as np

# 读取灰度图像
input_file = 'gray.jpg'

img = cv2.imread(input_file)
# 检查图像是否读取成功
if img is None:
	print(f"图像读取失败，请检查文件路径和文件名: {input_file}")
	exit()

# 线性点运算参数
a = 1.2  # k比例因子
b = 30.0  # 偏移量

# 线性点运算
result = a * img + b
result = np.clip(result, 0, 255).astype(np.uint8)

# 显示和保存结果
cv2.imwrite('linear_result.jpg', result)