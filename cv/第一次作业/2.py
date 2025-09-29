import cv2
import numpy as np

# 读取灰度图像
input_file_1 = 'color1.jpg'
input_file_2 = 'color2.jpg'


img_1 = cv2.imread(input_file_1, cv2.IMREAD_GRAYSCALE)#以灰度读取
img_2 = cv2.imread(input_file_2, cv2.IMREAD_GRAYSCALE)

img_2_resized = cv2.resize(img_2, (img_1.shape[1], img_1.shape[0]))#尺寸保持统一

# 线性加运算
result = img_1 + img_2_resized
result = np.clip(result, 0, 255).astype(np.uint8)		

# 显示和保存结果
cv2.imwrite('Addition_result.jpg', result)