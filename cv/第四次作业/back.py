import cv2
import numpy as np

inout_file = 'house.png'
image = cv2.imread(inout_file, cv2.IMREAD_GRAYSCALE)

# 使用反向滤波

# 定义一个低通滤波器函数
def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= cutoff:
                mask[i, j] = 1
    return mask

# 将图像转换为频域
image_fft = np.fft.fft2(image)# 傅里叶变换
image_fft_shifted = np.fft.fftshift(image_fft)# 将零频率分量移到频谱中心

# 定义滤波器参数
cutoff_frequency = 50  # 截止频率
filter_mask = ideal_low_pass_filter(image.shape, cutoff_frequency)

# 应用反向滤波器
restored_fft_shifted = image_fft_shifted / (filter_mask + 1e-5)  # 避免除以零
restored_fft = np.fft.ifftshift(restored_fft_shifted)
restored_image = np.fft.ifft2(restored_fft)
restored_image = np.abs(restored_image)


cv2.imwrite('restored_image.png', restored_image)
