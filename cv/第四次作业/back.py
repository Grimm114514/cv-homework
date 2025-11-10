import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

input_file = 'house.png'
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

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

# 高斯低通滤波器
def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            mask[i, j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    return mask

# 巴特沃斯低通滤波器
def butterworth_low_pass_filter(shape, cutoff, order):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            mask[i, j] = 1 / (1 + (distance / cutoff) ** (2 * order))
    return mask

# 定义运动模糊的点扩散函数（PSF）
def motion_blur_psf(shape, length, angle):
    rows, cols = shape
    psf = np.zeros((rows, cols), dtype=np.float32)
    center_row, center_col = rows // 2, cols // 2
    angle = np.deg2rad(angle)
    for i in range(length):
        x = int(center_col + i * np.cos(angle))
        y = int(center_row + i * np.sin(angle))
        if 0 <= x < cols and 0 <= y < rows:
            psf[y, x] = 1
    psf /= psf.sum()
    return psf

# 将图像转换为频域
image_fft = np.fft.fft2(image)  # 傅里叶变换
image_fft_shifted = np.fft.fftshift(image_fft)  # 将零频率分量移到频谱中心

# 定义滤波器参数
cutoff_frequency = 50  # 截止频率
order = 2  # 巴特沃斯滤波器阶数

# 应用理想低通滤波器
ideal_filter_mask = ideal_low_pass_filter(image.shape, cutoff_frequency)
restored_fft_shifted_ideal = image_fft_shifted / (ideal_filter_mask + 1e-5)  # 避免除以零
restored_fft_ideal = np.fft.ifftshift(restored_fft_shifted_ideal)
restored_image_ideal = np.fft.ifft2(restored_fft_ideal)
restored_image_ideal = np.abs(restored_image_ideal)
cv2.imwrite('restored_image_ideal.png', restored_image_ideal)

# 应用高斯低通滤波器
gaussian_filter_mask = gaussian_low_pass_filter(image.shape, cutoff_frequency)
restored_fft_shifted_gaussian = image_fft_shifted / (gaussian_filter_mask + 1e-5)
restored_fft_gaussian = np.fft.ifftshift(restored_fft_shifted_gaussian)
restored_image_gaussian = np.fft.ifft2(restored_fft_gaussian)
restored_image_gaussian = np.abs(restored_image_gaussian)
cv2.imwrite('restored_image_gaussian.png', restored_image_gaussian)

# 应用巴特沃斯低通滤波器
butterworth_filter_mask = butterworth_low_pass_filter(image.shape, cutoff_frequency, order)
restored_fft_shifted_butterworth = image_fft_shifted / (butterworth_filter_mask + 1e-5)
restored_fft_butterworth = np.fft.ifftshift(restored_fft_shifted_butterworth)
restored_image_butterworth = np.fft.ifft2(restored_fft_butterworth)
restored_image_butterworth = np.abs(restored_image_butterworth)
cv2.imwrite('restored_image_butterworth.png', restored_image_butterworth)

# 应用运动模糊复原
motion_length = 30  # 模糊长度
motion_angle = 45  # 模糊角度
motion_psf = motion_blur_psf(image.shape, motion_length, motion_angle)

# 将 PSF 转换到频域
motion_psf_fft = np.fft.fft2(motion_psf, s=image.shape)

# 进行运动模糊复原
restored_fft_motion = image_fft / (motion_psf_fft + 1e-5)  # 避免除以零
restored_fft_motion_shifted = np.fft.ifftshift(restored_fft_motion)
restored_image_motion = np.fft.ifft2(restored_fft_motion_shifted)
restored_image_motion = np.abs(restored_image_motion)

cv2.imwrite('restored_image_motion.png', restored_image_motion)
#使用多种复原方法，效果依然不良好

