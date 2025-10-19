import cv2
import numpy as np

def inverse_filtering(image, psf, eps=1e-3):
    # 将图像和 PSF 转换到频域
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)
    
    # 避免除以零，添加一个小的正则化项 eps
    psf_fft = psf_fft + eps
    
    # 逆滤波公式：F_restored = F_observed / H
    restored_fft = image_fft / psf_fft
    
    # 逆傅里叶变换回到空间域
    restored_image = np.fft.ifft2(restored_fft)
    restored_image = np.abs(restored_image)
    
    return restored_image

input_file = 'house.png'

image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# 定义一个简单的运动模糊 PSF
psf = np.zeros_like(image)
psf[int(image.shape[0] / 2), int(image.shape[1] / 2 - 5):int(image.shape[1] / 2 + 5)] = 1
psf = psf / psf.sum()

# 应用逆滤波
restored_image = inverse_filtering(image, psf)

# 保存结果
cv2.imwrite('restored_image.png', restored_image)