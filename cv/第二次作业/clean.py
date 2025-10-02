import cv2

"""
低通滤波
"""
input_file = '1.png'

img = cv2.imread(input_file)
if img is None:
	print(f"Could not read input image: {input_file}")

kernel_size = (3, 3)# 核大小
denoised_blur = cv2.blur(img, kernel_size)#均值滤波
denoised_median = cv2.medianBlur(img, 3)#中值滤波

cv2.imwrite('denoised_blur.png', denoised_blur)
cv2.imwrite('denoised_median.png', denoised_median)         




