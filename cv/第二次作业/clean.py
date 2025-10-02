import cv2

input_file= '1.png'
output_file1= 'blurred_1.png'

kernel_size=(3,3)
denoise_image=cv2.blur(cv2.imread(input_file),kernel_size)

cv2.imwrite(output_file1,denoise_image)