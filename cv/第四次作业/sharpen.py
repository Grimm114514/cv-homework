import cv2
import numpy as np


input_file = 'image.png'
output_file = 'sharpened_image.png'
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
kernel = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])

sharpened_image = cv2.filter2D(image, -1, kernel)
cv2.imwrite(output_file, sharpened_image)




