import cv2
import numpy as np  

input_file = '2.png'

def denoise(image_path, output_path='denoised.png'):
    # 读取灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read input image: {image_path}")
        return
    


    # 转换为二值图像
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # 进行开运算
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(output_path, opened)


if __name__ == '__main__':
    denoise(input_file)