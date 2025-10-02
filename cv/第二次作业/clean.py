import cv2

def gaussian_blur(image_path, kernel_size=(3,3), sigma=1.5):
   """
   参数:
   image_path: str, 图像文件路径
   kernel_size: tuple, 高斯核大小
   sigma: float, 高斯核标准差
   """
   #读取图像
   image = cv2.imread(image_path)
   if image is None:
       print("图像路径无效或图像无法读取")
   #应用高斯滤波
   blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
   return image, blurred_image

def bilateral_filter (image_path, d=9, sigma_color=20, sigma_space=75):
   """
   参数:
   image_path: str, 图像文件路径
   d: int, 每个像素邻域直径
   sigma_color: float, 颜色空间标准差
   sigma_space: float, 坐标空间标准差
   """
   #读取图像
   image = cv2.imread(image_path)
   if image is None:
       print("图像路径无效或图像无法读取")
   #应用双边滤波
   filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
   return image, filtered_image




if __name__ == "__main__":
    image_path = '1.png'  
    original, blurred = gaussian_blur(image_path)
    cv2.imwrite('blurred_1.png', blurred)

    original, filtered = bilateral_filter(image_path)
    cv2.imwrite('filtered_1.png', filtered)

    