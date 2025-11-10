import cv2
import numpy as np
import os

def dct_compression(input_file, output_file):
    # 读取输入图像，转换为灰度图像
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # 获取图像的高度和宽度
    h, w = img.shape

    # 确保图像的尺寸是8的倍数，不足的部分进行填充
    h_pad = (8 - h % 8) % 8  # 高度需要填充的像素数
    w_pad = (8 - w % 8) % 8  # 宽度需要填充的像素数
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

    # 将图像划分为8x8的块，并对每个块应用DCT（离散余弦变换）
    h_blocks, w_blocks = img_padded.shape[0] // 8, img_padded.shape[1] // 8  # 计算块的数量
    dct_blocks = np.zeros_like(img_padded, dtype=np.float32)  # 存储DCT变换后的结果

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = img_padded[i*8:(i+1)*8, j*8:(j+1)*8]  # 提取8x8的块
            dct_block = cv2.dct(np.float32(block))  # 对块进行DCT变换
            dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8] = dct_block

    # 更新的量化矩阵，用于更高的压缩率
    Q = np.array([
        [32, 22, 20, 32, 48, 80, 102, 122],
        [24, 24, 28, 38, 52, 116, 120, 110],
        [28, 26, 32, 48, 80, 114, 138, 112],
        [28, 34, 44, 58, 102, 174, 160, 124],
        [36, 44, 74, 112, 136, 218, 206, 154],
        [48, 70, 110, 128, 162, 208, 226, 184],
        [98, 128, 156, 174, 206, 242, 240, 202],
        [144, 184, 190, 196, 224, 200, 206, 198]
    ])

    # 对DCT系数进行量化处理
    quantized_blocks = np.zeros_like(dct_blocks, dtype=np.float32)  # 存储量化后的结果
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8]  # 提取DCT变换后的8x8块
            quantized_block = np.round(block / Q) * Q  # 量化处理
            quantized_blocks[i*8:(i+1)*8, j*8:(j+1)*8] = quantized_block

    # 使用逆DCT（IDCT）重建图像
    reconstructed = np.zeros_like(img_padded, dtype=np.float32)  # 存储重建后的图像

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = quantized_blocks[i*8:(i+1)*8, j*8:(j+1)*8]  # 提取量化后的8x8块
            idct_block = cv2.idct(block)  # 对块进行逆DCT变换
            reconstructed[i*8:(i+1)*8, j*8:(j+1)*8] = idct_block

    # 去除填充部分，恢复到原始图像的大小
    reconstructed = reconstructed[:h, :w]

    # 保存压缩后的图像，并指定JPEG的压缩质量
    cv2.imwrite(output_file, np.uint8(np.clip(reconstructed, 0, 255)), [cv2.IMWRITE_JPEG_QUALITY, 50])

if __name__ == "__main__":
    input_file = '1.png'  # 输入文件名
    output_file = 'compressed_1.png'  # 输出文件名
    dct_compression(input_file, output_file)  # 调用DCT压缩函数
    print(f"Compressed image saved as '{output_file}'")  # 打印保存成功的提示