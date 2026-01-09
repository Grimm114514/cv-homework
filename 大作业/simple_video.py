#!/usr/bin/env python3
# coding: utf-8
"""
简单的图片转视频工具，避免复杂的编码器问题
"""
import cv2
import os
import glob
import argparse


def images_to_video(image_folder, output_video, fps=30):
    """
    将文件夹中的图片转换为视频
    """
    # 查找所有图片
    pattern = os.path.join(image_folder, 'composition_*.jpg')
    image_files = sorted(glob.glob(pattern))
    
    if len(image_files) == 0:
        print(f"未找到图片: {pattern}")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张获取尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"无法读取: {image_files[0]}")
        return False
    
    h, w = first_img.shape[:2]
    
    # 确保尺寸为偶数
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1
    
    print(f"视频尺寸: {w}x{h}, 帧率: {fps}")
    
    # 使用更兼容的编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("无法创建视频文件")
        return False
    
    # 写入所有帧
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 跳过 {img_path}")
            continue
        
        # 调整尺寸
        img = img[:h, :w]
        out.write(img)
        
        if (i + 1) % 50 == 0:
            print(f"进度: {i+1}/{len(image_files)}")
    
    out.release()
    print(f"\n成功生成视频: {output_video}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_video', type=str, required=True)
    parser.add_argument('--fps', type=int, default=30)
    
    args = parser.parse_args()
    images_to_video(args.input_folder, args.output_video, args.fps)
