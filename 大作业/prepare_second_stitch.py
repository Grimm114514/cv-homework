#!/usr/bin/env python3
# coding: utf-8
"""
准备二次拼接：将第一次拼接的结果复制到 video_frames 下，以便进行第二次拼接
"""
import os
import shutil
import glob
import argparse
import cv2
import numpy as np


def prepare_folders(args):
    """
    将生成的图片复制到新文件夹，重命名为可以被 dataset 读取的格式
    支持 warp_result_*.jpg 和 composition_*.jpg 两种命名格式
    """
    # 创建目标文件夹
    target_folder1 = os.path.join(args.target_path, args.folder1_name)
    target_folder2 = os.path.join(args.target_path, args.folder2_name)
    
    os.makedirs(target_folder1, exist_ok=True)
    os.makedirs(target_folder2, exist_ok=True)
    
    # 复制 folder1 的图片 - 尝试三种命名格式
    source1_pattern1 = os.path.join(args.source1, 'warp_result_*.jpg')
    source1_pattern2 = os.path.join(args.source1, 'composition_*.jpg')
    source1_pattern3 = os.path.join(args.source1, '*.jpg')
    
    source1_files = sorted(glob.glob(source1_pattern1))
    if len(source1_files) == 0:
        source1_files = sorted(glob.glob(source1_pattern2))
    if len(source1_files) == 0:
        source1_files = sorted(glob.glob(source1_pattern3))
    
    if len(source1_files) == 0:
        print(f"错误: 未找到文件 {source1_pattern1}, {source1_pattern2} 或 {source1_pattern3}")
        return
    
    print(f"正在复制 {len(source1_files)} 张图片从 {args.source1} 到 {target_folder1}")
    for i, src_file in enumerate(source1_files):
        # 读取图片
        img = cv2.imread(src_file)
        if img is None:
            print(f"警告: 无法读取 {src_file}")
            continue
        
        # 保持宽高比 resize 到 512x512，填充黑边
        h, w = img.shape[:2]
        scale = min(512 / w, 512 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 创建 512x512 的黑色画布
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 计算居中位置
        x_offset = (512 - new_w) // 2
        y_offset = (512 - new_h) // 2
        
        # 将图片放到画布中心
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # 保存
        dst_file = os.path.join(target_folder1, f'{i+1:06d}.jpg')
        cv2.imwrite(dst_file, canvas)
        
        if (i + 1) % 50 == 0:
            print(f"  已复制: {i+1}/{len(source1_files)}")
    
    # 复制 folder2 的图片 - 尝试三种命名格式
    source2_pattern1 = os.path.join(args.source2, 'warp_result_*.jpg')
    source2_pattern2 = os.path.join(args.source2, 'composition_*.jpg')
    source2_pattern3 = os.path.join(args.source2, '*.jpg')
    
    source2_files = sorted(glob.glob(source2_pattern1))
    if len(source2_files) == 0:
        source2_files = sorted(glob.glob(source2_pattern2))
    if len(source2_files) == 0:
        source2_files = sorted(glob.glob(source2_pattern3))
    
    if len(source2_files) == 0:
        print(f"错误: 未找到文件 {source2_pattern1}, {source2_pattern2} 或 {source2_pattern3}")
        return
    
    print(f"正在复制 {len(source2_files)} 张图片从 {args.source2} 到 {target_folder2}")
    for i, src_file in enumerate(source2_files):
        # 读取图片
        img = cv2.imread(src_file)
        if img is None:
            print(f"警告: 无法读取 {src_file}")
            continue
        
        # 保持宽高比 resize 到 512x512，填充黑边
        h, w = img.shape[:2]
        scale = min(512 / w, 512 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 创建 512x512 的黑色画布
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 计算居中位置
        x_offset = (512 - new_w) // 2
        y_offset = (512 - new_h) // 2
        
        # 将图片放到画布中心
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        dst_file = os.path.join(target_folder2, f'{i+1:06d}.jpg')
        cv2.imwrite(dst_file, canvas)
        
        if (i + 1) % 50 == 0:
            print(f"  已复制: {i+1}/{len(source2_files)}")
    
    print(f"\n准备完成！")
    print(f"现在可以运行:")
    print(f"python test.py --test_path {args.target_path} --input1_name {args.folder1_name} --input2_name {args.folder2_name} --output_path {args.output_path} --gpu 0 --generate_video --fps 30")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备二次拼接的输入文件')
    
    parser.add_argument('--source1', type=str, required=True,
                        help='第一个源文件夹路径（包含 warp_result_*.jpg）')
    parser.add_argument('--source2', type=str, required=True,
                        help='第二个源文件夹路径（包含 warp_result_*.jpg）')
    parser.add_argument('--target_path', type=str, default='../../data/second_stitch/',
                        help='目标路径（将创建子文件夹）')
    parser.add_argument('--folder1_name', type=str, default='left',
                        help='第一个文件夹的名称')
    parser.add_argument('--folder2_name', type=str, default='right',
                        help='第二个文件夹的名称')
    parser.add_argument('--output_path', type=str, default='../../data/results/final/',
                        help='最终拼接结果的输出路径')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('准备二次拼接输入文件')
    print('=' * 60)
    print(f"源文件夹1: {args.source1}")
    print(f"源文件夹2: {args.source2}")
    print(f"目标路径: {args.target_path}")
    print(f"  - {args.folder1_name}/")
    print(f"  - {args.folder2_name}/")
    print('=' * 60)
    print()
    
    prepare_folders(args)
