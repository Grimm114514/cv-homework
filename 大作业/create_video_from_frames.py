#!/usr/bin/env python3
# coding: utf-8
"""
从图片序列生成视频
"""
import os
import sys
import glob
import cv2
import argparse


def create_video_from_images(image_folder, output_video, fps=30):
    """
    从图片文件夹生成视频
    """
    # 获取所有图片
    pattern = os.path.join(image_folder, 'composition_*.jpg')
    image_files = sorted(glob.glob(pattern))
    
    if len(image_files) == 0:
        print(f"错误: 未找到图片 {pattern}")
        return False
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"错误: 无法读取 {image_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # 确保尺寸是偶数
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    
    print(f"视频尺寸: {width}x{height}")
    print(f"帧率: {fps} fps")
    print(f"输出: {output_video}")
    print("")
    
    # 创建视频写入器 - 使用XVID编码器（更稳定）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("错误: 无法创建视频文件")
        return False
    
    # 逐帧写入
    print("正在生成视频...")
    for i, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"警告: 无法读取 {image_file}, 跳过")
            continue
        
        # 裁剪到正确尺寸
        frame = frame[:height, :width]
        
        # 写入帧
        success = video_writer.write(frame)
        if not success:
            print(f"警告: 写入第 {i+1} 帧失败")
        
        if (i + 1) % 100 == 0:
            print(f"  已写入: {i+1}/{len(image_files)} 帧")
    
    # 释放资源
    video_writer.release()
    
    # 检查文件是否创建成功
    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n✅ 视频生成成功！")
        print(f"位置: {output_video}")
        print(f"大小: {file_size_mb:.2f} MB")
        print(f"总帧数: {len(image_files)}")
        print(f"时长: {len(image_files)/fps:.2f} 秒")
        return True
    else:
        print("\n❌ 视频生成失败")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从图片序列生成视频')
    parser.add_argument('image_folder', type=str, help='包含composition_*.jpg的文件夹')
    parser.add_argument('output_video', type=str, help='输出视频文件名')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率（默认30）')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('从图片序列生成视频')
    print('=' * 60)
    print(f"输入文件夹: {args.image_folder}")
    print(f"输出视频: {args.output_video}")
    print(f"帧率: {args.fps} fps")
    print('=' * 60)
    print()
    
    success = create_video_from_images(args.image_folder, args.output_video, args.fps)
    sys.exit(0 if success else 1)
