#!/usr/bin/env python3
# coding: utf-8
"""
分批处理版本的 test_output.py
避免大规模图片处理时显存溢出
"""
import os
import sys
import shutil
import glob
import argparse
import subprocess
import torch


def split_and_process(args):
    """
    将输入分批处理，每批处理指定数量的图片
    """
    # 获取输入文件夹中的图片数量
    input1_path = os.path.join(args.test_path, args.input1_name)
    input2_path = os.path.join(args.test_path, args.input2_name)
    
    input1_files = sorted(glob.glob(os.path.join(input1_path, '*.jpg')))
    input2_files = sorted(glob.glob(os.path.join(input2_path, '*.jpg')))
    
    total_files = len(input1_files)
    print(f"总共找到 {total_files} 张图片")
    print(f"每批处理 {args.batch_size} 张")
    print(f"总批次: {(total_files + args.batch_size - 1) // args.batch_size}")
    print("=" * 60)
    
    # 创建临时文件夹
    temp_path = os.path.join(os.path.dirname(args.test_path), 'temp_batch')
    os.makedirs(temp_path, exist_ok=True)
    temp_input1 = os.path.join(temp_path, args.input1_name)
    temp_input2 = os.path.join(temp_path, args.input2_name)
    
    # 创建最终输出文件夹
    os.makedirs(args.output_path, exist_ok=True)
    for subfolder in ['warp1', 'warp2', 'mask1', 'mask2', 'ave_fusion']:
        os.makedirs(os.path.join(args.output_path, subfolder), exist_ok=True)
    
    # 分批处理
    batch_num = 0
    for start_idx in range(0, total_files, args.batch_size):
        batch_num += 1
        end_idx = min(start_idx + args.batch_size, total_files)
        
        print(f"\n{'='*60}")
        print(f"处理批次 {batch_num}: 图片 {start_idx+1}-{end_idx} / {total_files}")
        print(f"{'='*60}")
        
        # 清理临时文件夹
        if os.path.exists(temp_input1):
            shutil.rmtree(temp_input1)
        if os.path.exists(temp_input2):
            shutil.rmtree(temp_input2)
        os.makedirs(temp_input1)
        os.makedirs(temp_input2)
        
        # 复制当前批次的文件到临时文件夹
        print(f"准备批次 {batch_num} 的输入文件...")
        for i, idx in enumerate(range(start_idx, end_idx)):
            # 使用序号命名，从1开始
            src1 = input1_files[idx]
            src2 = input2_files[idx]
            dst1 = os.path.join(temp_input1, f'{i+1:06d}.jpg')
            dst2 = os.path.join(temp_input2, f'{i+1:06d}.jpg')
            shutil.copy2(src1, dst1)
            shutil.copy2(src2, dst2)
        
        # 强制清理 GPU 显存
        print("批次开始前清理 GPU 显存...")
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # 重置峰值显存统计
            torch.cuda.reset_peak_memory_stats()
        
        # 创建临时输出文件夹
        temp_output = os.path.join(temp_path, 'output')
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
        os.makedirs(temp_output)
        
        # 调用 test_output.py 处理当前批次
        python_cmd = sys.executable if sys.executable else 'python'
        cmd = [
            python_cmd, 'test_output.py',
            '--test_path', temp_path,
            '--input1_name', args.input1_name,
            '--input2_name', args.input2_name,
            '--output_path', temp_output,
            '--gpu', args.gpu
        ]
        
        print(f"运行 test_output.py...")
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.')
        
        if result.returncode != 0:
            print(f"批次 {batch_num} 处理失败!")
            return False
        
        # 将结果移动到最终输出文件夹（重新编号）
        print(f"合并批次 {batch_num} 的输出...")
        for subfolder in ['warp1', 'warp2', 'mask1', 'mask2', 'ave_fusion']:
            temp_subfolder = os.path.join(temp_output, subfolder)
            final_subfolder = os.path.join(args.output_path, subfolder)
            
            if not os.path.exists(temp_subfolder):
                continue
            
            temp_files = sorted(glob.glob(os.path.join(temp_subfolder, '*.jpg')))
            for i, temp_file in enumerate(temp_files):
                # 使用全局索引编号
                global_idx = start_idx + i + 1
                final_file = os.path.join(final_subfolder, f'{global_idx:06d}.jpg')
                shutil.move(temp_file, final_file)
        
        print(f"批次 {batch_num} 完成!")
        
        # 强制清理 GPU 显存和Python垃圾回收
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()  # 同步CUDA操作
        print(f"已清理显存，等待2秒...")
        import time
        time.sleep(2)
    
    # 清理临时文件夹
    print(f"\n{'='*60}")
    print("清理临时文件...")
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    print(f"{'='*60}")
    print(f"✅ 全部完成! 共处理 {total_files} 张图片，分 {batch_num} 批")
    print(f"输出路径: {args.output_path}")
    print(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分批处理版本的 test_output.py')
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--test_path', type=str, required=True,
                        help='包含输入文件夹的路径')
    parser.add_argument('--input1_name', type=str, default='input1',
                        help='第一个输入文件夹名称')
    parser.add_argument('--input2_name', type=str, default='input2',
                        help='第二个输入文件夹名称')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出路径')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='每批处理的图片数量（默认50张）')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('分批处理图像拼接')
    print('=' * 60)
    print(f"输入路径: {args.test_path}")
    print(f"  - 文件夹1: {args.input1_name}")
    print(f"  - 文件夹2: {args.input2_name}")
    print(f"输出路径: {args.output_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"GPU: {args.gpu}")
    print('=' * 60)
    
    success = split_and_process(args)
    sys.exit(0 if success else 1)
