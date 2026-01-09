# coding: utf-8
import argparse
import torch
from UDIS2.Composition.Codes.network import build_model, Network
import cv2
import os
import numpy as np
import glob
import re


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'UDIS2/Composition/model')


def load_frame_pair(warp1_path, warp2_path, mask1_path, mask2_path):
    """
    加载单帧的图像对和掩码
    """
    # load image1
    warp1 = cv2.imread(warp1_path)
    if warp1 is None:
        return None
    warp1 = warp1.astype(dtype=np.float32)
    warp1 = (warp1 / 127.5) - 1.0
    warp1 = np.transpose(warp1, [2, 0, 1])

    # load image2
    warp2 = cv2.imread(warp2_path)
    if warp2 is None:
        return None
    warp2 = warp2.astype(dtype=np.float32)
    warp2 = (warp2 / 127.5) - 1.0
    warp2 = np.transpose(warp2, [2, 0, 1])

    # load mask1
    mask1 = cv2.imread(mask1_path)
    if mask1 is None:
        return None
    mask1 = mask1.astype(dtype=np.float32)
    mask1 = mask1 / 255
    mask1 = np.transpose(mask1, [2, 0, 1])

    # load mask2
    mask2 = cv2.imread(mask2_path)
    if mask2 is None:
        return None
    mask2 = mask2.astype(dtype=np.float32)
    mask2 = mask2 / 255
    mask2 = np.transpose(mask2, [2, 0, 1])

    # convert to tensor
    warp1_tensor = torch.tensor(warp1).unsqueeze(0)
    warp2_tensor = torch.tensor(warp2).unsqueeze(0)
    mask1_tensor = torch.tensor(mask1).unsqueeze(0)
    mask2_tensor = torch.tensor(mask2).unsqueeze(0)

    return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor


def extract_frame_number(filename):
    """
    从文件名中提取帧编号
    例如: warp_result_1366.jpg -> 1366
    """
    match = re.search(r'_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return None


def composition_and_video(args):
    """
    对图片序列进行深度学习合成，并输出视频
    支持test_output.py生成的子文件夹结构: warp1/, warp2/, mask1/, mask2/
    """
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 加载模型
    print("正在加载模型...")
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) == 0:
        print('未找到模型文件!')
        return
    
    model_path = ckpt_list[-1]
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model'])
    print(f'已加载模型: {model_path}')
    net.eval()
    
    # 查找warp1图片文件（从warp1/子文件夹）
    warp1_folder = os.path.join(args.input_folder, 'warp1')
    if not os.path.exists(warp1_folder):
        print(f"错误: 未找到warp1文件夹: {warp1_folder}")
        print("请确保使用test_output.py生成包含warp1/, warp2/, mask1/, mask2/的输出")
        return
    
    warp1_pattern = os.path.join(warp1_folder, '*.jpg')
    warp1_files = sorted(glob.glob(warp1_pattern))
    
    if len(warp1_files) == 0:
        print(f"未找到warp1图片: {warp1_pattern}")
        return
    
    print(f"找到 {len(warp1_files)} 帧图片")
    
    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 处理第一帧获取视频尺寸
    print("正在处理第一帧以获取视频尺寸...")
    first_warp1 = warp1_files[0]
    basename = os.path.basename(first_warp1)
    
    # 从子文件夹读取对应文件
    warp2_file = os.path.join(args.input_folder, 'warp2', basename)
    mask1_file = os.path.join(args.input_folder, 'mask1', basename)
    mask2_file = os.path.join(args.input_folder, 'mask2', basename)
    
    data = load_frame_pair(first_warp1, warp2_file, mask1_file, mask2_file)
    if data is None:
        print("无法加载第一帧数据")
        return
    
    warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor = data
    if torch.cuda.is_available():
        warp1_tensor = warp1_tensor.cuda()
        warp2_tensor = warp2_tensor.cuda()
        mask1_tensor = mask1_tensor.cuda()
        mask2_tensor = mask2_tensor.cuda()
    
    with torch.no_grad():
        batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    stitched_image = batch_out['stitched_image']
    stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
    
    height, width = stitched_image.shape[:2]
    
    # 确保尺寸是偶数（某些编码器要求）
    if width % 2 != 0:
        width = width - 1
        stitched_image = stitched_image[:, :width, :]
    if height % 2 != 0:
        height = height - 1
        stitched_image = stitched_image[:height, :, :]
    
    print(f"输出图片尺寸: {width}x{height}")
    
    # 根据参数决定是否生成视频
    video_writer = None
    video_path = None
    
    if not args.no_video:
        # 创建视频写入器 - 使用更稳定的编码器
        # 尝试多个编码器：XVID > avc1 > mp4v
        video_path = os.path.join(args.output_folder, args.output_name)
        fourcc_list = [
            ('XVID', 'XVID'),
            ('avc1', 'H264'),
            ('mp4v', 'MPEG-4'),
        ]
        
        for fourcc_code, codec_name in fourcc_list:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
            if video_writer.isOpened():
                print(f"使用编码器: {codec_name} ({fourcc_code})")
                break
            video_writer.release()
        
        if video_writer is None or not video_writer.isOpened():
            print("警告: 无法创建视频文件，将只保存图片")
            video_writer = None
    else:
        print("跳过视频生成，仅保存图片帧")
    
    # 写入第一帧
    if video_writer is not None:
        video_writer.write(stitched_image)
    
    # 处理剩余帧
    print("开始批量处理...")
    for i, warp1_file in enumerate(warp1_files):
        basename = os.path.basename(warp1_file)
        
        # 从子文件夹读取对应文件
        warp2_file = os.path.join(args.input_folder, 'warp2', basename)
        mask1_file = os.path.join(args.input_folder, 'mask1', basename)
        mask2_file = os.path.join(args.input_folder, 'mask2', basename)
        
        # 加载数据
        data = load_frame_pair(warp1_file, warp2_file, mask1_file, mask2_file)
        if data is None:
            print(f"警告: 无法加载第 {i+1} 帧 ({basename})，跳过")
            continue
        
        warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor = data
        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()
        
        # 模型推理
        with torch.no_grad():
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
        
        stitched_image = batch_out['stitched_image']
        stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
        
        # 裁剪到正确尺寸
        stitched_image = stitched_image[:height, :width, :]
        
        # 保存单帧图片（可选）
        if args.save_frames:
            # 使用原始文件名（去掉扩展名）
            frame_name = os.path.splitext(basename)[0]
            frame_output = os.path.join(args.output_folder, f'composition_{frame_name}.jpg')
            cv2.imwrite(frame_output, stitched_image)
        
        # 写入视频
        if video_writer is not None:
            video_writer.write(stitched_image)
        
        if (i + 1) % 10 == 0:
            print(f"已处理: {i+1}/{len(warp1_files)} 帧")
    
    if video_writer is not None:
        video_writer.release()
        print(f"\n视频已保存至: {video_path}")
    else:
        print(f"\n图片已保存至: {args.output_folder}")
    
    print(f"总帧数: {len(warp1_files)}")
    if not args.no_video:
        print(f"帧率: {args.fps} fps")
        print(f"时长: {len(warp1_files)/args.fps:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量进行图像合成并生成视频')
    
    parser.add_argument('--input_folder', type=str, 
                        default='../../data/results/output1_2/',
                        help='输入文件夹路径（包含warp1/、warp2/、mask1/、mask2/子文件夹）')
    parser.add_argument('--output_folder', type=str, 
                        default='../../data/results/composition/',
                        help='输出文件夹路径')
    parser.add_argument('--output_name', type=str, 
                        default='composition_video.mp4',
                        help='输出视频文件名')
    parser.add_argument('--fps', type=int, 
                        default=30,
                        help='视频帧率')
    parser.add_argument('--gpu', type=str, 
                        default='0',
                        help='使用的GPU编号')
    parser.add_argument('--save_frames', action='store_true',
                        help='是否保存每一帧的合成图片')
    parser.add_argument('--no_video', action='store_true',
                        help='不生成视频，仅保存图片帧（避免FFmpeg错误）')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('深度学习图像合成 + 视频生成')
    print('=' * 60)
    print(f"输入文件夹: {args.input_folder}")
    print(f"  - 需要包含: warp1/, warp2/, mask1/, mask2/ 子文件夹")
    print(f"输出文件夹: {args.output_folder}")
    print(f"输出视频: {args.output_name}")
    print(f"帧率: {args.fps} fps")
    print(f"GPU: {args.gpu}")
    print(f"保存单帧: {args.save_frames}")
    print('=' * 60)
    print()
    
    composition_and_video(args)
