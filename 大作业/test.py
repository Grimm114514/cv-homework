# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from UDIS2.Composition.Codes.network import build_model, Network
from dataset import *
import os
import numpy as np
import skimage
import cv2


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset - pass custom input folder names to TestDataset
    test_data = TestDataset(data_path=args.test_path, input1_name=args.input1_name, input2_name=args.input2_name)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    
    # Create output directory if it doesn't exist
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    
    # Create output directory if it doesn't exist
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    net.eval()
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

            with torch.no_grad():
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

            warp_mesh_mask = batch_out['warp_mesh_mask']
            warp_mesh = batch_out['warp_mesh']


            warp_mesh_np = ((warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            warp_mesh_mask_np = warp_mesh_mask[0].cpu().detach().numpy().transpose(1,2,0)
            inpu1_np = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            
            # Save output images to specified directory
            output_filename = os.path.join(output_dir, f'warp_result_{i+1:04d}.jpg')
            # Save the warped image
            cv2.imwrite(output_filename, warp_mesh_np.astype(np.uint8))
            # Skip saving mask to save disk space
            # mask_filename = os.path.join(output_dir, f'warp_mask_{i+1:04d}.jpg')
            # cv2.imwrite(mask_filename, (warp_mesh_mask_np * 255).astype(np.uint8))

            # calculate psnr/ssim (using updated scikit-image API)
            psnr = skimage.metrics.peak_signal_noise_ratio(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255)
            ssim = skimage.metrics.structural_similarity(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255, channel_axis=2)


            print('i = {}, psnr = {:.6f}'.format( i+1, psnr))

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            torch.cuda.empty_cache()

    print("=================== Analysis ==================")
    print("psnr")
    psnr_list.sort(reverse = True)
    psnr_list_30 = psnr_list[0 : 331]
    psnr_list_60 = psnr_list[331: 663]
    psnr_list_100 = psnr_list[663: -1]
    print("top 30%", np.mean(psnr_list_30))
    print("top 30~60%", np.mean(psnr_list_60))
    print("top 60~100%", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse = True)
    ssim_list_30 = ssim_list[0 : 331]
    ssim_list_60 = ssim_list[331: 663]
    ssim_list_100 = ssim_list[663: -1]
    print("top 30%", np.mean(ssim_list_30))
    print("top 30~60%", np.mean(ssim_list_60))
    print("top 60~100%", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))
    print("##################end testing#######################")
    
    # Generate video from results if enabled
    if args.generate_video:
        print("\n=================== Generating Video ===================")
        generate_video_from_results(output_dir, args)


def generate_video_from_results(output_dir, args):
    """
    从生成的图片序列创建视频
    """
    import glob
    
    # 获取所有结果图片
    image_pattern = os.path.join(output_dir, 'warp_result_*.jpg')
    image_files = sorted(glob.glob(image_pattern))
    
    if len(image_files) == 0:
        print(f"未找到结果图片: {image_pattern}")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"无法读取图片: {image_files[0]}")
        return
    
    height, width = first_frame.shape[:2]
    
    # 确保尺寸是偶数
    if width % 2 != 0:
        width = width - 1
    if height % 2 != 0:
        height = height - 1
    
    print(f"视频尺寸: {width}x{height}")
    
    # 创建视频写入器
    video_filename = f'stitched_video_{args.input1_name}_{args.input2_name}.mp4'
    video_path = os.path.join(output_dir, video_filename)
    
    # 尝试多个编码器
    fourcc_list = [
        ('XVID', 'XVID'),
        ('avc1', 'H264'),
        ('mp4v', 'MPEG-4'),
    ]
    
    video_writer = None
    for fourcc_code, codec_name in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        if video_writer.isOpened():
            print(f"使用编码器: {codec_name} ({fourcc_code})")
            break
        video_writer.release()
    
    if video_writer is None or not video_writer.isOpened():
        print("无法创建视频文件")
        return
    
    # 写入所有帧
    for i, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"警告: 无法读取 {image_file}, 跳过")
            continue
        
        # 裁剪到正确尺寸
        frame = frame[:height, :width]
        video_writer.write(frame)
        
        if (i + 1) % 50 == 0:
            print(f"已写入: {i+1}/{len(image_files)} 帧")
    
    video_writer.release()
    print(f"\n视频已保存至: {video_path}")
    print(f"总帧数: {len(image_files)}")
    print(f"帧率: {args.fps} fps")
    print(f"时长: {len(image_files)/args.fps:.2f} 秒")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')
    # Add arguments for custom input folder names
    parser.add_argument('--input1_name', type=str, default='input1', help='Name of first input folder')
    parser.add_argument('--input2_name', type=str, default='input2', help='Name of second input folder')
    # Add argument for output directory
    parser.add_argument('--output_path', type=str, default='../../data/results/', help='Path to save output results')
    # Add video generation arguments
    parser.add_argument('--generate_video', action='store_true', help='Generate video from stitched images')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
