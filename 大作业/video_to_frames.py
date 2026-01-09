import cv2
import os

os.makedirs('./data/video_frames/input1', exist_ok=True)
os.makedirs('./data/video_frames/input2', exist_ok=True)
os.makedirs('./data/video_frames/input3', exist_ok=True)
os.makedirs('./data/video_frames/input4', exist_ok=True)

# 配置路径
video1_path = './input/1.mp4'  # 修改为你实际的视频路径
video2_path = './input/2.mp4'
video3_path = './input/3.mp4'  
video4_path = './input/4.mp4'
output_dir1 = './data/video_frames/input1'
output_dir2 = './data/video_frames/input2'
output_dir3 = './data/video_frames/input3'
output_dir4 = './data/video_frames/input4'

# 确保目录存在
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    print(f"正在处理 {video_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # UDIS通常需要固定尺寸，为了效果最好，建议resize为512x512或保留原比例
        # 这里我们先resize到512x512 (UDIS默认训练尺寸)，之后再resize回去或者直接拼接
        # 如果显存够大（5070Ti绝对够），可以尝试更高分辨率，但为了跑通代码，建议先用512
        frame = cv2.resize(frame, (512, 512)) 
        
        # 保存文件名必须一一对应，比如 000001.jpg
        filename = f"{idx:06d}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        idx += 1
    cap.release()
    print(f"完成，共 {idx} 帧保存至 {save_dir}")

process_video(video1_path, output_dir1)
process_video(video2_path, output_dir2)
process_video(video3_path, output_dir3)
process_video(video4_path, output_dir4)