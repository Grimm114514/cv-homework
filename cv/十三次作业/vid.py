import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MotionAnalyzer:
    """运动能量图(MEI)和运动历史图(MHI)分析器"""
    
    def __init__(self, threshold=25, mhi_duration=30):
        """
        初始化运动分析器
        
        参数:
            threshold: 帧差阈值，用于检测运动
            mhi_duration: MHI的持续时间（帧数）
        """
        self.threshold = threshold
        self.mhi_duration = mhi_duration
        
    def compute_mei_mhi(self, video_path):
        """
        计算视频的运动能量图和运动历史图
        
        参数:
            video_path: 视频文件路径
            
        返回:
            mei: 运动能量图 (Motion Energy Image)
            mhi: 运动历史图 (Motion History Image)
            frame_count: 总帧数
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("无法读取视频帧")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape
        
        # 初始化MEI和MHI
        mei = np.zeros((h, w), dtype=np.uint8)
        mhi = np.zeros((h, w), dtype=np.float32)
        
        frame_count = 0
        timestamp = 0
        
        print(f"\n处理视频: {Path(video_path).name}")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count
            
            # 转换为灰度图
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算帧差
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            
            # 二值化得到运动区域
            _, motion_mask = cv2.threshold(frame_diff, self.threshold, 1, cv2.THRESH_BINARY)
            
            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            # 更新MEI（运动能量图）- 所有运动区域的累积
            mei = cv2.bitwise_or(mei, motion_mask)
            
            # 更新MHI（运动历史图）- 带时间戳的运动历史
            # 在有运动的地方更新为当前时间戳
            mhi[motion_mask == 1] = timestamp
            
            # MHI衰减：移除过旧的运动历史
            mhi[mhi < (timestamp - self.mhi_duration)] = 0
            
            prev_gray = curr_gray
            
            if frame_count % 10 == 0:
                print(f"已处理 {frame_count} 帧...")
        
        cap.release()
        
        print(f"处理完成！总帧数: {frame_count}")
        print("=" * 60)
        
        # 归一化MHI到0-255范围以便显示
        if mhi.max() > 0:
            mhi_normalized = ((mhi / mhi.max()) * 255).astype(np.uint8)
        else:
            mhi_normalized = mhi.astype(np.uint8)
        
        # MEI转换为0-255
        mei = mei * 255
        
        return mei, mhi_normalized, frame_count
    
    def visualize_results(self, video_path, mei, mhi, save_path=None):
        """
        可视化MEI和MHI结果
        
        参数:
            video_path: 视频路径
            mei: 运动能量图
            mhi: 运动历史图
            save_path: 保存路径（可选）
        """
        # 读取第一帧和最后一帧作为参考
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        
        # 跳到最后一帧
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        cap.release()
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        video_name = Path(video_path).stem
        
        # 第一帧
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        plt.title('第一帧', fontsize=12)
        plt.axis('off')
        
        # 最后一帧
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        plt.title('最后一帧', fontsize=12)
        plt.axis('off')
        
        # MEI
        plt.subplot(2, 3, 3)
        plt.imshow(mei, cmap='hot')
        plt.title('运动能量图 (MEI)', fontsize=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # MHI
        plt.subplot(2, 3, 4)
        plt.imshow(mhi, cmap='jet')
        plt.title('运动历史图 (MHI)', fontsize=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # MEI叠加在第一帧上
        plt.subplot(2, 3, 5)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        mei_colored = cv2.applyColorMap(mei, cv2.COLORMAP_HOT)
        mei_colored_rgb = cv2.cvtColor(mei_colored, cv2.COLOR_BGR2RGB)
        overlay_mei = cv2.addWeighted(first_frame_rgb, 0.6, mei_colored_rgb, 0.4, 0)
        plt.imshow(overlay_mei)
        plt.title('MEI 叠加图', fontsize=12)
        plt.axis('off')
        
        # MHI叠加在第一帧上
        plt.subplot(2, 3, 6)
        mhi_colored = cv2.applyColorMap(mhi, cv2.COLORMAP_JET)
        mhi_colored_rgb = cv2.cvtColor(mhi_colored, cv2.COLOR_BGR2RGB)
        overlay_mhi = cv2.addWeighted(first_frame_rgb, 0.6, mhi_colored_rgb, 0.4, 0)
        plt.imshow(overlay_mhi)
        plt.title('MHI 叠加图', fontsize=12)
        plt.axis('off')
        
        plt.suptitle(f'视频运动分析: {video_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def compare_videos(self, video_paths, labels):
        """
        对比多个视频的MEI和MHI
        
        参数:
            video_paths: 视频路径列表
            labels: 视频标签列表
        """
        results = []
        
        for video_path in video_paths:
            mei, mhi, frame_count = self.compute_mei_mhi(video_path)
            results.append((mei, mhi, frame_count))
        
        # 可视化对比
        fig = plt.figure(figsize=(15, 5 * len(video_paths)))
        
        for idx, (video_path, label, (mei, mhi, frame_count)) in enumerate(zip(video_paths, labels, results)):
            # MEI
            plt.subplot(len(video_paths), 2, idx * 2 + 1)
            plt.imshow(mei, cmap='hot')
            plt.title(f'{label} - MEI (共{frame_count}帧)', fontsize=12)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # MHI
            plt.subplot(len(video_paths), 2, idx * 2 + 2)
            plt.imshow(mhi, cmap='jet')
            plt.title(f'{label} - MHI', fontsize=12)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
        
        plt.suptitle('多视频运动分析对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def process_single_video(video_path):
    """处理单个视频"""
    analyzer = MotionAnalyzer(threshold=25, mhi_duration=30)
    
    try:
        mei, mhi, frame_count = analyzer.compute_mei_mhi(video_path)
        
        # 保存结果
        video_stem = Path(video_path).stem
        output_dir = Path(video_path).parent
        
        cv2.imwrite(str(output_dir / f"{video_stem}_MEI.png"), mei)
        cv2.imwrite(str(output_dir / f"{video_stem}_MHI.png"), mhi)
        print(f"\n已保存MEI和MHI图像到当前目录")
        
        # 可视化
        save_path = output_dir / f"{video_stem}_analysis.png"
        analyzer.visualize_results(video_path, mei, mhi, save_path)
        
    except Exception as e:
        print(f"处理视频时出错: {e}")


def process_multiple_videos(video_paths, labels):
    """处理多个视频并对比"""
    analyzer = MotionAnalyzer(threshold=25, mhi_duration=30)
    
    try:
        analyzer.compare_videos(video_paths, labels)
        
        # 分别保存每个视频的结果
        for video_path, label in zip(video_paths, labels):
            mei, mhi, _ = analyzer.compute_mei_mhi(video_path)
            video_stem = Path(video_path).stem
            output_dir = Path(video_path).parent
            
            cv2.imwrite(str(output_dir / f"{video_stem}_MEI.png"), mei)
            cv2.imwrite(str(output_dir / f"{video_stem}_MHI.png"), mhi)
        
        print(f"\n所有视频的MEI和MHI图像已保存")
        
    except Exception as e:
        print(f"处理视频时出错: {e}")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("\n" + "=" * 60)
    print("运动能量图(MEI)与运动历史图(MHI)分析器")
    print("=" * 60)
    print("\n说明：")
    print("  - MEI (Motion Energy Image): 显示视频中所有发生运动的区域")
    print("  - MHI (Motion History Image): 显示运动的时序信息，亮度表示运动的新旧")
    print("\n使用方法：")
    print("  1. 准备两段视频: '站立到坐下.mp4' 和 '坐下到站立.mp4'")
    print("  2. 将视频放在当前目录下")
    print("  3. 运行本程序")
    print("=" * 60)
    
    # 视频文件路径（请根据实际情况修改）
    current_dir = Path(__file__).parent
    
    # 方式1: 处理单个视频
    # video_path = current_dir / "站立到坐下.mp4"
    # if video_path.exists():
    #     process_single_video(str(video_path))
    # else:
    #     print(f"\n错误: 找不到视频文件 {video_path}")
    
    # 方式2: 处理并对比两个视频
    video1 = current_dir / "1.mp4"
    video2 = current_dir / "2.mp4"
    
    # 检查视频文件是否存在
    videos_exist = []
    video_paths = []
    labels = []
    
    if video1.exists():
        videos_exist.append(True)
        video_paths.append(str(video1))
        labels.append("站立到坐下")
    else:
        print(f"\n提示: 未找到视频 '{video1.name}'")
    
    if video2.exists():
        videos_exist.append(True)
        video_paths.append(str(video2))
        labels.append("坐下到站立")
    else:
        print(f"\n提示: 未找到视频 '{video2.name}'")
    
    if len(video_paths) > 0:
        print(f"\n找到 {len(video_paths)} 个视频文件，开始处理...\n")
        
        if len(video_paths) == 1:
            # 只有一个视频
            process_single_video(video_paths[0])
        else:
            # 有多个视频，进行对比
            process_multiple_videos(video_paths, labels)
    else:
        print("\n" + "=" * 60)
        print("未找到视频文件！")
        print("\n请执行以下步骤：")
        print("1. 使用手机或摄像头拍摄两段视频：")
        print("   - '站立到坐下.mp4': 拍摄从站立姿势到坐下的过程")
        print("   - '坐下到站立.mp4': 拍摄从坐下姿势到站立的过程")
        print(f"2. 将视频文件放在目录: {current_dir}")
        print("3. 重新运行本程序")
        print("=" * 60)
        
        # 演示模式：创建示例说明
        print("\n或者修改代码中的视频路径，指向您的视频文件位置")
