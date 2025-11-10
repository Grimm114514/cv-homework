from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

paths = ["photos/1.jpg", "photos/2.jpg", "photos/3.jpg", "photos/4.jpg", "photos/5.jpg"]

save_dir = "result"
os.makedirs(save_dir, exist_ok=True)

i = 1
for path in paths:
    results = model(path, conf=0.5)  # 设置置信度阈值
    for result in results:  # 遍历每个结果
        save_path = os.path.join(save_dir, f"results_{i}.jpg")
        print(f"Saving result to: {save_path}")
        result.save(filename=save_path)  # 保存结果到指定目录
    i += 1