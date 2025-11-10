import cv2
import numpy as np

video_path = "1.mp4"

#读入视频
cap = cv2.VideoCapture(video_path)

#创建GMM
backSub = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=True)
# 形态学操作的卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

print("Processing video... Press 'q' to exit.")

# 处理视频帧
while True:
    ret,frame = cap.read()
    if not ret:
        print("End of video or cannot read the video.")
        break
    fgmask = backSub.apply(frame)
    # 形态学操作，去除噪声
    _,binary_mask = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    #显示原始帧
    cv2.imshow('Frame', frame)
    #显示GMM蒙版
    cv2.imshow('FG Mask', fgmask)
    #显示二值蒙版
    cv2.imshow('Clean Mask', clean_mask)
    res = cv2.bitwise_and(frame, frame, mask=clean_mask)
    #显示前景提取结果
    cv2.imshow('Foreground Extraction', res)

    # 按'q'键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
