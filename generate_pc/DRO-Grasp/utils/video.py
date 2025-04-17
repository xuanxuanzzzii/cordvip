import cv2
import os

def video_to_frames(video_path, output_folder, frames_per_second):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频的帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 计算每秒钟截取的帧数间隔
    frame_interval = int(original_fps / frames_per_second)
    
    frame_count = 0
    saved_frame_count = 0
    while True:
        # 读取视频的一帧
        ret, frame = cap.read()
        
        # 如果读取成功，ret 为 True
        if not ret:
            break
        
        # 只保存每隔一定帧数的帧
        if frame_count % frame_interval == 0:
            # 设置输出文件的路径和名称
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            
            # 保存当前帧为图片
            cv2.imwrite(frame_filename, frame)
            
            print(f"Saved: {frame_filename}")
            saved_frame_count += 1
        
        frame_count += 1
    
    # 释放视频对象
    cap.release()
    print(f"Video processing complete. {saved_frame_count} frames saved to {output_folder}")

# 示例使用
video_path = "/home/alan/Downloads/test.mp4"  # 替换为你的视频文件路径
output_folder = "/home/alan/Downloads/video/"  # 输出图片文件夹
frames_per_second = 7  # 每秒保存的帧数
video_to_frames(video_path, output_folder, frames_per_second)
