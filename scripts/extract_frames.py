import cv2
import os

# Set paths
video_dir = os.path.join('data', 'raw_videos')
frames_dir = os.path.join('data', 'frames')
os.makedirs(frames_dir, exist_ok=True)

# Parameters
frame_interval = 30  # Extract one frame every 30 frames (adjust as needed)

for video_file in os.listdir(video_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_count = 0
        video_name = os.path.splitext(video_file)[0]
        video_frames_dir = os.path.join(frames_dir, video_name)
        os.makedirs(video_frames_dir, exist_ok=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_filename = f"{video_name}_frame{count}.jpg"
                frame_path = os.path.join(video_frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            count += 1
        cap.release()
        print(f"Extracted {frame_count} frames from {video_file}")

print("Frame extraction complete.")
