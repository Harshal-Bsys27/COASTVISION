import cv2
import os
import argparse

def _check_zones(videos_dir: str) -> int:
    exts = (".mp4", ".avi", ".mov", ".mkv")
    print(f"[check-zones] videos_dir = {os.path.abspath(videos_dir)}")
    ok_all = True
    for zid in range(1, 7):
        found = None
        for ext in exts:
            p = os.path.join(videos_dir, f"zone{zid}{ext}")
            if os.path.exists(p):
                found = p
                break
        if not found:
            ok_all = False
            print(f"[check-zones] Zone {zid}: MISSING")
            continue
        cap = cv2.VideoCapture(found)
        opened = cap.isOpened()
        ret, frame = (False, None)
        if opened:
            ret, frame = cap.read()
        cap.release()
        if not opened:
            ok_all = False
            print(f"[check-zones] Zone {zid}: FOUND but FAILED to open -> {found}")
        elif not ret or frame is None:
            ok_all = False
            print(f"[check-zones] Zone {zid}: OPENED but FAILED decode -> {found}")
        else:
            h, w = frame.shape[:2]
            print(f"[check-zones] Zone {zid}: OK -> {found} ({w}x{h})")
    return 0 if ok_all else 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-zones", action="store_true")
    ap.add_argument("--videos-dir", default=os.path.join("frontend", "dashboard", "videos"))
    args = ap.parse_args()

    if args.check_zones:
        raise SystemExit(_check_zones(args.videos_dir))

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

if __name__ == "__main__":
    main()
