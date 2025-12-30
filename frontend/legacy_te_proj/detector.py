# detector.py
import time
import cv2
import torch
from ultralytics import YOLO
from config import ZONE_NAMES

# Load YOLOv8 model once
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt").to(device)
print(f"Model running on: {device.upper()}")

# simple immobility tracker for 'possible drowning' alerts
immobile_tracker = {}

def process_frame(frame, zone_name, draw=True, detect_only_persons=True):
    """
    Runs YOLO on a frame and returns (annotated_frame, alerts_list, detections_list)

    - alerts_list: list of dicts {"timestamp","zone","status"}
    - detections_list: list of (x1,y1,x2,y2,label,conf)
    """
    alerts = []
    detections = []

    # run model: ultralytics returns Results list
    results = model(frame, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls] if hasattr(model, "names") else str(cls)

            # If user asked to only detect persons (class 0)
            if detect_only_persons and cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            detections.append((x1, y1, x2, y2, label, conf))

            if draw:
                # styled bounding box + label
                color = (0, 200, 0)  # green
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                text = f"{label} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                lb_x1 = x1
                lb_y1 = max(0, y1 - th - 12)
                lb_x2 = x1 + tw + 12
                lb_y2 = y1
                cv2.rectangle(frame, (lb_x1, lb_y1), (lb_x2, lb_y2), color, -1)
                cv2.putText(frame, text, (x1 + 6, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # center dot
                cv2.circle(frame, (cx, cy), 4, color, -1)

            # simple immobility/drowning heuristic: same center for > 5s
            pid = f"{zone_name}_{cx}_{cy}"
            now = time.time()
            if pid not in immobile_tracker:
                immobile_tracker[pid] = (cx, cy, now)
            else:
                old_x, old_y, old_t = immobile_tracker[pid]
                dist = ((cx - old_x)**2 + (cy - old_y)**2)**0.5
                if dist < 10 and (now - old_t) > 5:
                    # create one alert (could be refined with staging & de-dup)
                    alerts.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "zone": zone_name,
                        "status": "⚠️ Possible Drowning"
                    })
                    # reset timer to avoid spamming same alert every frame
                    immobile_tracker[pid] = (cx, cy, now + 3)
                else:
                    immobile_tracker[pid] = (cx, cy, now)

    return frame, alerts, detections
