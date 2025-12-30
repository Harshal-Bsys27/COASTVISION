# streamlit_dashboard.py
import streamlit as st
import cv2, os, time
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp

# -------------------- Config --------------------
st.set_page_config(layout="wide", page_title="6-Zone Beach Safety Dashboard")

# Your actual video files
VIDEO_PATHS = [
    "videos/zone1.mp4",
    "videos/zone2.mp4",
    "videos/zone3.mp4",
    "videos/zone4.mp4",
    "videos/zone5.mp4"
]

DISPLAY_WIDTH = 480
RELATIVE_WATER_POLYGON = np.array([
    [0.12, 0.55],
    [0.88, 0.55],
    [1.00, 1.00],
    [0.00, 1.00]
])
ALERT_CSV = "alerts/alerts_log.csv"
SNAPSHOT_DIR = "alerts/snapshots"

# -------------------- Prepare folders --------------------
os.makedirs("alerts", exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -------------------- Load models --------------------
def load_models():
    model = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    return model, pose
model, pose = load_models()



# -------------------- Utilities --------------------
def scale_polygon(poly, frame_w, frame_h):
    return np.array([[int(x * frame_w), int(y * frame_h)] for x, y in poly], np.int32)

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def annotate_frame(frame, results, pose, rel_poly, zone_id):
    """Run YOLO results + pose, return annotated frame and alert list for this frame."""
    h, w = frame.shape[:2]
    water_poly = scale_polygon(rel_poly, w, h)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [water_poly], (180, 200, 255))
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

    alerts = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls == 0:  # Person
                in_water = point_in_polygon((cx, cy), water_poly)
                status = "DANGER" if in_water else "SAFE"
                color = (0, 0, 255) if in_water else (0, 200, 0)

                # Bounding box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Person - {status}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Pose landmarks
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if res.pose_landmarks:
                        for lm in res.pose_landmarks.landmark:
                            lx = int(x1 + lm.x * (x2 - x1))
                            ly = int(y1 + lm.y * (y2 - y1))
                            if 0 <= lx < w and 0 <= ly < h:
                                cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

                # Alert
                if in_water:
                    alerts.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "zone": zone_id,
                        "status": status,
                        "bbox": [x1, y1, x2, y2]
                    })
            else:  # Other objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame, alerts

# -------------------- Session state --------------------
if "caps" not in st.session_state:
    st.session_state.caps = [cv2.VideoCapture(p) if os.path.exists(p) else None for p in VIDEO_PATHS]

if "alerts_df" not in st.session_state:
    st.session_state.alerts_df = pd.DataFrame(columns=["timestamp","zone","status","bbox","snapshot"])

if "running" not in st.session_state:
    st.session_state.running = False

# -------------------- UI --------------------
st.title("🏖️ 6-Zone Beach Safety Dashboard (Prototype)")
st.caption("Simulated multi-zone monitoring with YOLO + Pose Estimation.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
with col2:
    if st.button("⏸ Stop"):
        st.session_state.running = False

fps_limit = st.slider("Max FPS per Zone", 1, 15, 8)
save_snapshots = st.checkbox("Save one snapshot per zone (overwrite)", value=False)
save_csv = st.checkbox("Write alerts to CSV", value=True)

cols = st.columns(3)
placeholders = [c.empty() for c in cols for _ in range(2)]  # 2 rows × 3 cols

# -------------------- Main loop --------------------
if st.session_state.running:
    try:
        while st.session_state.running:
            start_time = time.time()
            frames = []

            for i, cap in enumerate(st.session_state.caps):
                if cap is None:
                    img = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(img, f"Missing: sample{i}.mp4", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    frames.append((img, []))
                    continue

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        frames.append((np.zeros((100, 100, 3), dtype=np.uint8), []))
                        continue

                # Resize
                h, w = frame.shape[:2]
                scale = DISPLAY_WIDTH / w
                frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)))

                # YOLO + annotation
                results = model(frame, verbose=False, stream=True)
                annotated, alerts = annotate_frame(frame.copy(), results, pose, RELATIVE_WATER_POLYGON, f"Zone-{i+1}")

                # Alerts handling
                for a in alerts:
                    a["snapshot"] = ""
                    if save_snapshots:
                        snap_path = os.path.join(SNAPSHOT_DIR, f"zone{i+1}.jpg")
                        cv2.imwrite(snap_path, annotated)
                        a["snapshot"] = snap_path
                    st.session_state.alerts_df = pd.concat([st.session_state.alerts_df, pd.DataFrame([a])], ignore_index=True)

                if save_csv and not st.session_state.alerts_df.empty:
                    st.session_state.alerts_df.drop_duplicates().to_csv(ALERT_CSV, index=False)

                frames.append((annotated, alerts))

            # Display grid
            for idx, (ph, (img, alerts)) in enumerate(zip(placeholders, frames)):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ph.image(img_rgb, caption=f"Zone {idx+1} {'🚨' if alerts else ''}", use_container_width=True)

            # Alerts log
            st.dataframe(st.session_state.alerts_df.tail(10), width=1200)

            # Throttle FPS
            elapsed = time.time() - start_time
            if elapsed < 1.0 / fps_limit:
                time.sleep((1.0 / fps_limit) - elapsed)

    except Exception as e:
        st.error(f"Runtime Error: {e}")
        st.session_state.running = False

else:
    st.info("Press ▶️ Start to begin monitoring.")
    st.write("Recent alerts:")
    st.dataframe(st.session_state.alerts_df.tail(10), width=1200)

