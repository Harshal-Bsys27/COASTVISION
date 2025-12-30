"""PyQt dashboard showing six zone preview videos with YOLO person detection.

Videos should be named zone1.mp4 ... zone6.mp4 and placed under videos/.
Model weights default to models/yolov8n.pt (swap in best.pt later).
"""
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from PyQt5.QtCore import QDateTime, QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

# ---------------- PATHS & CONSTANTS ----------------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATHS = [BASE_DIR / "videos" / f"zone{i}.mp4" for i in range(1, 7)]
ZONE_NAMES = [f"Zone {i}" for i in range(1, 7)]
PREFERRED_MODEL = BASE_DIR / "models" / "best.pt"
FALLBACK_MODEL = BASE_DIR / "models" / "yolov8n.pt"
MODEL_PATH = PREFERRED_MODEL if PREFERRED_MODEL.exists() else FALLBACK_MODEL
ALERT_CSV = BASE_DIR / "alerts.csv"
DISPLAY_WIDTH = 480

# ---------------- MODEL ----------------
print(f"Loading YOLOv8 model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Model file missing. Place weights at frontend/dashboard/models/best.pt or yolov8n.pt, or update MODEL_PATH.")

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # prefer first GPU (RTX 3050 present on this laptop)
    device = "cuda"
    torch.backends.cudnn.benchmark = True  # optimize convs for constant input sizes
else:
    device = "cpu"
model = YOLO(str(MODEL_PATH)).to(device)
print(f"Model running on: {device.upper()}")

# ---------------- FRAME PROCESSING ----------------
immobile_tracker: dict[str, tuple[int, int, float]] = {}

def process_frame(frame, zone_name):
    alerts = []
    results = model(frame, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # only persons
                continue

            conf = float(box.conf[0]) * 100
            names = getattr(model, "names", None)
            if isinstance(names, dict):
                label = names.get(cls, "person")
            elif isinstance(names, (list, tuple)) and cls < len(names):
                label = names[cls]
            else:
                label = "person"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
            text = f"{label} {conf:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 200, 0), -1)
            cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            pid = f"{zone_name}_{cx}_{cy}"
            now = time.time()
            if pid not in immobile_tracker:
                immobile_tracker[pid] = (cx, cy, now)
            else:
                old_x, old_y, old_t = immobile_tracker[pid]
                dist = ((cx - old_x) ** 2 + (cy - old_y) ** 2) ** 0.5
                if dist < 10 and (now - old_t) > 5:
                    alerts.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "zone": zone_name,
                        "status": "⚠️ Person immobile - Possible drowning",
                    })
                    immobile_tracker[pid] = (cx, cy, now + 3)
                else:
                    immobile_tracker[pid] = (cx, cy, now)

    return frame, alerts

# ---------------- ZONE WIDGET ----------------
class ZoneWidget(QFrame):
    def __init__(self, zone_index, parent=None):
        super().__init__(parent)
        self.zone_index = zone_index
        self.zone_name = ZONE_NAMES[zone_index]

        self.setStyleSheet(
            """
            QFrame {
                background-color: #1B1B2F;
                border-radius: 12px;
                border: 2px solid #2E2E3E;
            }
            """
        )

        layout = QVBoxLayout()

        self.title = QLabel(self.zone_name)
        self.title.setFont(QFont("Arial", 14, QFont.Bold))
        self.title.setStyleSheet("color: #00E5FF; padding: 6px;")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        self.video_label = ClickableLabel(self.zone_index, parent)
        self.video_label.setFixedSize(DISPLAY_WIDTH, 270)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 6px;")
        layout.addWidget(self.video_label)

        self.status_label = QLabel("✅ Monitoring")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("color: #00FF88; padding: 4px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def update_frame(self, qimg):
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_status(self, alerts):
        if alerts:
            self.status_label.setText("🚨 " + alerts[-1]["status"])
            self.status_label.setStyleSheet(
                """
                color: #FF5555;
                font-weight: bold;
                padding: 4px;
                border: 2px solid #FF5555;
                border-radius: 6px;
                """
            )
        else:
            self.status_label.setText("✅ Monitoring")
            self.status_label.setStyleSheet(
                """
                color: #00FF88;
                padding: 4px;
                border: 2px solid #00FF88;
                border-radius: 6px;
                """
            )

# ---------------- POPUP VIEW ----------------
class ZoneViewer(QDialog):
    def __init__(self, zone_index, cap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{ZONE_NAMES[zone_index]} - Live View")
        self.resize(1280, 720)
        self.cap = cap
        self.zone_name = ZONE_NAMES[zone_index]
        self.is_paused = False

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.video_label)

        # Controls: pause/play + zoom helpers
        controls = QHBoxLayout()
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.zoom_in_btn = QPushButton("＋ Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn = QPushButton("－ Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_btn = QPushButton("⟳ Reset")
        self.reset_btn.clicked.connect(self.reset_zoom)
        for btn in (self.pause_btn, self.zoom_in_btn, self.zoom_out_btn, self.reset_btn):
            btn.setStyleSheet(
                """
                QPushButton { background-color: #162447; color: #E0E0E0; padding: 8px 12px; border-radius: 8px; }
                QPushButton:hover { background-color: #1f2d4f; }
                """
            )
            controls.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addLayout(controls)
        self.setLayout(layout)

        self.scale_factor = 0.9  # start slightly zoomed out
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        if self.is_paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame, _ = process_frame(frame, self.zone_name)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * self.scale_factor),
            int(pixmap.height() * self.scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.setText("▶ Play" if self.is_paused else "⏸ Pause")

    def zoom_in(self):
        self._apply_zoom(1.08)

    def zoom_out(self):
        self._apply_zoom(1 / 1.08)

    def _apply_zoom(self, factor: float):
        self.scale_factor = max(0.35, min(4.0, self.scale_factor * factor))
        self.update_frame()

    def wheelEvent(self, event):
        delta = event.pixelDelta().y() or event.angleDelta().y()
        if delta == 0:
            return
        step = max(-5, min(5, delta / 120))  # normalize scroll units (touchpad-friendly)
        factor = 1.0 + (0.08 * step)
        if factor <= 0:
            return
        self._apply_zoom(factor)

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.update_frame()

# ---------------- ALERT CARD ----------------
class AlertCard(QFrame):
    def __init__(self, alert):
        super().__init__()
        self.setStyleSheet(
            """
            QFrame {
                background-color: #2A2A3E;
                border: 1px solid #FF5555;
                border-radius: 8px;
                margin: 4px;
            }
            """
        )
        layout = QVBoxLayout()
        label = QLabel(f"{alert['timestamp']} | {alert['zone']} | {alert['status']}")
        label.setStyleSheet("color: #FF8888; font-size: 13px;")
        layout.addWidget(label)
        self.setLayout(layout)

# ---------------- DASHBOARD ----------------
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌊 CoastVision Dashboard")
        self.resize(1850, 1000)
        self.setStyleSheet("background-color: #101820; color: #E0E0E0;")

        self.header = QLabel("🌊 CoastVision - Beach Safety Monitoring")
        self.header.setFont(QFont("Arial", 24, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("background-color: #162447; color: #FFFFFF; padding: 16px; border-radius: 8px;")

        self.clock = QLabel()
        self.clock.setStyleSheet("color: #00FFAA; font-size: 16px; padding: 6px;")
        self.update_clock()

        gpu_status = QLabel(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU Mode'}")
        gpu_status.setStyleSheet("color: #00FFAA; font-size: 14px; padding: 6px;")

        header_bar = QHBoxLayout()
        header_bar.addWidget(self.header, stretch=4)
        header_bar.addWidget(self.clock, stretch=1)
        header_bar.addWidget(gpu_status, stretch=1)

        self.grid = QGridLayout()
        self.zones = []
        for i in range(6):
            zone = ZoneWidget(i, self)
            self.zones.append(zone)
            self.grid.addWidget(zone, i // 3, i % 3)

        self.alert_area = QScrollArea()
        self.alert_area.setWidgetResizable(True)
        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout()
        self.alert_container.setLayout(self.alert_layout)
        self.alert_area.setWidget(self.alert_container)
        self.alert_area.setFixedWidth(450)
        self.alert_area.setStyleSheet("background-color: #1F1F2E; border-radius: 10px;")

        alerts_box = QVBoxLayout()
        alerts_title = QLabel("🚨 Global Alerts Log")
        alerts_title.setFont(QFont("Arial", 16, QFont.Bold))
        alerts_title.setStyleSheet("color: #FF5555; padding: 6px;")
        alerts_box.addWidget(alerts_title)
        alerts_box.addWidget(self.alert_area)

        main_layout = QVBoxLayout()
        main_layout.addLayout(header_bar)
        content = QHBoxLayout()
        content.addLayout(self.grid)
        content.addLayout(alerts_box)
        main_layout.addLayout(content)
        self.setLayout(main_layout)

        self.caps = [self._open_capture(p) for p in VIDEO_PATHS]
        self.alerts_df = pd.DataFrame(columns=["timestamp", "zone", "status"])

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(100)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def _open_capture(self, path: Path):
        if not path.exists():
            print(f"Missing video for {path.name}; place file at {path}")
            return None
        return cv2.VideoCapture(str(path))

    def update_clock(self):
        now = QDateTime.currentDateTime()
        self.clock.setText(now.toString("ddd MMM dd yyyy | hh:mm:ss"))

    def update_frames(self):
        for i, cap in enumerate(self.caps):
            if cap is None:
                continue
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h * (DISPLAY_WIDTH / w))))
            processed, alerts = process_frame(frame, ZONE_NAMES[i])

            if alerts:
                self.alerts_df = pd.concat([self.alerts_df, pd.DataFrame(alerts)], ignore_index=True)
                self.alerts_df.drop_duplicates().to_csv(ALERT_CSV, index=False)
                for a in alerts:
                    card = AlertCard(a)
                    self.alert_layout.insertWidget(0, card)

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            self.zones[i].update_frame(qimg)
            self.zones[i].update_status(alerts)

    def open_zone_view(self, zone_index):
        if self.caps[zone_index] is None:
            return
        viewer = ZoneViewer(zone_index, self.caps[zone_index], self)
        viewer.exec_()

# ---------------- CLICKABLE LABEL ----------------
class ClickableLabel(QLabel):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.parent = parent

    def mousePressEvent(self, event):
        self.parent.open_zone_view(self.index)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec_())
