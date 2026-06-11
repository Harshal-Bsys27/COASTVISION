# main_dashboard.py
import sys, os, cv2, time, pandas as pd, torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QDialog, QFrame, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt, QDateTime
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATHS = [
    "videos/zone1.mp4", "videos/zone2.mp4", "videos/zone3.mp4",
    "videos/zone4.mp4", "videos/zone5.mp4", "videos/zone6.mp4"
]
ZONE_NAMES = [f"Zone {i+1}" for i in range(6)]
ALERT_CSV = "alerts.csv"
DISPLAY_WIDTH = 480

# ================= YOLO MODEL =================
print("Loading YOLOv8 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)
print(f"Model running on: {device.upper()}")

# ================= HELPER: PROCESS FRAME =================
immobile_tracker = {}

def process_frame(frame, zone_name):
    alerts = []
    results = model(frame, verbose=False, device=device)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            label = model.names[cls] if cls in model.names else "Unknown"

            # Only check drowning for person
            if cls != 0:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)

            # Add label above box
            text = f"{label} {conf:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 200, 0), -1)
            cv2.putText(frame, text, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Immobility tracking for drowning
            pid = f"{zone_name}_{cx}_{cy}"
            now = time.time()
            if pid not in immobile_tracker:
                immobile_tracker[pid] = (cx, cy, now)
            else:
                old_x, old_y, old_t = immobile_tracker[pid]
                dist = ((cx - old_x) ** 2 + (cy - old_y) ** 2) ** 0.5
                if dist < 10:
                    if now - old_t > 5:
                        status = f"⚠️ {label} - Possible Drowning"
                        alerts.append({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "zone": zone_name,
                            "status": status
                        })
                else:
                    immobile_tracker[pid] = (cx, cy, now)

    return frame, alerts

# ================= ZONE WIDGET =================
class ZoneWidget(QFrame):
    def __init__(self, zone_index, parent=None):
        super().__init__(parent)
        self.zone_index = zone_index
        self.zone_name = ZONE_NAMES[zone_index]

        self.setStyleSheet("""
            QFrame {
                background-color: #1B1B2F;
                border-radius: 12px;
                border: 2px solid #2E2E3E;
            }
        """)

        layout = QVBoxLayout()

        # Zone header
        self.title = QLabel(self.zone_name)
        self.title.setFont(QFont("Arial", 14, QFont.Bold))
        self.title.setStyleSheet("color: #00E5FF; padding: 6px;")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        # Video display
        self.video_label = ClickableLabel(self.zone_index, parent)
        self.video_label.setFixedSize(DISPLAY_WIDTH, 270)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 6px;")
        layout.addWidget(self.video_label)

        # Status footer
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
            self.status_label.setStyleSheet("""
                color: #FF5555; 
                font-weight: bold; 
                padding: 4px; 
                border: 2px solid #FF5555; 
                border-radius: 6px;
            """)
        else:
            self.status_label.setText("✅ Monitoring")
            self.status_label.setStyleSheet("""
                color: #00FF88; 
                padding: 4px;
                border: 2px solid #00FF88; 
                border-radius: 6px;
            """)

# ================= POPUP ZONE VIEW =================
# ================= POPUP ZONE VIEW =================
class ZoneViewer(QDialog):
    def __init__(self, zone_index, cap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{ZONE_NAMES[zone_index]} - Live View")
        self.resize(960, 540)
        self.cap = cap
        self.zone_name = ZONE_NAMES[zone_index]

        # Video label with scroll + pan
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.video_label)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        # Zooming
        self.scale_factor = 1.0

        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame, _ = process_frame(frame, self.zone_name)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Apply zoom scaling
        scaled_pixmap = pixmap.scaled(
            int(pixmap.width() * self.scale_factor),
            int(pixmap.height() * self.scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    # Handle scroll for zoom
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1  # zoom in
        else:
            self.scale_factor /= 1.1  # zoom out
        self.scale_factor = max(0.5, min(self.scale_factor, 3.0))  # clamp zoom
        self.update_frame()


# ================= ALERT CARD =================
class AlertCard(QFrame):
    def __init__(self, alert):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #2A2A3E;
                border: 1px solid #FF5555;
                border-radius: 8px;
                margin: 4px;
            }
        """)
        layout = QVBoxLayout()
        label = QLabel(f"{alert['timestamp']} | {alert['zone']} | {alert['status']}")
        label.setStyleSheet("color: #FF8888; font-size: 13px;")
        layout.addWidget(label)
        self.setLayout(layout)

# ================= DASHBOARD =================
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌊 CoastVision Dashboard")
        self.resize(1850, 1000)
        self.setStyleSheet("background-color: #101820; color: #E0E0E0;")

        # Header with clock
        self.header = QLabel("🌊 CoastVision - Beach Safety Monitoring")
        self.header.setFont(QFont("Arial", 24, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("background-color: #162447; color: #FFFFFF; padding: 16px; border-radius: 8px;")

        self.clock = QLabel()
        self.clock.setStyleSheet("color: #00FFAA; font-size: 16px; padding: 6px;")
        self.update_clock()

        gpu_status = QLabel(f"GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU Mode'}")
        gpu_status.setStyleSheet("color: #00FFAA; font-size: 14px; padding: 6px;")

        header_bar = QHBoxLayout()
        header_bar.addWidget(self.header, stretch=4)
        header_bar.addWidget(self.clock, stretch=1)
        header_bar.addWidget(gpu_status, stretch=1)

        # Zones grid
        self.grid = QGridLayout()
        self.zones = []
        for i in range(6):
            zone = ZoneWidget(i, self)
            self.zones.append(zone)
            self.grid.addWidget(zone, i // 3, i % 3)

        # Alerts panel
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

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(header_bar)
        content = QHBoxLayout()
        content.addLayout(self.grid)
        content.addLayout(alerts_box)
        main_layout.addLayout(content)
        self.setLayout(main_layout)

        # Video capture
        self.caps = [cv2.VideoCapture(p) if os.path.exists(p) else None for p in VIDEO_PATHS]
        self.alerts_df = pd.DataFrame(columns=["timestamp", "zone", "status"])

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(100)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

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

# ================= CLICKABLE LABEL =================
class ClickableLabel(QLabel):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.parent = parent

    def mousePressEvent(self, event):
        self.parent.open_zone_view(self.index)

# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec_())
