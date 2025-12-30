import os
import cv2
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QFrame, QPushButton, QDialog, QListWidget,
    QStackedWidget, QListWidgetItem, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsDropShadowEffect,
    QTextEdit
)
from PyQt5.QtGui import QFont, QColor, QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QTimer, QDateTime, QSize

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from detector import process_frame
from analytics import log_alerts, update_heatmap, plot_heatmap, plot_zone_bars
from config import VIDEO_PATHS, ZONE_NAMES, DISPLAY_WIDTH, get_device_string

# ---------- Theme colors (Ocean Futuristic) ----------
NEON_BG = "#0A192F"        # Deep navy background
CARD_BG = "#4A0A61"        # Panel background
NEON_CYAN = "#00E0FF"      # Aqua-cyan (main accent)
NEON_PURPLE = "#7B2CBF"    # Futuristic purple
NEON_GREEN = "#06D6A0"     # Sea green (safe)
NEON_RED = "#FF4C60"       # Coral red (alerts)
ACCENT = "#FF8C42"         # Orange highlight
TEXT = "#E6F1FF"           # Ice white text

# ---------- Zoom popup with mouse + keyboard control ----------
# ---------- Zoom popup with mouse + keyboard control ----------
class VideoPopup(QDialog):
    def __init__(self, zone_name, video_path):
        super().__init__()
        self.zone_name = zone_name
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.is_paused = False

        self.setWindowTitle(f"🔎 {zone_name} — Zoom View")
        self.resize(1200, 780)
        self.setStyleSheet(f"background: {NEON_BG}; color: {TEXT};")

        layout = QVBoxLayout(self)

        # ---------- GraphicsView for zoom/pan ----------
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # mouse drag pan
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # zoom relative to cursor
        layout.addWidget(self.view, stretch=1)

        # ---------- Controls ----------
        ctl = QHBoxLayout()
        btn_style = f"""
            QPushButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {NEON_CYAN}, stop:1 {NEON_PURPLE}
                );
                color: black; font-weight: bold;
                padding:8px; border-radius:8px;
            }}
            QPushButton:hover {{ background: {NEON_PURPLE}; color: white; }}
        """

        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.setStyleSheet(btn_style)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)

        self.reset_zoom = QPushButton("🔄 Reset View")
        self.reset_zoom.setStyleSheet(btn_style)
        self.reset_zoom.clicked.connect(self.reset_view)

        ctl.addWidget(self.play_pause_btn)
        ctl.addWidget(self.reset_zoom)
        layout.addLayout(ctl)

        # ---------- Pixmap state ----------
        self.pix_item = None

        # ---------- Timer for video playback ----------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS for smooth playback

        # ---------- Zoom state ----------
        self.scale_factor = 1.0
        self.max_zoom = 5.0
        self.min_zoom = 0.3

    # ---------- Frame updates ----------
    def update_frame(self):
        if self.is_paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        processed, _, _ = process_frame(frame, self.zone_name)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        if self.pix_item is None:
            self.pix_item = self.scene.addPixmap(pix)
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
        else:
            self.pix_item.setPixmap(pix)

    # ---------- Play / Pause ----------
    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        self.play_pause_btn.setText("▶ Resume" if self.is_paused else "⏸ Pause")

    # ---------- Reset ----------
    def reset_view(self):
        if self.pix_item:
            self.view.resetTransform()
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
            self.scale_factor = 1.0

    # ---------- Mouse wheel zoom ----------
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0 and self.scale_factor < self.max_zoom:
            zoom_factor = zoom_in_factor
            self.scale_factor *= zoom_in_factor
        elif event.angleDelta().y() < 0 and self.scale_factor > self.min_zoom:
            zoom_factor = zoom_out_factor
            self.scale_factor *= zoom_out_factor
        else:
            return

        self.view.scale(zoom_factor, zoom_factor)

    # ---------- Keyboard shortcuts ----------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            if self.scale_factor < self.max_zoom:
                self.view.scale(1.25, 1.25)
                self.scale_factor *= 1.25
        elif event.key() == Qt.Key_Minus:
            if self.scale_factor > self.min_zoom:
                self.view.scale(0.8, 0.8)
                self.scale_factor *= 0.8
        elif event.key() == Qt.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)


# ---------- Zone Card ----------
class ZoneCard(QFrame):
    def __init__(self, index, video_path):
        super().__init__()
        self.index = index
        self.video_path = video_path
        self.zone_name = ZONE_NAMES[index]

        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD_BG};
                border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.05);
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Enhanced title
        self.title = QLabel(self.zone_name)
        self.title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.title.setStyleSheet(f"""
            color: {NEON_CYAN}; 
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
        """)
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        # Larger preview with fixed aspect ratio
        self.preview = QLabel()
        self.preview.setFixedSize(480, 270)  # 16:9 aspect ratio, larger size
        self.preview.setStyleSheet("""
            background: #080A10; 
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0px;
        """)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setScaledContents(False)  # Don't scale content to prevent distortion
        self.preview.mousePressEvent = self.open_popup
        layout.addWidget(self.preview)

        # Enhanced status indicator
        self.status = QLabel("● Monitoring")
        self.status.setFont(QFont("Segoe UI", 12))
        self.status.setStyleSheet(f"""
            color: {NEON_GREEN}; 
            padding: 8px 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        """)
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

    def update_preview(self, qimg, status_text=None):
        self.preview.setPixmap(QPixmap.fromImage(qimg))
        if status_text:
            self.status.setText(status_text)

    def enterEvent(self, event):  # Fixed indentation
        """Hover in effect."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD_BG};
                border: 2px solid {NEON_CYAN};
                border-radius: 12px;
                padding: 8px;
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):  # Fixed indentation
        """Hover out effect."""
        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD_BG};
                border: 1px solid rgba(255,255,255,0.03);
                border-radius: 12px;
                padding: 8px;
            }}
        """)
        super().leaveEvent(event)

    def open_popup(self, event):
        if not os.path.exists(self.video_path):
            return
        popup = VideoPopup(self.zone_name, self.video_path)
        popup.exec_()

# ---------- Sidebar (nav) ----------
class SideNav(QListWidget):
    def __init__(self, items):
        super().__init__()
        self.setFixedWidth(220)
        self.setSpacing(10)
        self.setStyleSheet(f"""
            QListWidget {{
                background: rgba(10,12,20,0.8);
                color: {TEXT};
                border: none;
            }}
            QListWidget::item {{
                padding: 12px;
                margin: 6px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 {NEON_CYAN}, stop:1 {NEON_PURPLE});
                color: black;
                font-weight: bold;
            }}
        """)
        for i, (k, label) in enumerate(items):
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, k)
            it.setSizeHint(QSize(200, 44))
            self.addItem(it)

# ---------- Main Dashboard UI (stacked pages) ----------
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoastVision Dashboard — Futuristic")
        self.resize(1600, 960)
        self.setStyleSheet(f"background: {NEON_BG}; color: {TEXT};")

        # ---------- Header ----------
        top = QHBoxLayout()

        header_label = QLabel("🌊 CoastVision Dashboard")
        header_label.setFont(QFont("Segoe UI", 23, QFont.Bold))
        header_label.setStyleSheet(f"color: {NEON_CYAN};")

        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(30)
        glow.setXOffset(0)
        glow.setYOffset(0)
        glow.setColor(QColor(0, 224, 255))
        header_label.setGraphicsEffect(glow)

        top.addWidget(header_label, alignment=Qt.AlignLeft)

        self.clock = QLabel()
        self.clock.setFont(QFont("Consolas", 14))
        self.clock.setStyleSheet(f"color: {NEON_PURPLE};")
        top.addWidget(self.clock, alignment=Qt.AlignRight)

        gpu = QLabel(get_device_string())
        gpu.setFont(QFont("Segoe UI", 12))
        gpu.setStyleSheet(f"color: {NEON_GREEN}; padding-left:12px;")
        top.addWidget(gpu, alignment=Qt.AlignRight)

        # ---------- Sidebar ----------
        nav_items = [
            ("monitor", "Live Monitoring"),
            ("heatmap", "People Heatmap"),
            ("alerts", "Alerts & Logs"),
            ("analytics", "Analytics"),
            ("zone_status", "Zone Drowning Status"),  # NEW SIDEBAR ITEM
            ("settings", "Settings")
        ]
        self.sidenav = SideNav(nav_items)
        self.sidenav.currentRowChanged.connect(self.on_nav_changed)

        # ---------- Stacked Pages ----------
        self.stack = QStackedWidget()
        self.page_monitor = QWidget()
        self.page_heatmap = QWidget()
        self.page_alerts = QWidget()
        self.page_analytics = QWidget()
        self.page_zone_status = QWidget()  # NEW PAGE
        self.page_settings = QWidget()
        self.stack.addWidget(self.page_monitor)
        self.stack.addWidget(self.page_heatmap)
        self.stack.addWidget(self.page_alerts)
        self.stack.addWidget(self.page_analytics)
        self.stack.addWidget(self.page_zone_status)  # ADD TO STACK
        self.stack.addWidget(self.page_settings)

        # Build pages
        self._build_monitor_page()
        self._build_heatmap_page()
        self._build_alerts_page()
        self._build_analytics_page()
        self._build_zone_status_page()  # NEW
        self._build_settings_page()  # <-- this must exist!

        # By default show monitoring
        self.sidenav.setCurrentRow(0)
        self.stack.setCurrentIndex(0)

        # ---------- Main layout ----------
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Sidebar
        self.sidenav.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {CARD_BG};
                border: none;
                color: {TEXT};
                font-size: 14px;
            }}
            QListWidget::item:selected {{
                background-color: {NEON_CYAN};
                color: black;
                font-weight: bold;
            }}
            """
        )
        main_layout.addWidget(self.sidenav, stretch=0)

        # Neon divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet(f"color: {NEON_CYAN}; background: {NEON_CYAN};")
        divider.setFixedWidth(2)
        main_layout.addWidget(divider)

        # --- Main content column (scrollable) ---
        # Create a QWidget for the main content and put it in a QScrollArea
        self.main_content_widget = QWidget()
        self.main_content_layout = QVBoxLayout(self.main_content_widget)
        self.main_content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_content_layout.setSpacing(0)
        self.main_content_layout.addLayout(top)
        self.main_content_layout.addWidget(self.stack)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.main_content_widget)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        main_layout.addWidget(self.scroll_area, stretch=1)

        self.setLayout(main_layout)

        # ---------- Video capture + timer ----------
        self.caps = [cv2.VideoCapture(p) if os.path.exists(p) else None for p in VIDEO_PATHS]
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30 FPS for smooth video playback
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()
        self.is_paused = False
        self.chart_mode = "bar"  # default chart mode for consistency with toggle button
        self.heatmap_canvas = None  # ensure attribute exists

        # Initialize zone states and confidences
        self.zone_states = ["Unknown"] * len(VIDEO_PATHS)
        self.zone_confidences = [0.0] * len(VIDEO_PATHS)

    # ---------- page builders ----------
    def _build_monitor_page(self):
        layout = QVBoxLayout(self.page_monitor)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        # --- Control bar ---
        ctl = QHBoxLayout()
        btn_style = f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {NEON_CYAN}, stop:1 {NEON_PURPLE}); color:black; padding:8px; border-radius:8px;"
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setStyleSheet(btn_style)
        self.pause_btn.clicked.connect(self.pause)
        self.resume_btn = QPushButton("▶ Resume")
        self.resume_btn.setStyleSheet(btn_style)
        self.resume_btn.clicked.connect(self.resume)
        ctl.addWidget(self.pause_btn)
        ctl.addWidget(self.resume_btn)
        layout.addLayout(ctl)

        # --- 3x3 Grid for 6 zones (fixed, not scrollable) ---
        grid_frame = QFrame()
        grid_frame.setStyleSheet(f"background: rgba(10,20,40,0.25); border-radius: 18px;")
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setSpacing(24)
        grid_layout.setContentsMargins(24, 24, 24, 24)
        self.zone_cards = []
        idx = 0
        for row in range(3):
            for col in range(3):
                if idx < len(VIDEO_PATHS):
                    card = ZoneCard(idx, VIDEO_PATHS[idx])
                    self.zone_cards.append(card)
                    grid_layout.addWidget(card, row, col)
                    idx += 1
                else:
                    # Add empty QFrame for visual balance
                    empty = QFrame()
                    empty.setStyleSheet("background: transparent;")
                    grid_layout.addWidget(empty, row, col)
        layout.addWidget(grid_frame)

        # --- State summary section (NEW) ---
        # Add state summary section
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background: {CARD_BG};
                border-radius: 15px;
                margin-top: 12px;
            }}
        """)
        summary_layout = QVBoxLayout(summary_frame)
        
        # Title for summary section
        summary_title = QLabel("Zone Status Summary")
        summary_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        summary_title.setStyleSheet(f"color: {NEON_CYAN}; margin-bottom: 8px;")
        summary_layout.addWidget(summary_title)

        # Scrollable area for status updates
        self.state_summary_scroll = QScrollArea()
        self.state_summary_scroll.setWidgetResizable(True)
        self.state_summary_scroll.setFrameShape(QFrame.NoFrame)
        
        state_summary_widget = QWidget()
        self.state_summary_vbox = QVBoxLayout(state_summary_widget)
        self.state_summary_vbox.setSpacing(8)
        self.state_summary_vbox.setContentsMargins(12, 12, 12, 12)
        
        self.state_summary_scroll.setWidget(state_summary_widget)
        summary_layout.addWidget(self.state_summary_scroll)
        
        layout.addWidget(summary_frame)
        layout.addStretch()

    def _build_heatmap_page(self):
        """Build the analytics page with chart and toggle button."""
        layout = QVBoxLayout(self.page_heatmap)
        layout.setSpacing(0)
        layout.setContentsMargins(10, 10, 10, 10)

        # Container for title and button
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(10)
        header.setFixedHeight(100)
        header.setStyleSheet("background: rgba(13,13,26,0.7);")

        # Chart title
        title = QLabel("Analytics Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        header_layout.addWidget(title)

        # Toggle button with improved style
        self.toggle_btn = QPushButton("Switch to Heatmap")
        self.toggle_btn.setFixedWidth(200)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {NEON_CYAN}, stop:1 {NEON_PURPLE});
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {NEON_PURPLE}, stop:1 {NEON_CYAN});
            }}
        """)
        self.toggle_btn.clicked.connect(self._toggle_chart_mode)
        header_layout.addWidget(self.toggle_btn, alignment=Qt.AlignCenter)
        layout.addWidget(header)

        # Chart container frame (centered, fixed height)
        self.chart_container = QFrame()
        self.chart_container.setStyleSheet(f"background: {CARD_BG}; border-radius: 10px;")
        self.chart_container.setMinimumHeight(500)
        self.chart_container.setMaximumHeight(700)
        self.chart_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_layout = QHBoxLayout(self.chart_container)  # Use QHBoxLayout for centering
        self.chart_layout.setContentsMargins(40, 40, 40, 40)
        self.chart_layout.setSpacing(0)
        layout.addWidget(self.chart_container, stretch=1, alignment=Qt.AlignCenter)

        # Chart widgets (only one visible at a time, always present)
        self.chart_mode = "bar"
        self.chart_widget = QWidget()
        self.chart_widget.setLayout(QVBoxLayout())
        self.chart_widget.layout().setAlignment(Qt.AlignCenter)
        self.heatmap_widget = QWidget()
        self.heatmap_widget.setLayout(QVBoxLayout())
        self.heatmap_widget.layout().setAlignment(Qt.AlignCenter)
        self.chart_layout.addWidget(self.chart_widget, alignment=Qt.AlignCenter)
        self.chart_layout.addWidget(self.heatmap_widget, alignment=Qt.AlignCenter)
        self.heatmap_widget.hide()
        self.chart_widget.show()
        self._refresh_chart()
        self._chart_switching = False

    def _build_alerts_page(self):
        layout = QVBoxLayout(self.page_alerts)
        header = QLabel("🚨 Alerts & Logs")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet(f"color:{NEON_PURPLE};")
        layout.addWidget(header)

        # Alerts list
        self.alert_list = QListWidget()
        self.alert_list.setStyleSheet("background: rgba(10,12,20,0.6); color: #ffcccc;")
        layout.addWidget(self.alert_list)

        # Export button
        export_btn = QPushButton("📥 Export Alerts (CSV)")
        export_btn.setStyleSheet(f"background:{NEON_CYAN}; color:black; font-weight:bold; padding:8px; border-radius:6px;")
        export_btn.clicked.connect(self.export_alerts)
        layout.addWidget(export_btn)

        # Preload existing alerts
        if os.path.exists("alerts.csv"):
            try:
                df = pd.read_csv("alerts.csv")
                for _, r in df.iterrows():
                    self.alert_list.addItem(f"{r['timestamp']} | {r['zone']} | {r['status']}")
            except Exception:
                pass

    def _build_analytics_page(self):
        layout = QVBoxLayout(self.page_analytics)
        header = QLabel("📈 Analytics & Reports")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet(f"color:{ACCENT};")
        layout.addWidget(header)

        # ----------- Analytics Report Section -----------
        self.analytics_report = QTextEdit()
        self.analytics_report.setReadOnly(True)
        self.analytics_report.setFont(QFont("Consolas", 12))
        self.analytics_report.setStyleSheet(f"""
            background: {CARD_BG};
            color: {TEXT};
            border-radius: 10px;
            padding: 18px;
            margin-top: 12px;
        """)
        layout.addWidget(self.analytics_report)

        # Export PDF button
        export_pdf_btn = QPushButton("📄 Export Analytics (PDF)")
        export_pdf_btn.setStyleSheet(f"background:{NEON_PURPLE}; color:white; font-weight:bold; padding:8px; border-radius:6px;")
        export_pdf_btn.clicked.connect(self.export_analytics_pdf)
        layout.addWidget(export_pdf_btn)

    def _build_zone_status_page(self):
        layout = QVBoxLayout(self.page_zone_status)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(18)

        title = QLabel("Zone-wise Drowning Detection Status")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet(f"color: {NEON_CYAN}; margin-bottom: 18px;")
        layout.addWidget(title, alignment=Qt.AlignTop | Qt.AlignHCenter)

        self.zone_status_vbox = QVBoxLayout()
        self.zone_status_vbox.setSpacing(12)
        layout.addLayout(self.zone_status_vbox)
        layout.addStretch()

    def _build_settings_page(self):
        layout = QVBoxLayout(self.page_settings)
        header = QLabel("⚙️ Settings")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet(f"color:{NEON_CYAN};")
        layout.addWidget(header)
        layout.addWidget(QLabel("Model: yolov8n.pt (change in detector.py)"))
        layout.addStretch()

    def export_alerts(self):
        """Export alerts to CSV file."""
        try:
            rows = [self.alert_list.item(i).text() for i in range(self.alert_list.count())]
            if not rows:
                return
            df = pd.DataFrame([r.split(" | ") for r in rows], columns=["timestamp", "zone", "status"])
            df.to_csv("exported_alerts.csv", index=False)
            print("✅ Alerts exported to exported_alerts.csv")
        except Exception as e:
            print("❌ Error exporting alerts:", e)

    def export_analytics_pdf(self):
        """Export analytics report as PDF."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            pdf_file = "analytics_report.pdf"
            c = canvas.Canvas(pdf_file, pagesize=A4)
            width, height = A4

            # Title
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, height - 50, "📊 CoastVision Analytics Report")

            # Add zone counts
            counts = [str(x) for x in update_heatmap.__self__.get_zone_counts()] if hasattr(update_heatmap, "__self__") else []
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 100, f"Zone counts: {', '.join(counts) if counts else 'No data'}")

            # Footer
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(50, 30, "Generated by CoastVision Dashboard")

            c.save()
            print("✅ Analytics exported to analytics_report.pdf")
        except Exception as e:
            print("❌ Error exporting PDF:", e)


    # ---------- runtime ----------
    def on_nav_changed(self, idx):
        self.stack.setCurrentIndex(idx)

    def _update_clock(self):
        self.clock.setText(QDateTime.currentDateTime().toString("ddd MMM dd yyyy | hh:mm:ss"))

    def _tick(self):
        """Main loop for grabbing frames and updating UI."""
        if self.is_paused:
            return

        detections_per_zone = {}
        zone_states = []
        zone_confidences = []
        for i, cap in enumerate(self.caps):
            if cap is None:
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Get original dimensions
            h, w = frame.shape[:2]
            
            # Calculate aspect ratio preserving scale
            target_width = 480  # Match preview width
            scale = target_width / float(w)
            target_height = int(h * scale)
            
            # Resize frame while maintaining aspect ratio
            frame = cv2.resize(frame, (target_width, target_height))

            # Run detection and update UI
            processed, alerts, detections = process_frame(frame, ZONE_NAMES[i])
            detections_per_zone[ZONE_NAMES[i]] = detections

            # Extract drowning/safe state from last detection if available
            state = "Unknown"
            confidence = 0.0
            if alerts:
                # Try to parse status from alert
                last_status = alerts[-1]["status"]
                if "Drowning" in last_status:
                    state = "Drowning"
                    confidence = 0.95
                elif "Possible Drowning" in last_status:
                    state = "Possible Drowning"
                    confidence = 0.7
                elif "Safe" in last_status:
                    state = "Safe"
                    confidence = 0.2
            zone_states.append(state)
            zone_confidences.append(confidence)

            if alerts:
                log_alerts(alerts)
                for a in alerts:
                    self.alert_list.insertItem(0, f"{a['timestamp']} | {a['zone']} | {a['status']}")

            # Update preview with centered image
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Center the image in the preview label
            self.zone_cards[i].preview.setPixmap(pixmap)
            status_text = ("🚨 " + alerts[-1]["status"]) if alerts else "● Monitoring"
            self.zone_cards[i].status.setText(status_text)

        self.zone_states = zone_states
        self.zone_confidences = zone_confidences

        # Update state summary section (for monitor page)
        self._update_state_summary()
        # Update zone status sidebar page
        self._update_zone_status_page()

        # Update analytics
        if detections_per_zone:
            self._update_heatmap(detections_per_zone)

    def _update_state_summary(self):
        # Clear previous
        for i in reversed(range(self.state_summary_vbox.count())):
            item = self.state_summary_vbox.itemAt(i)
            w = item.widget()
            if w:
                self.state_summary_vbox.removeWidget(w)
                w.deleteLater()
        # Add new
        for i, (state, conf) in enumerate(zip(self.zone_states, self.zone_confidences)):
            color = NEON_GREEN if state == "Safe" else NEON_RED if state == "Drowning" else ACCENT if state == "Possible Drowning" else TEXT
            label = QLabel(f"{ZONE_NAMES[i]}: <b style='color:{color};'>{state}</b> <span style='color:#aaa;'>({conf:.2f})</span>")
            label.setFont(QFont("Segoe UI", 12))
            label.setStyleSheet(f"background: rgba(0,0,0,0.18); border-radius: 8px; padding: 8px 18px; color: {color};")
            self.state_summary_vbox.addWidget(label)
        # Totals
        total_drowning = sum(1 for s in self.zone_states if s == "Drowning")
        total_possible = sum(1 for s in self.zone_states if s == "Possible Drowning")
        total_safe = sum(1 for s in self.zone_states if s == "Safe")
        total_label = QLabel(
            f"<b style='color:{NEON_RED};'>Total Drowning: {total_drowning}</b> | "
            f"<b style='color:{ACCENT};'>Possible Drowning: {total_possible}</b> | "
            f"<b style='color:{NEON_GREEN};'>Safe: {total_safe}</b>"
        )
        total_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        total_label.setStyleSheet("margin-top: 12px;")
        self.state_summary_vbox.addWidget(total_label)

        # Update summary section style
        self.state_summary_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
            }}
            QWidget {{
                background: rgba(0,0,0,0.3);
                border-radius: 12px;
                padding: 12px;
            }}
        """)

    def _update_zone_status_page(self):
        # Clear previous
        for i in reversed(range(self.zone_status_vbox.count())):
            item = self.zone_status_vbox.itemAt(i)
            w = item.widget()
            if w:
                self.zone_status_vbox.removeWidget(w)
                w.deleteLater()
        # Add new
        for i, (state, conf) in enumerate(zip(self.zone_states, self.zone_confidences)):
            color = NEON_GREEN if state == "Safe" else NEON_RED if state == "Drowning" else ACCENT if state == "Possible Drowning" else TEXT
            label = QLabel(f"{ZONE_NAMES[i]}: <b style='color:{color};'>{state}</b> <span style='color:#aaa;'>({conf:.2f})</span>")
            label.setFont(QFont("Segoe UI", 15))
            label.setStyleSheet(f"background: rgba(0,0,0,0.18); border-radius: 8px; padding: 12px 24px; color: {color};")
            self.zone_status_vbox.addWidget(label)
        # Totals
        total_drowning = sum(1 for s in self.zone_states if s == "Drowning")
        total_possible = sum(1 for s in self.zone_states if s == "Possible Drowning")
        total_safe = sum(1 for s in self.zone_states if s == "Safe")
        total_label = QLabel(
            f"<b style='color:{NEON_RED};'>Total Drowning: {total_drowning}</b> | "
            f"<b style='color:{ACCENT};'>Possible Drowning: {total_possible}</b> | "
            f"<b style='color:{NEON_GREEN};'>Safe: {total_safe}</b>"
        )
        total_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        total_label.setStyleSheet("margin-top: 18px;")
        self.zone_status_vbox.addWidget(total_label)

    def _update_heatmap(self, detections_per_zone):
        """Update internal state for analytics."""
        update_heatmap(detections_per_zone)

        if not hasattr(self, "_tick_count"):
            self._tick_count = 0
        self._tick_count += 1

        # reduce refresh rate (now every 2 ticks for smoother charts)
        if self._tick_count >= 2:
            self._tick_count = 0
            if hasattr(self, "chart_container") and self.chart_container.isVisible():
                self._refresh_chart()

    def _refresh_chart(self):
        """Refresh whichever chart is active (bar or heatmap)."""
        if getattr(self, "_chart_switching", False):
            return  # skip refresh during switch debounce

        # Remove previous canvas from both widgets
        for widget in [self.chart_widget, self.heatmap_widget]:
            for i in reversed(range(widget.layout().count())):
                item = widget.layout().itemAt(i)
                w = item.widget()
                if w:
                    widget.layout().removeWidget(w)
                    w.setParent(None)
                    w.deleteLater()

        if self.chart_mode == "bar":
            self.chart_widget.show()
            self.heatmap_widget.hide()
            fig = plot_zone_bars()
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background: transparent;")
            self.chart_widget.layout().addWidget(canvas, alignment=Qt.AlignCenter)
        else:
            self.chart_widget.hide()
            self.heatmap_widget.show()
            fig = plot_heatmap()
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background: transparent;")
            self.heatmap_widget.layout().addWidget(canvas, alignment=Qt.AlignCenter)

    def _toggle_chart_mode(self):
        """Switch between heatmap and bar chart with debounce, no open/close effect."""
        self._chart_switching = True
        if self.chart_mode == "bar":
            self.chart_mode = "heatmap"
            self.toggle_btn.setText("Switch to Bar Chart")
        else:
            self.chart_mode = "bar"
            self.toggle_btn.setText("Switch to Heatmap")
        # Instead of hiding/showing the chart container, just switch widgets
        QTimer.singleShot(100, self._finish_chart_switch)

    def _finish_chart_switch(self):
        self._chart_switching = False
        self._refresh_chart()

    def pause(self):
        """Pause video processing."""
        self.is_paused = True

    def resume(self):
        """Resume video processing."""
        self.is_paused = False
        self._refresh_chart()

    def _update_analytics_report(self):
        # Generate analytics report from alerts.csv
        try:
            if os.path.exists("alerts.csv"):
                df = pd.read_csv("alerts.csv")
                total_alerts = len(df)
                drowning_alerts = df[df["status"].str.contains("Drowning")].shape[0]
                possible_alerts = df[df["status"].str.contains("Possible Drowning")].shape[0]
                safe_alerts = df[df["status"].str.contains("Safe")].shape[0]
                report = (
                    f"Total Alerts: {total_alerts}\n"
                    f"Drowning Events: {drowning_alerts}\n"
                    f"Possible Drowning Events: {possible_alerts}\n"
                    f"Safe Events: {safe_alerts}\n\n"
                    "Zone-wise breakdown:\n"
                )
                for zone in ZONE_NAMES:
                    zone_df = df[df["zone"] == zone]
                    report += (
                        f"  {zone}: "
                        f"Drowning: {zone_df[zone_df['status'].str.contains('Drowning')].shape[0]}, "
                        f"Possible: {zone_df[zone_df['status'].str.contains('Possible Drowning')].shape[0]}, "
                        f"Safe: {zone_df[zone_df['status'].str.contains('Safe')].shape[0]}\n"
                    )
                self.analytics_report.setPlainText(report)
            else:
                self.analytics_report.setPlainText("No analytics data available yet.")
        except Exception as e:
            self.analytics_report.setPlainText(f"Error loading analytics: {e}")
        # Generate analytics report from alerts.csv
        try:
            if os.path.exists("alerts.csv"):
                df = pd.read_csv("alerts.csv")
                total_alerts = len(df)
                drowning_alerts = df[df["status"].str.contains("Drowning")].shape[0]
                possible_alerts = df[df["status"].str.contains("Possible Drowning")].shape[0]
                safe_alerts = df[df["status"].str.contains("Safe")].shape[0]
                report = (
                    f"Total Alerts: {total_alerts}\n"
                    f"Drowning Events: {drowning_alerts}\n"
                    f"Possible Drowning Events: {possible_alerts}\n"
                    f"Safe Events: {safe_alerts}\n\n"
                    "Zone-wise breakdown:\n"
                )
                for zone in ZONE_NAMES:
                    zone_df = df[df["zone"] == zone]
                    report += (
                        f"  {zone}: "
                        f"Drowning: {zone_df[zone_df['status'].str.contains('Drowning')].shape[0]}, "
                        f"Possible: {zone_df[zone_df['status'].str.contains('Possible Drowning')].shape[0]}, "
                        f"Safe: {zone_df[zone_df['status'].str.contains('Safe')].shape[0]}\n"
                    )
                self.analytics_report.setPlainText(report)
            else:
                self.analytics_report.setPlainText("No analytics data available yet.")
        except Exception as e:
            self.analytics_report.setPlainText(f"Error loading analytics: {e}")
