# CoastVision AI — Project Plan & Summary

> **A complete walkthrough of the CoastVision system — from idea to deployment.**
> Use this document to understand how everything works, end-to-end.
>
> **Repository:** [Harshal-Bsys27/COASTVISION](https://github.com/Harshal-Bsys27/COASTVISION)
> **Detailed Technical Reference:** See `COASTVISION_MASTER_GUIDE.md` for API docs, config, troubleshooting, and the full changelog.

## Current Build Snapshot (March 2026)

This document includes historical planning context. The current implementation now includes:

- Dedicated Lifeguards tab in the React dashboard
- Telegram-based lifeguard notifications with zone-specific routing
- Add, Test, Stop, Resume, and Remove controls per lifeguard
- HLS-first streaming with automatic fallback to MJPEG and frame polling

For the most presentation-ready, up-to-date explanation, see:

- `docs/presentation_system_guide.md`

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Our Approach (Solution Overview)](#2-our-approach-solution-overview)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Step-by-Step: How We Built It](#4-step-by-step-how-we-built-it)
   - Phase 1: Data Collection & Annotation
   - Phase 2: Model Training (YOLO)
   - Phase 3: Evaluation & Iteration
   - Phase 4: Backend — The Brain of the System
   - Phase 5: Video Streaming Pipeline
   - Phase 6: Frontend — The Dashboard
   - Phase 7: Alert & Safety Systems
   - Phase 8: Analytics & Monitoring
5. [How Detection Actually Works (Inference Pipeline)](#5-how-detection-actually-works-inference-pipeline)
6. [How the Frontend Talks to the Backend](#6-how-the-frontend-talks-to-the-backend)
7. [Dynamic Zone Discovery — No Hardcoding](#7-dynamic-zone-discovery--no-hardcoding)
8. [Key Features Summary](#8-key-features-summary)
9. [Tech Stack at a Glance](#9-tech-stack-at-a-glance)
10. [Future Enhancements](#10-future-enhancements)
11. [Original Project Plan (Phases)](#11-original-project-plan-phases)
12. [Useful Links & Resources](#12-useful-links--resources)

---

## 1. The Problem

Drowning is one of the leading causes of accidental death, especially at beaches and swimming pools. Traditional lifeguard surveillance depends entirely on human attention, which can fail due to fatigue, blind spots, crowded environments, or momentary distraction.

**CoastVision solves this** by adding an AI-powered second pair of eyes — a real-time computer vision system that continuously monitors video feeds from cameras, detects people and drowning incidents, and instantly alerts lifeguards through voice announcements, dashboard warnings, and mobile push notifications.

---

## 2. Our Approach (Solution Overview)

The core idea is simple:

> **Camera feeds → AI model detects drowning → System alerts humans immediately.**

We break this into four layers:

1. **Data & Model** — We collected and annotated drowning/swimming images, then trained a YOLO (You Only Look Once) object detection model to recognize three classes: *Drowning*, *Swimming*, and *Person out of water*.

2. **Backend Server** — A Python Flask server reads video files, runs the YOLO model on each frame using GPU acceleration (NVIDIA RTX 3050), draws bounding boxes on detected objects, and serves the annotated video to the frontend.

3. **Frontend Dashboard** — A React web application displays live video feeds from all camera zones in a grid, shows real-time analytics with charts, and provides voice alerts when an emergency is detected.

4. **Safety Layer** — When the model detects drowning with high confidence, the system triggers voice announcements in the browser, logs alerts to CSV, saves snapshot images, and pushes zone-specific notifications to lifeguards via Telegram.

```
[Camera Videos] → [YOLO on GPU] → [Annotated Frames + Alerts]
                                          ↓
                     ┌────────────────────┼────────────────────┐
                     ↓                    ↓                    ↓
              Voice Alert          Dashboard View      Lifeguard Telegram
           (Browser Speech)      (React + Charts)      (Zone-specific push)
```

---

## 3. System Architecture at a Glance

```
┌────────────────────────────────────────┐
│         User's Browser (port 5173)     │
│                                        │
│   React + MUI + Chart.js + hls.js      │
│   Tabs: Monitoring | Analytics |       │
│         Event Logs | Settings |        │
│         Lifeguards | Video Manager     │
└──────────────┬─────────────────────────┘
               │  HTTP REST API (polling)
               │  HLS / MJPEG streams
               ▼
┌────────────────────────────────────────┐
│    Python Flask Backend (port 8000)    │
│                                        │
│   • YOLO inference on GPU (RTX 3050)   │
│   • 1 thread per camera zone           │
│   • HLS encoding via FFmpeg            │
│   • Alert engine + CSV logging         │
│   • Person count tracking (per zone)   │
│   • Lifeguard management + Telegram    │
└──────────────┬─────────────────────────┘
               │
      ┌────────▼────────┐
      │  Video Files     │  frontend/dashboard/videos/
      │  YOLO Model      │  models/best.pt
      │  Alert Data      │  data/alerts/
      └─────────────────┘
```

---

## 4. Step-by-Step: How We Built It

### Phase 1: Data Collection & Annotation

**Goal:** Get labeled images of drowning, swimming, and people near water.

1. **Sources**: We used the Roboflow Universe dataset ("Swimming and Drowning Detection"), supplemented with frames extracted from our own videos.
2. **Frame Extraction**: The script `scripts/extract_frames.py` pulls individual frames from raw video files stored in `data/raw_videos/`.
3. **Annotation**: Each image was annotated with bounding boxes in YOLO format — one `.txt` file per image with lines like `<class_id> <x_center> <y_center> <width> <height>` (all normalized 0–1).
4. **Dataset split**: Organized into `dataset/train/`, `dataset/valid/`, `dataset/test/` with parallel `images/` and `labels/` folders.
5. **Classes** (defined in `dataset/data.yaml`):
   - `0` = Drowning
   - `1` = Person out of water
   - `2` = Swimming

**Why this matters:** The quality and diversity of your dataset directly determines how well the model performs. More varied images (different lighting, angles, water colors, crowds) = better generalization.

---

### Phase 2: Model Training (YOLO)

**Goal:** Train a YOLO model that can detect drowning in real-time.

1. **Model choice**: YOLOv8 Nano (`yolov8n.pt`) — optimized for speed on edge GPUs like the RTX 3050. YOLO processes the entire image in one pass ("You Only Look Once"), making it extremely fast for real-time detection.
2. **Training**: Used the Ultralytics library. The training script (`scripts/train_yolov8.py`) fine-tunes the pre-trained COCO model on our custom drowning dataset.
   ```
   Pre-trained COCO weights (80 generic classes)
       ↓ fine-tune on our 3-class dataset
   Custom model → models/best.pt (Drowning, Swimming, Person out of water)
   ```
3. **Hardware**: Trained on an NVIDIA RTX 3050 (6GB VRAM) with FP16 precision, batch size 8, image size 640px.
4. **Output**: The best checkpoint is saved as `runs/detect/train/weights/best.pt`, which we copy to `models/best.pt` for the backend to use.

**Key insight:** We use a **dual-model system** — the fine-tuned model (`best.pt`) detects drowning-specific classes, and a secondary COCO model (`yolov8n.pt`) handles generic "person" detection. This ensures we can always count people even if the custom model doesn't include a "person" class.

---

### Phase 3: Evaluation & Iteration

- After training, we evaluated using **mAP** (mean Average Precision), precision, and recall on the validation and test sets.
- The Ultralytics framework automatically generates training curves (`results.png`), confusion matrices, and sample prediction images in the `runs/detect/train/` folder.
- If accuracy was low, we improved by: adding more training data, increasing epochs, adjusting augmentation settings (mosaic, flip, mixup), or trying a larger model variant (yolov8s).
- The model can be improved at any time by retraining and simply replacing `models/best.pt` — no code changes needed.

---

### Phase 4: Backend — The Brain of the System

**File:** `backend/server.py` (~1870 lines)

The backend is a Python Flask server that does all the heavy lifting:

1. **Startup**: Loads the YOLO model(s) onto GPU, scans the video directory for camera feeds, and creates one **daemon thread per zone** (camera).
2. **Zone workers**: Each thread reads video frames in a loop, runs YOLO inference every 2nd frame (for efficiency), draws bounding boxes on the frame, and caches the annotated JPEG.
3. **Detection hold**: Bounding boxes are "held" for 1.5 seconds even if the model doesn't detect the object in the next frame — this prevents flickering.
4. **Person count tracking**: Every 10 seconds (and immediately on count changes), the backend records how many people are in each zone. This data is served to the frontend for timeline charts.
5. **Alert engine**: When a detection exceeds the alert confidence threshold (0.55), it's logged to CSV, a snapshot image is saved, and lifeguards are notified.
6. **API**: Exposes ~30 REST endpoints for the frontend to consume (zones, streams, detections, alerts, analytics, video management, lifeguard management).

**Key design decisions:**
- **GPU inference serialization**: A threading lock ensures only one zone runs YOLO at a time, preventing GPU out-of-memory errors.
- **FP16 inference**: Uses half-precision floating point to halve VRAM usage and double throughput.
- **Dynamic zone discovery**: Zones are auto-created from any video file placed in the videos directory — no hardcoded zone count.

---

### Phase 5: Video Streaming Pipeline

The backend needs to deliver annotated video frames to the browser efficiently. We implemented a **3-tier fallback chain**:

1. **HLS (HTTP Live Streaming)** — Primary method
   - Each zone's worker pipes raw annotated frames into an FFmpeg subprocess via stdin.
   - FFmpeg encodes them to H.264 video and outputs 1-second HLS segments (`.ts` files) with a playlist (`.m3u8`).
   - The frontend uses the `hls.js` library to play the stream in a standard `<video>` element.
   - **Advantage**: Browser hardware-decodes H.264 (GPU), giving smooth playback at ~0.25 MB/s per zone instead of ~3 MB/s with MJPEG.

2. **MJPEG (Motion JPEG)** — Fallback
   - Streams individual JPEG frames over a multipart HTTP response.
   - Works without FFmpeg but uses more bandwidth and CPU for decoding.

3. **Frame Polling** — Last resort
   - Frontend polls `frame.jpg` every 200ms.
   - Simplest approach, works in any browser, but provides the lowest quality experience.

**The frontend automatically tries HLS first, then falls back to MJPEG, then to polling** — no user intervention needed.

---

### Phase 6: Frontend — The Dashboard

**File:** `frontend/web/src/App.jsx` (~2700 lines)

A single-page React application built with Material-UI, organized into 5 tabs:

| Tab | What it Shows |
|-----|--------------|
| **Monitoring** | Live video grid of all camera zones with detection overlays. Click any zone to expand fullscreen with zoom/pan. Shows person count and emergency indicators per zone. |
| **Analytics** | 4 sub-sections — Overview (pie/bar charts), Person Count (real-time timeline charts per zone), Detections (confidence analysis), Live Feed (recent alerts grid). |
| **Event History** | Searchable table of all recorded alert events with timestamps, zones, types, and confidence scores. |
| **Settings** | Displays backend config, GPU info, connection status, threshold values. |
| **Video Manager** | Upload, delete, and rename video files with drag-and-drop. Adding a video automatically creates a new camera zone. |

**How the frontend stays updated**: It uses a custom `usePollJson` hook that polls backend APIs at regular intervals (1.2s for zones, 1.5s for alerts, 3s for health, 5s for timeline data). This keeps the dashboard in near-real-time sync without WebSockets.

---

### Phase 7: Alert & Safety Systems

When the YOLO model detects drowning (confidence ≥ 0.55), multiple safety layers activate simultaneously:

1. **Voice Alert** — The browser uses the Web Speech Synthesis API to speak: *"Alert! Drowning detected in Zone X. Please check immediately."* Rate-limited to 2 alerts per minute to prevent spam.

2. **Alarm Sound** — An AudioContext oscillator plays two sharp beeps before the voice announcement.

3. **Dashboard Visual** — The zone card shows a blinking red "EMERGENCY" chip. Alert appears in Event History and the Live Feed.

4. **CSV Logging** — Alert details (timestamp, zone, label, confidence, bounding box) are appended to `data/alerts/alerts.csv`. A snapshot image of the moment is saved to `data/alerts/images/`.

5. **Lifeguard Push** — Registered lifeguards receive zone-specific alerts on Telegram. Each lifeguard mapping uses `lifeguard_<zoneId>` and supports Test, Stop, Resume, and Remove controls from the Lifeguards tab.

---

### Phase 8: Analytics & Monitoring

The Analytics tab gives a comprehensive view of what's happening across all zones:

- **Overview**: Pie chart showing distribution of detection types (Drowning vs Swimming vs Person out of water). Bar chart comparing zone activity levels.
- **Person Count Timeline**: Real-time line charts (one per zone) showing how many people were detected over time. Built with Chart.js, featuring gradient fills, custom tooltips, and per-zone accent colors. Data updates every 5 seconds.
- **Detections**: Confidence analysis bar chart showing the distribution of detection confidence scores. Per-zone breakdown cards.
- **Live Feed**: Rolling grid of the 30 most recent alert events with emergency highlighting.

**Key metrics displayed**: Total detections, monitored zones count, emergency alerts, average confidence.

---

## 5. How Detection Actually Works (Inference Pipeline)

This is the core of the system — what happens to every video frame:

```
1. Zone worker reads a frame from the video file (e.g., zone1.mp4)
       ↓
2. Frame is resized to max 1280px (maintains aspect ratio)
       ↓
3. YOLO inference runs on GPU (FP16, imgsz=640)
   ├── Main model (best.pt): Detects Drowning, Swimming, Person out of water
   └── Person model (yolov8n.pt): Detects "person" class (if main model doesn't have it)
       ↓
4. Results are filtered by confidence thresholds:
   • Overlay threshold (0.35): All detections drawn on the frame as bounding boxes
   • Alert threshold (0.55): Only high-confidence detections trigger alerts
   • Person threshold (0.25): Lower bar for person counting
       ↓
5. Bounding boxes drawn on the frame with thick borders, labels, and confidence scores
       ↓
6. If alert-worthy detection found → log it, save snapshot, broadcast to lifeguards
       ↓
7. Person count recorded (stored per zone, 10-second intervals)
       ↓
8. Annotated frame encoded as JPEG → cached for API serving
   Also piped to FFmpeg for HLS encoding (if HLS enabled)
       ↓
9. Frontend fetches the frame/stream and displays it in the dashboard
```

**Performance**: ~15–25ms per inference on RTX 3050. With inference every 2nd frame at 24 FPS, each zone effectively gets 12 detections per second.

---

## 6. How the Frontend Talks to the Backend

The frontend and backend communicate entirely via **HTTP REST APIs**. The frontend polls several endpoints at regular intervals:

| What | Endpoint | Poll Interval | Purpose |
|------|----------|--------------|---------|
| Zone list | `/api/zones` | 1.2s | Discover new/removed camera zones dynamically |
| Health | `/api/health` | 3s | GPU status, connection, config values |
| Alerts | `/api/alerts` | 1.5s | New alerts for voice announcements + event log |
| Analysis | `/api/analysis` | 1.5s | Aggregated stats for analytics charts |
| Person count | `/api/zones/{id}/timeline` | 5s | Timeline data for person count charts |

For video, the frontend connects to HLS streams (`/api/zones/{id}/hls/stream.m3u8`) which are persistent connections managed by hls.js, not polling.

---

## 7. Dynamic Zone Discovery — No Hardcoding

One of the key architectural decisions: **zones are not hardcoded**. The system dynamically discovers camera zones by scanning the `frontend/dashboard/videos/` directory for video files.

- Drop any video file (mp4, avi, mkv, mov, webm, etc.) into the videos folder → new zone appears in the dashboard within ~1 second.
- Remove a video file → zone disappears.
- Rename zone via the inline editor on the dashboard.
- Or use the **Video Manager** tab to upload/delete/rename videos from the browser.

This means the system can scale from 1 camera to as many as the GPU can handle, without any code changes or server restarts.

---

## 8. Key Features Summary

| Feature | Description |
|---------|-------------|
| **YOLO Object Detection** | Fine-tuned YOLOv8 model detects drowning, swimming, person out of water in real-time |
| **Multi-Zone Monitoring** | Unlimited camera zones, each processed in its own thread |
| **HLS Streaming** | Smooth, hardware-decoded video via H.264 encoding + hls.js |
| **MJPEG/Polling Fallback** | Automatic downgrade if HLS unavailable |
| **Voice Alerts** | Browser speaks drowning alerts via Web Speech API |
| **Lifeguard Telegram Alerts** | Zone-specific Telegram notifications with Add/Test/Stop/Resume/Remove controls |
| **Person Count Timeline** | Chart.js line graphs showing people count per zone over 24 hours |
| **Analytics Dashboard** | 4 sub-sections: Overview, Person Count, Detections, Live Feed |
| **Custom Zone Names** | Rename zones inline from the dashboard |
| **Video Manager** | Upload, delete, rename camera feeds from the browser |
| **Dynamic Zones** | Auto-discovery from video directory — no hardcoding |
| **GPU Acceleration** | FP16 inference, cuDNN benchmark, TF32, model fusion |
| **CSV Alert Logging** | All alerts persisted with timestamps and snapshot images |
| **Click-to-Expand** | Fullscreen zone view with zoom/pan |

---

## 9. Tech Stack at a Glance

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | YOLOv8 (Ultralytics), PyTorch, CUDA 12.x, RTX 3050 |
| **Backend** | Python, Flask, OpenCV, FFmpeg, NumPy |
| **Frontend** | React, Material-UI (MUI), Vite, Chart.js, hls.js |
| **Alerts** | Web Speech API, AudioContext, Telegram Bot API |
| **Data** | Roboflow (annotation), YOLO format labels, CSV logging |
| **Hosting** | Local machine (localhost:8000 backend, localhost:5173 frontend) |

---

## 10. Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Heatmap / Crowd Density** | Visual overlay showing where people cluster in a zone |
| **Android Lifeguard App** | Native mobile app (React Native/Expo) replacing the PWA |
| **Rip Current Detection** | New YOLO class + water flow analysis for rip current warnings |
| **Pose Estimation** | Use MediaPipe or OpenPose to track swimmer body posture for more accurate drowning detection |
| **Citizen-Science Uploads** | Allow public to submit beach footage for dataset expansion |
| **Cloud Deployment** | Host backend on AWS/GCP with GPU instances for multi-site deployment |

---

## 11. Original Project Plan (Phases)

This was our initial development roadmap, which the project followed and expanded upon:

### Phase 1: Data Collection & Annotation
- Gather datasets from public sources (Roboflow, UCF101, HMDB51) and custom video collection.
- Organize raw videos/images in `/data/raw_videos/`.
- Extract frames using `/scripts/extract_frames.py`.
- Annotate images with bounding boxes in YOLO format, store in `/dataset/`.

### Phase 2: Model Development
- Train YOLOv8 model for drowning/person detection.
- Experiment with model sizes (nano, small, medium, large) based on hardware constraints.
- Tune hyperparameters (epochs, batch size, augmentation). Scripts in `/scripts/train_yolov8.py`.

### Phase 3: Evaluation
- Evaluate on validation/test sets using mAP, precision, recall, F1.
- Analyze confusion matrices and training curves.
- Document findings in `/docs`.

### Phase 4: Inference & Deployment
- Real-time inference via Flask backend with GPU acceleration.
- HLS + MJPEG streaming pipeline for browser-compatible video delivery.
- React-based web dashboard with Material-UI.
- Alert engine with voice announcements and lifeguard notification.

### Phase 5: Future Enhancements
- Pose estimation for advanced swimmer tracking.
- Rip current detection, crowd management and density mapping.
- Heatmaps and risk analytics.
- Cloud deployment for multi-site monitoring.

---

## 12. Useful Links & Resources

- [Roboflow Universe — Swimming & Drowning Detection Dataset](https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [UCF101 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51 Human Motion Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Roboflow — Image Annotation Platform](https://roboflow.com/)
- [LabelImg — Manual Annotation Tool](https://github.com/tzutalin/labelImg)
- [hls.js — HLS Player for Browsers](https://github.com/video-dev/hls.js/)
- [Chart.js — JavaScript Charting Library](https://www.chartjs.org/)

---

*This document is the summary/viva guide for the CoastVision project. For detailed API references, configuration options, full changelog, and troubleshooting, refer to `COASTVISION_MASTER_GUIDE.md`.*
