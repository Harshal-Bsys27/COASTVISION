# 🏖️ CoastVision AI — Master Project Guide

> **Complete reference booklet for the CoastVision AI Beach Surveillance System**
> Last updated: March 2026 | Author: Harshal Barhate

## 0. Current Build Snapshot (March 2026)

This guide contains historical implementation details. For presentation-ready, current behavior, also see:

- `docs/presentation_system_guide.md`

Latest functional updates in the current build:

- Dedicated **Lifeguards** tab in React dashboard for Telegram operations.
- Telegram controls now include **Add / Test / Stop / Resume / Remove**.
- Zone-specific routing is enforced by lifeguard ID pattern (`lifeguard_<zoneId>`).
- Telegram registrations and pause state are persisted in `data/telegram_users.json`.
- Streaming is **HLS-first**, with automatic fallback to MJPEG and frame polling.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Folder Structure](#4-folder-structure)
5. [Backend (Python Flask Server)](#5-backend-python-flask-server)
6. [Frontend (React Dashboard)](#6-frontend-react-dashboard)
7. [YOLO Model & Detection](#7-yolo-model--detection)
8. [Dataset](#8-dataset)
9. [Training the Model](#9-training-the-model)
10. [Voice Alert System](#10-voice-alert-system)
11. [Lifeguard Mobile Alert System](#11-lifeguard-mobile-alert-system)
12. [Analytics Dashboard](#12-analytics-dashboard)
13. [HLS Video Streaming](#13-hls-video-streaming)
14. [API Reference](#14-api-reference)
15. [Configuration & Environment Variables](#15-configuration--environment-variables)
16. [How to Run the Project](#16-how-to-run-the-project)
17. [GPU Optimization (RTX 3050)](#17-gpu-optimization-rtx-3050)
18. [How Everything Connects](#18-how-everything-connects)
19. [Improving the Model](#19-improving-the-model)
20. [Troubleshooting](#20-troubleshooting)
21. [Development Changelog](#21-development-changelog)

---

## 1. Project Overview

### What is CoastVision?

CoastVision is an **AI-powered real-time beach/pool surveillance system** designed to detect drowning incidents and other emergencies using computer vision. It processes live video feeds from multiple camera zones, runs YOLO object detection on each frame, and provides:

- **Real-time video monitoring** with detection overlays (bounding boxes)
- **Automatic drowning detection** using a fine-tuned YOLOv8 model
- **Voice alerts** when emergencies are detected (browser speech synthesis)
- **Lifeguard notification system** with real-time push alerts
- **Analytics dashboard** with charts, detection timelines, and zone activity
- **Alert logging** with CSV records and snapshot images

### Purpose

Built as a **TE (Third Year Engineering) Project** to demonstrate how AI/ML can be applied to real-world safety — specifically **beach and pool surveillance** to prevent drowning deaths.

### How it Works (Simple Flow)

```
Camera Videos → Backend (YOLO Detection) → Annotated Frames → Frontend Dashboard
                        ↓
              Alert Generated (drowning detected)
                        ↓
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
   Voice Alert    Dashboard Alert   Lifeguard Push
   (Browser TTS)  (Visual + Sound)  (SSE/Mobile)
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER'S BROWSER                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │         React Dashboard (localhost:5173)            │ │
│  │                                                     │ │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────────────┐ │ │
│  │  │Monitoring│ │ Analytics │ │  Event History    │ │ │
│  │  │  (Tab 0) │ │  (Tab 1)  │ │    (Tab 2)       │ │ │
│  │  └──────────┘ └───────────┘ └──────────────────┘ │ │
│  │  ┌──────────┐ ┌───────────────────────────────┐  │ │
│  │  │ Settings │ │  Voice Alerts (Speech API)    │  │ │
│  │  │ (Tab 3)  │ │  Emergency Sound (AudioCtx)  │  │ │
│  │  └──────────┘ └───────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / MJPEG / SSE
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Python Flask Backend (port 8000)            │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Zone Workers (1 thread per camera zone)           │ │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │ │
│  │  │Zone 1│ │Zone 2│ │Zone 3│ │Zone N│   ...       │ │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘            │ │
│  │     │        │        │        │                   │ │
│  │     └────────┴────────┴────────┘                   │ │
│  │                    │                                │ │
│  │         ┌──────────▼──────────┐                    │ │
│  │         │   YOLO Inference    │                    │ │
│  │         │   (GPU - RTX 3050)  │                    │ │
│  │         └──────────┬──────────┘                    │ │
│  │                    │                                │ │
│  │    ┌───────────────┼───────────────┐               │ │
│  │    ▼               ▼               ▼               │ │
│  │ Annotated      Alert History   Lifeguard           │ │
│  │ JPEG/MJPEG     (in-memory)     Broadcast           │ │
│  │ Stream         + CSV logging   (SSE queues)        │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Video Files: frontend/dashboard/videos/zone*.mp4       │
│  Model: models/best.pt (fine-tuned) + yolov8n.pt (COCO)│
└─────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Server runtime |
| **Flask** | 3.1.0 | REST API web framework |
| **Flask-CORS** | 4.0.1 | Cross-origin requests (frontend ↔ backend) |
| **Waitress** | 3.0.2 | Production WSGI server (used in run_backend.ps1) |
| **PyTorch** | 2.9.1 | Deep learning framework (GPU inference) |
| **Ultralytics** | 8.3.241 | YOLOv8/YOLOv11 model training and inference |
| **OpenCV** | 4.12.0 | Video capture, frame processing, drawing detections |
| **NumPy** | 2.2.6 | Array operations |
| **CUDA** | 12.x | GPU acceleration via NVIDIA RTX 3050 |

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| **React** | 18.3.1 | UI component framework |
| **Material-UI (MUI)** | 6.2.0 | Pre-built UI components + styling |
| **Vite** | 5.4.8 | Fast bundler and dev server |
| **Chart.js** | 4.5.1 | Person Count Timeline charts |
| **react-chartjs-2** | 5.3.1 | React wrapper for Chart.js |
| **chartjs-adapter-date-fns** | 3.0.0 | Time-axis adapter for Chart.js |
| **date-fns** | 4.1.0 | Date utility library |
| **hls.js** | latest | HLS video streaming in browser |
| **Web Speech API** | Browser built-in | Voice alert announcements |
| **AudioContext API** | Browser built-in | Emergency alarm sounds |
| **react-zoom-pan-pinch** | 3.5.0 | Video feed zoom/pan in fullscreen |

### Hardware
| Component | Specs |
|---|---|
| **GPU** | NVIDIA RTX 3050 (75W, 6GB VRAM) |
| **CUDA Compute** | 8.6 (Ampere architecture) |
| **Inference Speed** | ~15-25ms per frame at 640px |

---

## 4. Folder Structure

```
COASTVISION/
│
├── backend/
│   ├── server.py              ← Main Flask backend (~1870 lines)
│   └── server_old.py          ← Previous version (backup)
│
├── frontend/
│   ├── web/
│   │   ├── src/
│   │   │   └── App.jsx        ← Main React dashboard (~2700 lines)
│   │   ├── index.html          ← HTML entry point
│   │   ├── package.json        ← NPM dependencies
│   │   ├── vite.config.js      ← Vite config (port 5173, host 0.0.0.0)
│   │   └── lifeguard.html      ← PWA page for lifeguard mobile access
│   └── dashboard/
│       └── videos/             ← Zone video files (zone1.mp4, zone2.mp4, ...)
│
├── models/
│   ├── best.pt                 ← Fine-tuned YOLO model (drowning detection)
│   └── drowning/               ← Additional model variants
│
├── dataset/
│   ├── data.yaml               ← Dataset config (3 classes)
│   ├── train/images/           ← Training images
│   ├── train/labels/           ← Training labels (YOLO format)
│   ├── valid/images/           ← Validation images
│   ├── valid/labels/           ← Validation labels
│   ├── test/images/            ← Test images
│   └── test/labels/            ← Test labels
│
├── scripts/
│   ├── train_yolov8.py         ← Training script
│   ├── inference_yolov8.py     ← Standalone inference test
│   ├── extract_frames.py       ← Extract frames from videos
│   └── check_class_distribution.py ← Dataset analysis
│
├── data/
│   ├── alerts/
│   │   ├── alerts.csv          ← Alert log (all recorded events)
│   │   ├── images/             ← Snapshot images of alert moments
│   │   └── lifeguards.json     ← Registered lifeguard data
│   ├── frames/                 ← Extracted video frames
│   └── raw_videos/             ← Original source videos
│
├── yolov8n.pt                  ← Pre-trained YOLOv8 nano (COCO - person detection)
├── yolo11n.pt                  ← Pre-trained YOLOv11 nano (COCO)
├── requirements.txt            ← Python dependencies
├── run_backend.ps1             ← PowerShell script to start backend
├── run_frontend.ps1            ← PowerShell script to start frontend
│
├── yolov5/                     ← YOLOv5 repository (reference)
├── YOLOv8/                     ← YOLOv8 training outputs
├── runs/                       ← Ultralytics training run outputs
│   └── detect/train/           ← Latest training results + weights
│
└── docs/                       ← Documentation files
```

---

## 5. Backend (Python Flask Server)

**File**: `backend/server.py` (~1870 lines)

### What it Does

The backend is the **brain** of CoastVision. It:

1. **Loads YOLO models** onto GPU (RTX 3050)
2. **Dynamically discovers video files** from `frontend/dashboard/videos/` (any format: mp4, avi, mkv, mov, webm, etc.)
3. **Runs inference** on each frame to detect objects
4. **Draws bounding boxes** with thick, visible borders and confidence scores
5. **Serves annotated video** via HLS (HTTP Live Streaming), MJPEG streams, or single JPEG frames
6. **Pipes annotated frames** to FFmpeg for HLS encoding (h264_nvenc or libx264 fallback)
7. **Records alerts** when drowning/emergency is detected (CSV + images)
8. **Manages lifeguards** (registration, zone assignment, real-time alerts)
9. **Tracks person counts** over time per zone (10-second intervals, 24h rolling window)
10. **Stores custom zone names** for user-defined zone labeling
11. **Manages video files** (upload, delete, rename via API)

### Key Components

#### 1. Dual-Model System
```
┌─────────────────────┐     ┌──────────────────────┐
│  models/best.pt     │     │  yolov8n.pt          │
│  (Fine-tuned)       │     │  (COCO Pre-trained)  │
│                     │     │                      │
│  Detects:           │     │  Detects:            │
│  - Drowning         │     │  - Person            │
│  - Person out water │     │  (80 COCO classes)   │
│  - Swimming         │     │                      │
└─────────────────────┘     └──────────────────────┘
```

The main model (`best.pt`) handles drowning-specific classes. If it doesn't have a "person" class, a secondary COCO model (`yolov8n.pt`) is automatically loaded to detect people. Both models run on GPU.

#### 2. Zone Worker Threads
Each video zone gets its own **daemon thread** that:
- Reads frames at the configured FPS (default 24)
- Runs YOLO inference every N frames (default every 2nd frame)
- Caches the latest annotated JPEG for API requests
- Holds detections for 1.5 seconds to prevent box flickering
- Records person count every 10 seconds (or immediately on count changes)
- Clears stale detections when no new detections arrive after hold period
- Loops the video when it reaches the end

#### 3. Person Count Tracking
```python
_zone_person_history: Dict[int, deque]  # (timestamp, count) per zone
MAX_HISTORY_POINTS = 1440               # 24 hours at ~1 per minute
```
- Records current person count every 10 seconds per zone
- Immediately records when count changes (0→3 or 5→2)
- Always records including 0 count (when no people detected)
- Data served via `/api/zones/{id}/timeline` endpoint
- Used by frontend Person Count Timeline charts

#### 4. Custom Zone Names
```python
_zone_custom_names: Dict[int, str]  # zone_id → display name
```
- Users can rename zones (e.g., "Zone 1" → "Main Beach Area")
- Names accessible via `GET/POST /api/zones/{id}/name`
- Included in `/api/zones` response

#### 5. Video File Management
- **Dynamic discovery**: Auto-scans video directory for all supported formats
- **Upload**: `POST /api/videos/upload` with multipart form data (up to 10GB)
- **Delete**: `DELETE /api/videos/<filename>` with zone cleanup
- **Rename**: `POST /api/videos/<filename>/rename`
- **Reload**: `POST /api/zones/reload` to force re-scan

#### 6. Alert System
When a detection exceeds the alert confidence threshold (default 0.55):
- Added to in-memory `ALERT_HISTORY` (last 400 alerts)
- Written to `data/alerts/alerts.csv`
- Snapshot image saved to `data/alerts/images/`
- Broadcast to connected lifeguards via SSE

#### 4. GPU Optimizations
```python
torch.set_grad_enabled(False)         # No autograd needed for inference
torch.backends.cudnn.benchmark = True  # Optimize convolution algorithms
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster math
MODEL.fuse()                           # Fuse Conv+BN layers
half=True                              # FP16 inference (halves VRAM usage)
```

#### 5. Detection Box Drawing
- **Thick bounding boxes** (4-6px) with black outline for visibility
- **White label background** with colored border
- **Confidence shown as decimal** (e.g., "Drowning 0.85")
- **Color coding**: Green for normal, Orange for drowning/emergency
- Designed to be clearly visible in the dashboard grid view

---

## 6. Frontend (React Dashboard)

**File**: `frontend/web/src/App.jsx` (~2700 lines)

### Dashboard Tabs

| Tab | Name | Purpose |
|---|---|---|
| **0** | Monitoring | Live video grid of all zones with detection overlays |
| **1** | Analytics | Sub-sectioned analytics: Overview, Person Count, Detections, Live Feed |
| **2** | Event History | Table of all recorded detection events |
| **3** | Settings | Backend config, GPU info, connection status |
| **4** | Video Manager | Upload, delete, rename video files |

### Tab 0: Monitoring (Live View)

The main surveillance view showing a responsive grid of all camera zones.

**Key Features:**
- **HLS streaming** (primary) for smooth, hardware-decoded video via `<video>` element
- **MJPEG streaming** as fallback for real-time video playback
- **Frame polling** as last-resort fallback
- **Detection count overlay** on each zone card
- **Person counter** showing how many people detected
- **Emergency indicator** (blinking red chip) when drowning detected
- **Click-to-expand** any zone for fullscreen view with zoom/pan
- **Custom zone names** displayed on card headers (click to rename inline)
- **Pause/Play** controls for all zones
- **Zone reload** to detect new video files without restart

**How Video Playback Works (HLS — Primary):**
```
1. Backend zone worker reads frame from MP4
2. YOLO runs inference → draws bounding boxes on frame
3. Raw BGR frame piped to FFmpeg subprocess via stdin
4. FFmpeg encodes to H.264 (libx264 / h264_nvenc) → HLS .ts segments
5. Frontend hls.js library loads /api/zones/{id}/hls/stream.m3u8
6. <video> element plays HLS → browser hardware-decodes H.264
7. Result: smooth, low-latency, low-bandwidth video playback
```

**Fallback Chain:**
```
HLS (hardware-decoded video) → MJPEG (JPEG stream) → Polling (single JPEGs)
```
If HLS fails (e.g., FFmpeg unavailable), the frontend automatically falls back to MJPEG, then to frame polling.

### Tab 1: Analytics

The analytics tab is divided into **4 navigable sub-sections** via a styled button bar. Only the active sub-section renders at a time (prevents stutter from rendering all charts simultaneously).

**Always Visible — Key Metrics Cards** (4 large cards):
- Total Detections (48px font)
- Monitored Zones
- Emergency Alerts
- Average Confidence

**Sub-section: Overview**

- **Pie Chart** (SVG donut, 240x240): Distribution of detection types with animated segments, glow effect, legend with count and percentage.
- **Bar Chart** (320px height): Zone activity comparison with color-coded bars, custom zone names as labels, summary stats (Most Active, Avg/Zone, Coverage).

**Sub-section: Person Count**

- **Person Count Timeline** (Chart.js Line charts, one per zone in 2-column grid)
  - Real-time person count over time (last 24 hours)
  - Per-zone accent colors (teal, green, amber, purple, pink, cyan)
  - Gradient fill under line, smooth curve (`tension: 0.35`)
  - **Axis labels**: X = "Time" (5-min intervals, HH:MM), Y = "People Count" (integers only)
  - **Stats row**: Current count, Peak, Average displayed above each chart
  - **Custom tooltip**: Shows exact time (HH:MM:SS) + "People in zone: X"
  - **Axis legend footer**: "X-axis: Time (HH:MM) | Y-axis: Number of people detected in zone"
  - Polls `/api/zones/{id}/timeline` every 5 seconds
  - Empty state: "Collecting data..." with description

**Sub-section: Detections**

- **Confidence Bar Chart**: Last 40 detection events as vertical bars (height = confidence, color = emergency/normal).
- **Detection Stats**: Normal Detections, Emergency Events, Total Events.
- **Per-Zone Breakdown Cards**: One card per zone showing detection count, percentage, progress bar.

**Sub-section: Live Feed**

- **Recent Activity Grid**: Up to 30 alert cards in 3-column grid (scrollable, 600px max height).
- Each card shows: label, custom zone name chip, confidence %, timestamp.
- Emergency alerts highlighted in red with pulsing dot.

### Tab 2: Event History

A scrollable table showing all recorded alerts with:
- Timestamp, Zone, Type (color-coded), Message, Confidence

### Tab 3: Settings

Displays backend configuration including:
- API endpoint and connection status
- GPU name, VRAM, CUDA status
- Confidence thresholds, FPS, inference settings

### Data Polling

The frontend polls the backend at regular intervals:
```javascript
usePollJson('/api/zones', 1200)       // Zone list every 1.2s (dynamic discovery)
usePollJson('/api/health', 3000)      // Backend health every 3s
usePollJson('/api/alerts', 1500)      // Alert list every 1.5s
usePollJson('/api/analysis', 1500)    // Analysis data every 1.5s
usePollJson('/api/zones/{id}/timeline', 5000)  // Person count per zone every 5s
```

### Key Components

| Component | Purpose |
|-----------|--------|
| `usePollJson(url, interval)` | Hook for polling any JSON API endpoint at intervals |
| `ZoneStreamView` | HLS > MJPEG > poll-based video player with auto-fallback |
| `PersonCountTimeline` | Chart.js line chart with gradient fill, stats cards, axis labels |
| `ZoneNameEditor` | Inline zone rename with dialog |
| `VideoManagerTab` | File management UI with drag-and-drop upload, delete, rename |
| `useEmergencyVoiceAlert` | Voice alert hook with rate limiting and deduplication |

### Tab 4: Video Manager

A dedicated file management interface for video feeds:
- **Upload**: Drag-and-drop or file picker with progress indicator (supports up to 10GB files)
- **File list**: Shows all video files in the videos directory with size and format
- **Delete**: Remove video files with confirmation
- **Rename**: Inline rename with validation
- **Dynamic**: Adding/removing files automatically updates the zone grid (no restart needed)

---

## 7. YOLO Model & Detection

### What is YOLO?

YOLO (You Only Look Once) is a real-time object detection model. It looks at an entire image once and predicts all bounding boxes and class labels simultaneously.

### Models Used

| Model | File | Purpose | Classes |
|---|---|---|---|
| **Fine-tuned YOLOv8n** | `models/best.pt` | Primary — drowning detection | Drowning, Person out of water, Swimming |
| **Pre-trained YOLOv8n** | `yolov8n.pt` | Secondary — person detection | 80 COCO classes (only "person" used) |
| **YOLOv11n** | `yolo11n.pt` | Alternative (newer architecture) | 80 COCO classes |

### Detection Pipeline

```
Video Frame (BGR)
    │
    ▼
Resize to max 1280px (COASTVISION_MAX_SIDE)
    │
    ▼
YOLO Inference (imgsz=640, FP16, GPU)
    │
    ├── Main Model (best.pt): drowning, swimming, person_out_of_water
    │
    └── Person Model (yolov8n.pt): person (only if main model lacks person class)
    │
    ▼
Filter by confidence thresholds:
  - Overlay threshold: 0.35 (all detections drawn on frame)
  - Alert threshold: 0.55 (only high-confidence = actual alerts)
  - Person threshold: 0.25 (lower for person detection)
    │
    ▼
Draw bounding boxes + labels on frame
    │
    ▼
Record alerts (CSV + image snapshot + broadcast to lifeguards)
    │
    ▼
Encode as JPEG (quality 88) → Serve to frontend
```

### Detection Classes

Defined in `dataset/data.yaml`:
```yaml
nc: 3
names:
  0: Drowning
  1: Person out of water
  2: Swimming
```

---

## 8. Dataset

### Source

Downloaded from **Roboflow Universe**:
- Project: "Swimming and Drowning Detection"
- URL: https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection
- License: CC BY 4.0

### Structure

```
dataset/
├── data.yaml          ← Points to train/valid/test splits
├── train/
│   ├── images/        ← Training images (JPG/PNG)
│   └── labels/        ← YOLO format labels (.txt files)
├── valid/
│   ├── images/        ← Validation images
│   └── labels/        ← Validation labels
└── test/
    ├── images/        ← Test images
    └── labels/        ← Test labels
```

### YOLO Label Format

Each `.txt` label file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1) relative to image dimensions.

Example:
```
0 0.45 0.62 0.12 0.18    ← Drowning at center (0.45, 0.62), size 12%×18%
2 0.70 0.50 0.15 0.25    ← Swimming at (0.70, 0.50), size 15%×25%
```

---

## 9. Training the Model

### Training Script

**File**: `scripts/train_yolov8.py`

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Start from pre-trained nano model

results = model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device=0  # GPU
)
```

### Recommended Settings for RTX 3050 6GB

```python
model.train(
    data='dataset/data.yaml',
    epochs=200,          # More epochs = better (with patience)
    imgsz=640,           # Optimal resolution
    batch=8,             # Safe for 6GB VRAM (try 12 if it fits)
    device=0,            # RTX 3050
    patience=30,         # Stop if no improvement for 30 epochs
    lr0=0.01,            # Default learning rate
    augment=True,        # Data augmentation (auto)
    mosaic=1.0,          # Mosaic augmentation
    flipud=0.5,          # Vertical flip
    fliplr=0.5,          # Horizontal flip
    mixup=0.1,           # MixUp augmentation
)
```

### After Training

The best model weights are saved at:
```
runs/detect/train/weights/best.pt
```

**To use the new model**: Copy `best.pt` to `models/best.pt` and restart the backend.

### Training Outputs

After training, you get:
- `runs/detect/train/weights/best.pt` — Best model weights
- `runs/detect/train/weights/last.pt` — Last checkpoint
- `runs/detect/train/results.png` — Training curves (loss, mAP)
- `runs/detect/train/confusion_matrix.png` — Class-wise accuracy
- `runs/detect/train/val_batch*.jpg` — Validation predictions

---

## 10. Voice Alert System

### How Voice Alerts Work

The frontend uses the **Web Speech Synthesis API** (browser built-in, no external service needed).

### Alert Flow

```
Backend detects drowning (conf ≥ 0.55)
    ↓
Alert added to /api/alerts response
    ↓
Frontend polls /api/alerts every 1.5s
    ↓
useEmergencyVoiceAlert() hook processes alerts
    ↓
Checks rate limiting:
  - Max 2 voice alerts per minute
  - Minimum 30 seconds between alerts
  - Deduplication by alert ID (timestamp + zone + label)
    ↓
Plays alarm sound (AudioContext oscillator — 2 beeps)
    ↓
Speaks announcement via SpeechSynthesis:
  "Alert! Drowning detected in Zone 1. Please check immediately."
```

### Voice Configuration

```javascript
utterance.rate = 1.0;       // Normal speed
utterance.volume = 1.0;     // Full volume
utterance.lang = 'en-US';   // English
// Prefers Google or Female voice if available
```

### Message Format

- **Drowning**: "Alert! Drowning detected in Zone {N}. Please check immediately."
- **Other emergency**: "Emergency alert in Zone {N}. Please respond."
- **All-clear**: Available via `speakAnnouncement()` function (rate 0.9)

### Controls

- **Sound toggle button** (speaker icon) in the dashboard header
- **Auto-announce** option for periodic status updates

---

## 11. Lifeguard Mobile Alert System

### Architecture

```
┌─────────────────┐       ┌──────────────────┐
│  Admin Dashboard │       │  Lifeguard Phone │
│  (React Web)     │       │  (Browser/PWA)   │
│                  │       │                  │
│  - See all guards│       │  - Login         │
│  - Assign zones  │──────→│  - Get alerts    │
│  - Send broadcast│  HTTP │  - Respond       │
│                  │  SSE  │  - SSE streaming  │
└────────┬─────────┘       └──────────────────┘
         │                          │
         │     ┌────────────┐       │
         └────→│  Backend   │←──────┘
               │  Flask API │
               │  Port 8000 │
               └────────────┘
```

### Lifeguard Registration & Zone Assignment

1. Lifeguard opens `lifeguard.html` on their phone (PWA capable)
2. Enters name and phone number → POST `/api/lifeguards/register`
3. Gets a session token stored in LocalStorage
4. Admin assigns zones via POST `/api/lifeguards/{id}/assign`

### Real-Time Alert Delivery

Two methods for delivering alerts to lifeguards:

1. **SSE (Server-Sent Events)** — Real-time push via `GET /api/lifeguards/{id}/stream`
   - Persistent connection
   - Instant alert delivery
   - Keepalive every 30 seconds
   
2. **Polling** — Fallback via `GET /api/lifeguards/{id}/alerts`
   - Lifeguard app polls every 3 seconds
   - Gets alerts for their assigned zones

### Alert Broadcast Logic

When the backend detects a drowning event:
```python
def _broadcast_alert_to_lifeguards(alert):
    zone = alert.get("zone")
    for lg_id, lg in LIFEGUARDS.items():
        assigned = lg.get("zones", [])
        # Empty zones list = receive ALL alerts
        if not assigned or zone in assigned:
            LIFEGUARD_ALERTS[lg_id].appendleft(alert)
            if lg_id in LIFEGUARD_SSE_QUEUES:
                LIFEGUARD_SSE_QUEUES[lg_id].put_nowait(alert)
```

### Lifeguard Mobile Page (PWA)

**File**: `frontend/web/lifeguard.html` (773 lines)

- Progressive Web App (installable on mobile home screen)
- Login with name/phone
- Real-time alert cards with zone, label, confidence
- "Respond" button to acknowledge alerts
- Vibration when new alert arrives
- Works over WiFi when phone is on the same network

### Data Persistence

Lifeguard data is stored in `data/alerts/lifeguards.json`:
```json
{
  "abc12345": {
    "id": "abc12345",
    "name": "John",
    "phone": "9876543210",
    "zones": [1, 2],
    "online": true,
    "last_seen": 1709312400.0,
    "created_at": "2026-03-01T10:00:00+00:00"
  }
}
```

---

## 12. Analytics Dashboard

### Sub-Section Navigation

The analytics tab is now divided into 4 sub-sections, controlled by `analyticsSection` state. A styled button bar at the top lets users switch between sections. Only the active section renders, preventing scroll stutter.

| Sub-Section | Key | Content |
|---|---|---|
| **Overview** | `overview` | Pie chart (detection types) + Bar chart (zone activity) + summary stats |
| **Person Count** | `timeline` | Per-zone person count timeline charts (Chart.js) |
| **Detections** | `detections` | Confidence bars + type breakdown + per-zone detection cards |
| **Live Feed** | `activity` | Recent alert events grid (up to 30 events) |

### Key Metrics (Always Visible)

| Metric | Source | Description |
|---|---|---|
| Total Detections | `analysis.alerts_total` | Count of all alerts recorded |
| Monitored Zones | `zones.length` | Number of active camera zones |
| Emergency Alerts | Filtered from alerts | Alerts containing "drown" or "emerg" |
| Avg Confidence | Calculated from alerts | Mean confidence of all alert items |

### Person Count Timeline (Chart.js)

Each zone gets a dedicated `PersonCountTimeline` component:

```
┌──────────────────────────────────────────┐
│  ● Zone Name                 data pts │
├──────────────────────────────────────────┤
│  👤 Current   📈 Peak   📊 Average  │
│     3          7        4.2       │
├──────────────────────────────────────────┤
│  People                             │
│  Count  5─┬───────┬───              │
│         │ ╱     │    ╲             │
│         3╱      │     ╲──2         │
│         │       │              │    │
│        0┴───────┴────────────┴──  │
│         10:00   10:15  10:30 Time  │
├──────────────────────────────────────────┤
│  X: Time (HH:MM) | Y: People count  │
└──────────────────────────────────────────┘
```

- **Accent colors**: Each zone gets a unique color from palette
- **Gradient fill**: Area under line fades from accent color to transparent
- **Tooltip**: Shows exact time (HH:MM:SS) + person count
- **5-min tick intervals** on X-axis, integer-only Y-axis
- **Data source**: Polls `/api/zones/{id}/timeline` every 5 seconds

### Data Source

All analytics data comes from:
- `/api/analysis` — aggregated alert stats
- `/api/alerts` — recent alert events
- `/api/zones/{id}/timeline` — person count history per zone

---

## 13. HLS Video Streaming

### Why HLS?

The original MJPEG streaming sends individual JPEG images over HTTP multipart. Each frame is independently compressed (~30–80 KB), browser does software decoding, and there's no inter-frame compression. Result: jittery playback and high bandwidth (~2–4 MB/s per zone).

**HLS (HTTP Live Streaming)** with H.264 encoding solves this:
- **Inter-frame compression**: Only changes between frames are encoded (P-frames), reducing bandwidth ~10×
- **Hardware decoding**: The browser's `<video>` element uses GPU video decoder, not CPU
- **Smooth playback**: Standard video playback pipeline — identical to watching a YouTube video
- **Low latency**: 1-second segments with a 4-segment sliding window ≈ 2–4s latency

### Architecture

```
Zone Worker Thread                FFmpeg Subprocess               Frontend (hls.js)
─────────────────                 ────────────────               ──────────────────
Read frame from MP4               Receives raw BGR via stdin      hls.js fetches .m3u8
  │                                  │                              │
Run YOLO inference                Convert BGR → YUV420P           Parse playlist
  │                                  │                              │
Draw bounding boxes               Encode H.264 (libx264)          Download .ts segment
  │                                  │                              │
Pipe raw frame bytes ──stdin──▶   Output HLS segments (.ts)       Feed to <video>
  to FFmpeg process                  + playlist (.m3u8)              │
                                     │                            Hardware H.264 decode
                                  Serve via Flask endpoints         │
                                  /api/zones/{id}/hls/            Render frames (GPU)
```

### Backend Implementation

Each zone's worker thread pipes annotated frames to a dedicated FFmpeg subprocess:

```python
# FFmpeg command (simplified)
ffmpeg -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 \
       -i pipe:0 -pix_fmt yuv420p \
       -c:v libx264 -preset ultrafast -tune zerolatency \
       -b:v 2M -g 24 -sc_threshold 0 \
       -f hls -hls_time 1 -hls_list_size 4 \
       -hls_flags delete_segments+append_list+independent_segments \
       -hls_segment_filename seg%05d.ts stream.m3u8
```

**Key parameters:**
| Parameter | Value | Purpose |
|---|---|---|
| `-pix_fmt yuv420p` | YUV 4:2:0 | Convert BGR to standard video chroma |
| `-preset ultrafast` | Fastest | Minimize encoding latency |
| `-tune zerolatency` | Low latency | Disable lookahead, smallest delay |
| `-b:v 2M` | 2 Mbps | Target bitrate per zone |
| `-g 24` | 24 frames | Keyframe every 1 second (= FPS) |
| `-hls_time 1` | 1 second | Segment duration |
| `-hls_list_size 4` | 4 segments | Sliding window size |

**Encoder selection:**
- **h264_nvenc** (GPU hardware encoder): Used when NVIDIA driver ≥ 570 is available. Offloads encoding to dedicated NVENC chip — zero impact on YOLO inference.
- **libx264** (software fallback): Used when NVENC is unavailable. The `ultrafast` preset keeps CPU usage minimal.

### Frontend Implementation

The frontend uses [hls.js](https://github.com/video-dev/hls.js/) to play HLS streams in a `<video>` element:

```javascript
import Hls from "hls.js";

const hls = new Hls({
  liveSyncDurationCount: 2,     // Stay 2 segments behind live
  liveMaxLatencyDurationCount: 4,
  maxBufferLength: 2,           // Buffer 2 seconds max
  enableWorker: true,           // Web Worker for parsing
  lowLatencyMode: true,
  backBufferLength: 0,          // Don't keep old segments
});
hls.loadSource(`${API}/api/zones/${zoneId}/hls/stream.m3u8`);
hls.attachMedia(videoElement);  // Attach to <video>
```

**Fallback chain** (automatic):
1. **HLS** → Try hls.js with `<video>` element
2. **MJPEG** → If HLS fails, fall back to `<img src="stream.mjpg">`
3. **Polling** → If MJPEG fails, fall back to polling `frame.jpg` every 200ms

### HLS Temp Directory

HLS segments are stored in a temporary directory that is automatically cleaned up on shutdown:
```
%TEMP%/coastvision_hls_XXXXX/
├── zone1/
│   ├── stream.m3u8      ← HLS playlist
│   ├── seg00000.ts      ← Video segment (1 second)
│   ├── seg00001.ts
│   ├── seg00002.ts
│   └── seg00003.ts
├── zone2/
│   └── ...
└── zone12/
    └── ...
```

### Bandwidth Comparison

| Method | Per Zone | 12 Zones | Quality |
|---|---|---|---|
| MJPEG | ~3 MB/s | ~36 MB/s | Moderate (no inter-frame) |
| HLS (2M) | ~0.25 MB/s | ~3 MB/s | High (H.264 inter-frame) |

---

## 14. API Reference

### Core APIs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health, GPU info, config |
| GET | `/api/zones` | List all camera zones and their status (includes custom names) |
| POST | `/api/zones/reload` | Force rescan for new zone video files |
| GET | `/api/zones/{id}/frame.jpg` | Latest annotated JPEG frame |
| GET | `/api/zones/{id}/stream.mjpg` | MJPEG video stream |
| GET | `/api/zones/{id}/hls/stream.m3u8` | HLS playlist for zone |
| GET | `/api/zones/{id}/hls/{filename}` | HLS .ts segment file |
| GET | `/api/hls/status` | HLS encoder status for all zones |
| GET | `/api/zones/{id}/detections` | Current detection boxes for a zone |
| GET | `/api/zones/{id}/timeline` | Person count history (last 24h, 10s intervals) |
| GET/POST | `/api/zones/{id}/name` | Get or set custom zone name |
| GET | `/api/analytics/timeline` | All zones timeline data in one call |
| GET | `/api/alerts` | Recent alerts (query: `?limit=120&zone=1`) |
| GET | `/api/analysis` | Aggregated stats (total, by_zone, by_label) |

### Video Management APIs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/videos` | List video files + directory path |
| POST | `/api/videos/upload` | Upload a new video file (multipart, up to 10GB) |
| DELETE | `/api/videos/{filename}` | Delete a video file + cleanup zone |
| POST | `/api/videos/{filename}/rename` | Rename a video file (body: `{new_name}`) |

### Lifeguard APIs

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/lifeguards/register` | Register new lifeguard (body: `{name, phone}`) |
| GET | `/api/lifeguards` | List all lifeguards (admin) |
| GET | `/api/lifeguards/{id}` | Get lifeguard details |
| POST | `/api/lifeguards/{id}/assign` | Assign zones (body: `{zones: [1,2]}`) |
| GET | `/api/lifeguards/{id}/alerts` | Get alerts for assigned zones |
| POST | `/api/lifeguards/{id}/respond` | Mark responding to alert |
| POST | `/api/lifeguards/{id}/heartbeat` | Update online status |
| GET | `/api/lifeguards/{id}/stream` | SSE real-time alert stream |
| POST | `/api/admin/broadcast` | Send manual alert to all lifeguards |

### Response Formats

**GET /api/health**:
```json
{
  "status": "ok",
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "gpu_vram_gb": 6.0,
  "cuda_smoke_ok": true,
  "hls_enabled": true,
  "hls_nvenc": false,
  "zones": 12,
  "conf": 0.35,
  "alert_conf": 0.55,
  "fps": 24,
  "imgsz": 640
}
```

**GET /api/zones**:
```json
{
  "items": [
    { "id": 1, "video": "beach_cam.mp4", "name": "Main Beach Area", "ok": true },
    { "id": 2, "video": "pool_east.mp4", "name": "Zone 2", "ok": true }
  ]
}
```

**GET /api/zones/{id}/timeline**:
```json
{
  "zone": 1,
  "timeline": [
    { "timestamp": 1709312400.0, "count": 3 },
    { "timestamp": 1709312410.0, "count": 5 },
    { "timestamp": 1709312420.0, "count": 4 }
  ]
}
```

**GET /api/alerts**:
```json
{
  "items": [
    {
      "ts": "2026-03-01T10:30:45+00:00",
      "zone": 1,
      "label": "Drowning",
      "conf": 0.87,
      "bbox": [120, 200, 280, 400],
      "msg": "Drowning detected"
    }
  ]
}
```

**GET /api/analysis**:
```json
{
  "alerts_total": 42,
  "alerts_by_zone": {"1": 20, "2": 15, "3": 7},
  "alerts_by_label": {"Drowning": 12, "Swimming": 25, "Person out of water": 5}
}
```

---

## 15. Configuration & Environment Variables

All backend settings are configurable via environment variables:

### Detection Settings
| Variable | Default | Description |
|---|---|---|
| `COASTVISION_CONF` | 0.35 | Minimum confidence to show detection overlay |
| `COASTVISION_PERSON_CONF` | 0.25 | Minimum confidence for person detection |
| `COASTVISION_ALERT_CONF` | 0.55 | Minimum confidence to trigger alert |
| `COASTVISION_IOU` | 0.45 | IoU threshold for NMS |
| `COASTVISION_MAX_DET` | 200 | Maximum detections per frame |
| `COASTVISION_ALERT_CLASSES` | (all) | Comma-separated alert class names |

### Performance Settings
| Variable | Default | Description |
|---|---|---|
| `COASTVISION_FPS` | 24 | Target frames per second |
| `COASTVISION_INFER_EVERY` | 2 | Run YOLO every N frames (2 = detect every other frame) |
| `COASTVISION_IMGSZ` | 640 | YOLO input resolution |
| `COASTVISION_MAX_SIDE` | 1280 | Max frame dimension before resize |
| `COASTVISION_HALF` | 1 | Use FP16 inference (saves VRAM) |
| `COASTVISION_DET_HOLD_S` | 1.5 | Hold detection boxes for N seconds |
| `COASTVISION_GRID_MAX_W` | 640 | Max width for grid thumbnails |
| `COASTVISION_GRID_JPEG_QUALITY` | 80 | JPEG quality for grid thumbnails |

### System Settings
| Variable | Default | Description |
|---|---|---|
| `COASTVISION_DEVICE` | cuda:0 | PyTorch device (cuda:0 or cpu) |
| `COASTVISION_VIDEO_DIR` | (auto) | Path to zone video files |
| `COASTVISION_ALERT_COOLDOWN_S` | 4 | Seconds between same-zone alerts |
| `COASTVISION_TF32` | 1 | Enable TF32 for faster GPU math |
| `COASTVISION_CUDNN_BENCHMARK` | 1 | Enable cuDNN auto-tuning |
| `COASTVISION_REQUIRE_CUDA` | 0 | Fail if CUDA unavailable |

### HLS Streaming Settings
| Variable | Default | Description |
|---|---|---|
| `COASTVISION_HLS_ENABLED` | 1 | Enable HLS streaming (0 to disable) |
| `COASTVISION_HLS_SEGMENT_S` | 1 | Duration of each HLS segment in seconds |
| `COASTVISION_HLS_LIST_SIZE` | 4 | Number of segments in the sliding window |
| `COASTVISION_HLS_BITRATE` | 2M | Target video bitrate (e.g., "2M", "1500k") |
| `COASTVISION_HLS_PRESET` | p4 | NVENC encoder preset (p1=fastest to p7=best quality) |

---

## 16. How to Run the Project

### Prerequisites

1. **Python 3.11+** with virtual environment
2. **Node.js 18+** with npm
3. **NVIDIA GPU driver** installed (for CUDA)
4. **FFmpeg** installed (for HLS streaming): `winget install Gyan.FFmpeg`
5. **Video files** placed in `frontend/dashboard/videos/` as `zone1.mp4`, `zone2.mp4`, etc.

### Step 1: Install Python Dependencies

```powershell
cd COASTVISION
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```powershell
cd frontend/web
npm install
```

### Step 3: Start Backend

**Option A — Using script (recommended):**
```powershell
.\run_backend.ps1
```

**Option B — Direct:**
```powershell
.\.venv\Scripts\Activate.ps1
python backend/server.py
```

Backend runs on **http://localhost:8000**

### Step 4: Start Frontend

**Option A — Using script:**
```powershell
.\run_frontend.ps1
```

**Option B — Direct:**
```powershell
cd frontend/web
npm run dev
```

Frontend runs on **http://localhost:5173**

### Step 5: Open Dashboard

Open `http://localhost:5173` in your browser. You should see:
- Zone video cards with live detection
- Analytics tab with charts
- Voice alerts when drowning is detected

### Adding New Camera Zones

**Option A — Via Video Manager (recommended):**
1. Go to the **Video Manager** tab in the dashboard
2. Drag-and-drop a video file or click to upload
3. New zone appears automatically within ~1 second

**Option B — Manual:**
1. Place any video file in `frontend/dashboard/videos/` (any name, any format: mp4, avi, mkv, mov, webm, etc.)
2. Zone appears automatically (polled every 1.2 seconds)
3. Or click the reload button in the dashboard

**Note**: Zone naming is no longer restricted to `zone*.mp4` — any video filename works. Zones are dynamically discovered and assigned IDs automatically. You can rename zones via the inline `ZoneNameEditor` on each card.

---

## 17. GPU Optimization (RTX 3050)

### VRAM Budget (6GB)

```
Model (best.pt FP16):     ~10 MB
Person model (FP16):      ~10 MB
CUDA context:             ~300 MB
Frame buffers:            ~50 MB
Inference workspace:      ~500 MB
────────────────────────────────
Total used:               ~870 MB
Available:                ~5.1 GB  ← Lots of headroom
```

### Performance Optimizations Applied

1. **FP16 (Half Precision)**: Halves memory and doubles throughput
2. **cuDNN Benchmark**: Auto-selects fastest convolution algorithm
3. **TF32**: Uses Tensor Cores for 3× faster matrix math
4. **Model Fusion**: Fuses Conv+BatchNorm layers (fewer operations)
5. **Gradient Disabled**: No backward pass needed (inference only)
6. **Inference Serialization**: Single GPU lock prevents OOM from concurrent threads
7. **Every-2nd-Frame Detection**: Cuts GPU load in half, boxes held for 1.5s

### Expected Performance

| Metric | Value |
|---|---|
| Inference time per frame | ~15-25ms |
| Effective FPS per zone | 12 (infer every 2nd frame at 24fps) |
| Concurrent zones supported | 4-8 zones smoothly |
| VRAM usage | ~1-2 GB |

---

## 18. How Everything Connects

### Complete Data Flow

```
1. VIDEO FILE (zone1.mp4)
   │
   ▼
2. ZONE WORKER THREAD (reads frames in loop, daemon thread)
   │
   ▼
3. FRAME PROCESSING
   ├── Resize to max 1280px
   ├── Run YOLO inference on GPU (every 2nd frame)
   ├── Draw bounding boxes (thick, with labels + confidence)
   ├── Encode as JPEG (quality 88)
   └── Cache in ZoneState.last_jpeg
   │
   ▼
4. API SERVING (Flask routes)
   ├── /api/zones/{id}/stream.mjpg → MJPEG video stream
   ├── /api/zones/{id}/frame.jpg → Single frame snapshot
   ├── /api/zones/{id}/detections → Current detection data
   └── /api/alerts → Alert history
   │
   ▼
5. FRONTEND (React, polling every 1.5-3s)
   ├── Monitoring: displays MJPEG stream in <img> tag
   ├── Analytics: processes alerts into charts
   ├── Voice: speaks drowning alerts via Speech API
   └── Sound: plays alarm beeps via AudioContext
   │
   ▼
6. LIFEGUARD SYSTEM (parallel path)
   ├── Backend broadcasts to SSE queues
   ├── Lifeguard phone receives via SSE or polling
   └── Lifeguard responds → logged in backend
```

### What Happens When Model Improves?

```
Better YOLO model (more data, more training)
    ↓
Higher accuracy detections
    ↓
Fewer false positives (less spam voice alerts)
    ↓
Better confidence scores (more reliable thresholds)
    ↓
Correct labels (Drowning vs Swimming vs Person)
    ↓
Voice says the right thing at the right time
    ↓
Analytics show meaningful data
    ↓
Lifeguards get accurate, timely alerts
```

**Nothing in the code needs to change** — just replace `models/best.pt` with the new trained model.

---

## 19. Improving the Model

### Strategy 1: More Training Data

- Target: **500+ images per class** minimum
- Sources: Roboflow Universe, Google Images, custom video screenshots
- Diversity: different lighting, water color, camera angles, crowds
- Use `scripts/extract_frames.py` to get frames from your own videos

### Strategy 2: Better Training Parameters

```python
model.train(
    data='dataset/data.yaml',
    epochs=200,         # Longer training
    imgsz=640,          # Standard resolution
    batch=8,            # Safe for 6GB VRAM
    device=0,           # RTX 3050
    patience=30,        # Early stopping
    augment=True,       # Auto augmentation
    mosaic=1.0,         # Mosaic: combines 4 images
    flipud=0.5,         # Flip vertically
    fliplr=0.5,         # Flip horizontally
    mixup=0.1,          # Blend two images together
    hsv_h=0.015,        # Hue shift
    hsv_s=0.7,          # Saturation shift
    hsv_v=0.4,          # Brightness shift
    degrees=10,         # Random rotation
    translate=0.1,      # Random translation
    scale=0.5,          # Random scaling
)
```

### Strategy 3: Continue from Existing Model

Instead of training from scratch, resume from your current best:
```python
model = YOLO('models/best.pt')  # Start from YOUR trained model
model.train(data='dataset/data.yaml', epochs=100, ...)
```

### Strategy 4: Larger Model (if VRAM allows)

```
yolov8n.pt  → 3.2M params (current - fast, less accurate)
yolov8s.pt  → 11.2M params (small - good balance)
yolov8m.pt  → 25.9M params (medium - more accurate, needs more VRAM)
```

Try `yolov8s.pt` with `batch=4` if you want better accuracy.

### Estimated Training Times (RTX 3050)

| Dataset Size | Epochs | Approx Time |
|---|---|---|
| 500 images | 200 | ~30-45 min |
| 1000 images | 200 | ~1-1.5 hrs |
| 2000 images | 300 | ~3-4 hrs |

---

## 20. Troubleshooting

### Backend Won't Start

| Problem | Solution |
|---|---|
| "No model weights found" | Place `best.pt` in `models/` folder |
| "CUDA smoke test failed" | Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Port 8000 already in use | Kill existing process or set `COASTVISION_PORT=8001` |
| Video not found | Place `zone1.mp4` etc. in `frontend/dashboard/videos/` |

### Frontend Issues

| Problem | Solution |
|---|---|
| Blank video cards | Check backend is running on port 8000 |
| CORS errors | Backend uses `flask-cors` — should work automatically |
| Voice not working | Click anywhere on page first (browser requires user interaction) |
| No analytics data | Wait for detections — analytics populate from alert history |

### GPU / Performance

| Problem | Solution |
|---|---|
| CUDA out of memory | Reduce `COASTVISION_IMGSZ` to 480 or `batch` to 4 during training |
| Slow inference | Ensure `COASTVISION_HALF=1` and `COASTVISION_INFER_EVERY=2` |
| Boxes flickering | Increase `COASTVISION_DET_HOLD_S` to 2.0 |
| Video stuttering | Lower `COASTVISION_FPS` to 12 or increase `COASTVISION_INFER_EVERY` to 3 |

### Model Quality

| Problem | Solution |
|---|---|
| Too many false positives | Increase `COASTVISION_ALERT_CONF` to 0.7 |
| Missing detections | Lower `COASTVISION_CONF` to 0.25 |
| Wrong labels | Retrain with more/cleaner data |
| Low confidence scores | Train more epochs, add more data |

---

## Quick Reference Card

```
┌────────────────────────────────────────────────┐
│           CoastVision Quick Reference          │
├────────────────────────────────────────────────┤
│                                                │
│  Start Backend:   .\run_backend.ps1            │
│  Start Frontend:  .\run_frontend.ps1           │
│                                                │
│  Dashboard:  http://localhost:5173             │
│  API:        http://localhost:8000             │
│  Lifeguard:  http://<PC-IP>:5173/lifeguard.html│
│                                                │
│  Add Zone:   Video Manager tab (upload)        │
│              or place any video in              │
│              frontend/dashboard/videos/         │
│                                                │
│  Rename Zone: Click zone name in dashboard     │
│                                                │
│  New Model:  Copy best.pt to models/best.pt   │
│              Restart backend                    │
│                                                │
│  Train:      python scripts/train_yolov8.py   │
│                                                │
│  Classes:    0=Drowning                        │
│              1=Person out of water             │
│              2=Swimming                        │
│                                                │
│  GPU:        RTX 3050 6GB VRAM                │
│  Framework:  PyTorch 2.x + CUDA 12.4          │
│  Model:      YOLOv8 Nano (fine-tuned)         │
│                                                │
│  Dashboard Tabs:                               │
│    0 = Monitoring (live video grid)            │
│    1 = Analytics (4 sub-sections)              │
│    2 = Event History                           │
│    3 = Settings                                │
│    4 = Video Manager                           │
│                                                │
│  Analytics Sub-sections:                       │
│    Overview | Person Count | Detections | Feed │
│                                                │
└────────────────────────────────────────────────┘
```

---

## 21. Development Changelog

All development changes in reverse chronological order.

### 2026-03-03 — Analytics Overhaul & Person Count Improvements

**Backend (`backend/server.py`)**
- **Person count accuracy fix**: Rewrote `_record_person_count()` to record every 10 seconds (was 60s) for smoother chart lines, record immediately on count changes, and always record including 0 count (previously skipped when no people detected).
- **Stale detection cleanup**: Added logic to clear `last_dets` after hold period when no new detections arrive, ensuring person count correctly drops to 0.
- **Detection loop update**: Now always calls `_record_person_count()` even when `dets` is empty.

**Frontend (`frontend/web/src/App.jsx`)**
- **Analytics sub-sections**: Split the monolithic analytics tab into 4 navigable sub-sections:
  - Overview — Pie chart + bar chart + zone stats
  - Person Count — Timeline charts per zone
  - Detections — Confidence bars + breakdown stats + per-zone cards
  - Live Feed — Recent activity grid (expanded from 12 to 30 events)
- **Sub-section tab bar**: Styled button bar with icons. Only active section renders → eliminates scroll stutter.
- **`analyticsSection` state**: New state variable controlling which sub-section is visible.
- **Bar chart / activity feed / "Most Active"**: Now show custom zone names.

---

### 2026-03-03 — Person Count Timeline UI Enhancement

**Frontend (`frontend/web/src/App.jsx`)**
- **`PersonCountTimeline` component rewrite**:
  - Axis titles (X: "Time", Y: "People Count") rendered on chart scales
  - Axis legend footer: "X-axis: Time (HH:MM) | Y-axis: Number of people detected in zone"
  - 5-minute tick intervals, integer-only Y-axis (`stepSize: 1`)
  - Stats row: Current count, Peak, Average
  - Per-zone accent colors (teal, green, amber, purple, pink, cyan)
  - Gradient fill under line (fades from color to transparent)
  - Custom tooltip: exact time (HH:MM:SS) + "People in zone: X"
  - Data points count chip, empty state, card redesign

---

### 2026-03-03 — Person Count Timeline & Custom Zone Names

**Backend (`backend/server.py`)**
- New data structures: `_zone_person_history` (deque per zone), `_zone_custom_names`, `MAX_HISTORY_POINTS = 1440`
- New helpers: `_record_person_count()`, `_get_zone_display_name()`, `_set_zone_name()`
- New endpoints: `GET /api/zones/{id}/timeline`, `GET/POST /api/zones/{id}/name`, `GET /api/analytics/timeline`
- Updated `/api/zones` response to include `name` field

**Frontend (`frontend/web/src/App.jsx`)**
- Chart.js integration: Installed chart.js, react-chartjs-2, chartjs-adapter-date-fns, date-fns
- `PersonCountTimeline` component: polls timeline API, renders Line chart with time x-axis
- `ZoneNameEditor` component: inline rename dialog
- `zoneNames` state (Map) synced from zones API
- Person Count Timeline section in analytics tab (one chart per zone, 2-column grid)

---

### 2026-03-02 — Dynamic Zone Discovery & Video Manager

**Backend (`backend/server.py`)**
- Dynamic zone discovery: auto-created from video files, no hardcoded zone limit, supports any video format
- Video management API: upload, delete, rename with validation
- Zone reload endpoint: `POST /api/zones/reload`
- `MAX_CONTENT_LENGTH` set to 10GB

**Frontend (`frontend/web/src/App.jsx`)**
- Video Manager tab: drag-and-drop upload, delete with confirmation, rename inline
- Dynamic grid: adapts to any number of zones
- Modal fix: broken template literal + pre-load frame on open

---

### 2026-03-01 — GPU Fix & Blank Screen Fix

**Backend**
- GPU issue: VS Code used system Python instead of venv Python → always start with `.venv\Scripts\python.exe`
- Verified: CUDA available, RTX 3050 6GB, `device: cuda:0`, `cudnn_benchmark: true`

**Frontend (`frontend/web/src/App.jsx`)**
- Blank screen fix: `zoneNames` state never declared — added `useState(new Map())`
- Removed duplicate zone name sync useEffect
- Added Chart.js Filler plugin registration

---

### 2026-03-01 — Git Cleanup

- 10k changes caused by untracked `LifeguardApp/` folder (node_modules, .expo, android build artifacts)
- Added `LifeguardApp/` to `.gitignore`, committed and pushed to main (commit `1826311`)
- Created `feat/lifeguard-app` branch for mobile app work

---

### Pre-2026-03 — Initial Build

- YOLO model training pipeline (YOLOv8 + YOLOv5)
- Flask backend with inference loop, HLS/MJPEG streaming, alerts
- React dashboard with MUI, zone grid, event logs, settings
- Alert engine with confidence thresholds and CSV logging
- Voice announcement (Web Speech API)
- Lifeguard management API (register, assign, heartbeat, SSE)
- Colab training documentation

---

*This document covers the complete CoastVision AI project as of March 2026. It is updated whenever features are added, bugs are fixed, or architecture changes. See also `docs/project_plan.md` for a summary/viva guide explaining the system from end to end.*
