# CoastVision

AI-powered coastal surveillance system for multi-zone beach monitoring, drowning-risk detection, and real-time alerting.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb)
![YOLO](https://img.shields.io/badge/Model-Ultralytics%20YOLO-111827)
![Platform](https://img.shields.io/badge/Platform-Windows%20focused-2563eb)

## Table of Contents

- [Overview](#overview)
- [Current Capabilities](#current-capabilities)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Run Modes](#run-modes)
- [Configuration](#configuration)
- [API Reference (Key Endpoints)](#api-reference-key-endpoints)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Latest Model Performance](#latest-model-performance)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

CoastVision is an end-to-end AI surveillance project built to support beach safety operations. The system ingests zone-wise video feeds, performs YOLO-based detection, serves annotated streams through a Flask backend, and displays live monitoring + analytics in a modern React dashboard.

Primary goal: help lifeguards identify risky events quickly with visual context, timeline data, and operational alert workflows.

## Current Capabilities

- Multi-zone video ingestion and stable zone mapping.
- Real-time object detection using custom trained model (`models/best.pt`).
- Live stream access with HLS-first playback and MJPEG/frame fallbacks for reliability.
- Dashboard with:
  - live zone cards
  - fullscreen zone monitoring
  - analytics charts
  - event logs and alerts
  - video management (upload, rename, delete)
- Dedicated Lifeguards tab with Telegram controls:
  - register/test/remove chat IDs
  - stop/resume notifications without losing registration
  - zone-specific routing (`lifeguard_<zoneId>` -> Zone `<zoneId>` alerts)
- Lifeguard workflow APIs:
  - registration
  - zone assignment
  - alert fetch and response
  - heartbeat and stream channels
- Validation pipeline for reporting and viva-ready performance summaries.

## System Architecture

```text
Video Files / Feeds
        |
        v
Flask Backend (backend/server.py)
  - Zone manager
  - YOLO inference
  - Alert generation
  - Analytics + timelines
  - MJPEG/HLS/frame APIs
        |
        v
React Dashboard (frontend/web)
  - Live monitoring UI
  - Analytics and logs
  - Lifeguard/admin actions
```

## Tech Stack

| Layer | Technologies |
|------|--------------|
| Backend | Python, Flask, Waitress, Flask-CORS, OpenCV |
| AI/ML | PyTorch, Ultralytics YOLO, NumPy |
| Frontend | React, Vite, MUI, Chart.js, HLS.js |
| Data | CSV logs, YOLO dataset YAML, image/video assets |
| Tooling | PowerShell scripts, pip, npm |

## Project Structure

```text
COASTVISION/
├── backend/                  # Flask backend + detection pipeline
├── frontend/
│   ├── web/                  # Main React dashboard (current UI)
│   ├── dashboard/            # Legacy PyQt dashboard assets
│   └── legacy_te_proj/       # Archived legacy prototype
├── scripts/                  # Train/infer/evaluate helper scripts
├── models/                   # Trained model weights (best.pt)
├── dataset/                  # YOLO train/valid/test + data.yaml
├── data/                     # Alerts, logs, runtime media
├── docs/                     # Guides, plans, integration notes
├── run_backend.ps1           # Backend launcher (foreground/background)
└── run_frontend.ps1          # Frontend launcher
```

## Quick Start

### 1) Clone the repository

```powershell
git clone https://github.com/Harshal-Bsys27/COASTVISION.git
cd COASTVISION
```

### 2) Create virtual environment

> Note: `run_backend.ps1` currently expects the environment folder name to be `venv`.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4) Start backend

```powershell
.\run_backend.ps1
```

### 5) Start frontend (new terminal)

```powershell
.\run_frontend.ps1
```

### 6) Open the app

- Frontend: http://localhost:5173
- Backend health: http://127.0.0.1:8000/api/health

## Run Modes

### Backend foreground mode

```powershell
.\run_backend.ps1
```

### Backend background mode

```powershell
.\run_backend.ps1 -Detach
```

### Stop background backend

```powershell
.\run_backend.ps1 -Stop
```

## Configuration

Environment variables commonly used for performance/device control:

| Variable | Purpose | Example |
|---------|---------|---------|
| `COASTVISION_DEVICE` | Force inference device | `cuda:0` |
| `COASTVISION_REQUIRE_CUDA` | Fail startup if CUDA is unavailable | `1` |
| `COASTVISION_HALF` | FP16 inference on supported GPU | `1` |
| `COASTVISION_TF32` | Enable TF32 on NVIDIA Ampere+ | `1` |
| `COASTVISION_CUDNN_BENCHMARK` | cuDNN autotune for speed | `1` |
| `COASTVISION_VIDEO_DIR` | Override video source directory | `C:\path\to\videos` |
| `COASTVISION_MAX_SIDE` | Resize guard for large frames | `960` or `1280` |
| `COASTVISION_IMGSZ` | YOLO inference size | `640` |
| `COASTVISION_FPS` | Processing frame rate cap | `12` |
| `COASTVISION_INFER_EVERY` | Infer every Nth frame | `2` |

Frontend API URL (optional):

- Set `VITE_API_URL` in frontend environment to point to custom backend host.
- If not set, frontend defaults to `http://127.0.0.1:8000`.

## API Reference (Key Endpoints)

### Core

- `GET /api/health`
- `GET /api/zones`
- `POST /api/zones/reload`
- `GET /api/analysis`
- `GET /api/alerts`

### Zone Streams and Detection

- `GET /api/zones/<zid>/frame.jpg`
- `GET /api/zones/<zid>/stream.mjpg`
- `GET /api/zones/<zid>/hls/stream.m3u8`
- `GET /api/zones/<zid>/detections`
- `GET /api/zones/<zid>/timeline`
- `GET|POST /api/zones/<zid>/name`

### Video Management

- `GET /api/videos`
- `POST /api/videos/upload`
- `DELETE /api/videos/<filename>`
- `POST /api/videos/<filename>/rename`

### Lifeguard Operations

- `POST /api/lifeguards/register`
- `GET /api/lifeguards`
- `GET /api/lifeguards/<lg_id>`
- `POST /api/lifeguards/<lg_id>/assign`
- `GET /api/lifeguards/<lg_id>/alerts`
- `POST /api/lifeguards/<lg_id>/respond`
- `POST /api/lifeguards/<lg_id>/heartbeat`
- `GET /api/lifeguards/<lg_id>/stream`
- `POST /api/admin/broadcast`

### Telegram Notifications

- `GET /api/telegram/status`
- `POST /api/telegram/register`
- `POST /api/telegram/unregister/<lg_id>`
- `GET /api/telegram/<lg_id>`
- `POST /api/telegram/<lg_id>/test`
- `POST /api/telegram/<lg_id>/pause`
- `POST /api/telegram/<lg_id>/resume`

## Model Training and Evaluation

### Train

```powershell
python scripts/train_yolov8.py
```

### Evaluate

```powershell
python scripts/evaluate_model.py --model models/best.pt --data dataset/data.yaml --device 0 --imgsz 640 --save-json
```

Evaluation summary file:

- `scripts/evaluation_results.md`

## Latest Model Performance

Validation dataset:

- Images: 1478
- Instances: 2748

Overall metrics:

- Precision: 0.83
- Recall: 0.819
- mAP50: 0.865
- mAP50-95: 0.53

Per-class mAP50:

- Drowning: 0.905
- Person out of water: 0.852
- Swimming: 0.837

Presentation one-liner:

> Our custom YOLOv8 model achieves **86.5% mAP50** and **53% mAP50-95** on the validation set, with best class performance on drowning detection (**90.5% mAP50**).

## Documentation

- `COASTVISION_MASTER_GUIDE.md`
- `docs/presentation_system_guide.md` (presentation/viva-ready system explanation)
- `docs/project_plan.md`
- `docs/dashboard_integration.md`
- `docs/colab_training.md`
- `docs/colab_training_full_example.md`
- `docs/colab_training_with_auto_backup.md`

## Roadmap

- Improve event-level drowning behavior modeling beyond single-frame detection.
- Add richer incident triage and priority scoring.
- Add deployment packaging for production setup.
- Strengthen test coverage for backend and dashboard API integration.

## Contributing

Contributions are welcome. If you want to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request with clear description and test notes.

## License

This project is intended for academic and research use unless otherwise specified.

If you want permissive open-source distribution, add an MIT `LICENSE` file at project root.
