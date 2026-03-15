# CoastVision Presentation System Guide

Last updated: March 2026
Audience: Project presentation, viva, technical demo

## 1) Executive Summary

CoastVision is an AI-powered coastal safety monitoring platform that analyzes multi-zone video feeds to detect drowning and emergency events in near real-time. The system combines:

- A Python Flask backend for AI inference, alert generation, and APIs
- A React dashboard frontend for operations, analytics, and control
- GPU-accelerated YOLO inference for detection quality and speed
- Multi-channel alerting: dashboard visual alerts, browser voice alerts, and Telegram lifeguard notifications

Primary outcome: reduce response time by giving lifeguards accurate, zone-specific alerts with live visual context.

## 2) Problem Statement and Solution

### Problem

Manual beach/pool monitoring is difficult at scale when multiple camera zones are active. Operators can miss critical events due to:

- Visual overload
- Delayed recognition of risky behavior
- Lack of immediate, structured notifications

### Solution

CoastVision provides:

- Continuous AI monitoring per zone
- Automatic event detection (drowning/emergency-related classes)
- Zone-wise alert routing to assigned lifeguards
- Operational dashboard for live monitoring, playback, analytics, and incident traceability

## 3) System Architecture

```text
Zone Video Files / Camera Feeds
        |
        v
Backend (Flask + YOLO + OpenCV)
  - Zone workers (thread per zone)
  - AI inference (GPU when available)
  - Alert generation and persistence
  - HLS/MJPEG/frame endpoints
  - Lifeguard + Telegram APIs
        |
        v
Frontend (React + MUI + Chart.js)
  - Live dashboard cards
  - Fullscreen zone view
  - Analytics and event history
  - Lifeguard notifications management
```

## 4) Technologies and Libraries

### Backend

- Python 3.x
- Flask
- Flask-CORS
- Waitress (production-style serving)
- OpenCV (`opencv-python`)
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- Ultralytics YOLO
- NumPy
- Requests (Telegram Bot API calls)

### Frontend

- React 18
- Vite
- Material UI (`@mui/material`, icons, emotion)
- Chart.js + `react-chartjs-2`
- `chartjs-adapter-date-fns`
- `hls.js`
- `react-zoom-pan-pinch`

## 5) Core Runtime Logic

## 5.1 Zone Discovery and Mapping

- Backend scans the configured video directory for supported video files.
- Files like `zone3.mp4` map directly to zone 3.
- Other filenames receive stable auto IDs.
- This allows dynamic add/remove of zones without hardcoding only 6 zones.

## 5.2 Inference Pipeline

Per zone worker loop:

1. Read frame from zone video/camera
2. Resize for configured performance target
3. Run YOLO inference (main model + optional person model)
4. Draw bounding boxes and labels
5. Emit detections to live endpoints
6. Create alerts if confidence and class rules are met
7. Persist alert rows/images and update in-memory history

## 5.3 Alert Generation Rules

- Main classes are interpreted for drowning/emergency semantics.
- Alert confidence threshold is stricter than display threshold.
- Cooldown is applied to avoid duplicate noisy alerts.
- Alerts are tagged with zone, label, confidence, timestamp, and optional image path.

## 5.4 Streaming Modes (Important for Q&A)

Frontend playback strategy per zone:

1. HLS (primary, actual video playback)
2. MJPEG (fallback)
3. Frame polling from `frame.jpg` (safety fallback)

So yes: the system is designed to play actual video via HLS first, with robust fallback modes to prevent blank feeds.

## 6) Lifeguard Notification System

## 6.1 Lifeguard-Zone Mapping

- Lifeguard IDs follow pattern `lifeguard_<zoneId>`.
- Example: `lifeguard_1` is mapped to Zone 1.
- Telegram notifications route zone-specifically to avoid cross-zone confusion.

## 6.2 Telegram Features Implemented

- Register chat ID per lifeguard
- Remove registration
- Test alert per lifeguard
- Stop/Resume notifications per lifeguard (without removing chat ID)
- Persist registrations in `data/telegram_users.json`

## 6.3 Stop/Resume Behavior

- Stop: pauses all Telegram signals for that lifeguard
- Resume: re-enables notifications
- Remove: unregisters chat ID completely

## 6.4 Mobile Onboarding Flow

1. Open Telegram bot and press Start
2. Get chat ID from `@userinfobot`
3. Add chat ID in corresponding lifeguard row in dashboard
4. Use Test button to verify delivery

## 7) Dashboard Modules

Main tabs and purpose:

- Dashboard: live zone grid and quick status
- Analytics: trend and summary metrics
- Event History: chronological alert logs
- Settings: backend/device and operational settings
- Lifeguards: Telegram registration, test, stop/resume, remove
- Videos: upload, rename, delete zone videos

## 8) Data and Persistence

Main persisted data:

- `data/alerts/alerts.csv`: event log records
- `data/alerts/images/`: alert snapshots
- `data/telegram_users.json`: Telegram registration + paused state
- `data/alerts/lifeguards.json`: lifeguard registry

In-memory structures are used for fast UI APIs (recent alerts, SSE/event queues, crowd status, timelines), with key data persisted to disk.

## 9) APIs (Presentation-Relevant)

Core:

- `GET /api/health`
- `GET /api/zones`
- `GET /api/alerts`
- `GET /api/analysis`

Streaming:

- `GET /api/zones/<zid>/hls/stream.m3u8`
- `GET /api/zones/<zid>/stream.mjpg`
- `GET /api/zones/<zid>/frame.jpg`

Telegram/Lifeguards:

- `GET /api/telegram/status`
- `POST /api/telegram/register`
- `POST /api/telegram/unregister/<lg_id>`
- `GET /api/telegram/<lg_id>`
- `POST /api/telegram/<lg_id>/test`
- `POST /api/telegram/<lg_id>/pause`
- `POST /api/telegram/<lg_id>/resume`

## 10) GPU and Performance Strategy

- Default target device is CUDA if available.
- Optional mixed precision (`COASTVISION_HALF`) for speed.
- Tuned settings for throughput and smoother UI playback.
- Fallback to CPU if GPU is unavailable or unsupported.

Typical performance controls:

- `COASTVISION_IMGSZ`
- `COASTVISION_FPS`
- `COASTVISION_INFER_EVERY`
- `COASTVISION_MAX_SIDE`

## 11) Security and Reliability Notes

- CORS enabled for frontend-backend local integration.
- Telegram token loaded from environment (`.env` supported).
- Fail-safe stream fallback chain (HLS -> MJPEG -> frame polling).
- Alert cooldown and filtering reduce spam.

## 12) Limitations and Future Enhancements

Current limitations:

- No full RBAC/auth layer for admin/operator roles yet.
- Rule-based alert heuristics may produce edge-case false positives.
- Local-file feed assumption unless camera ingestion is configured.

Future enhancements:

- Better behavior-level temporal models for drowning risk
- Incident severity scoring and smart triage
- Improved auth, audit logging, and deployment hardening

## 13) Presentation Q&A Cheat Sheet

Q: Is it real-time or batch?
A: Real-time zone workers process continuous streams and serve live overlays.

Q: Is it frame-by-frame snapshots only?
A: No. HLS video playback is primary; MJPEG/frame are fallbacks.

Q: How are alerts routed to lifeguards?
A: Zone-specific mapping (`lifeguard_<zoneId>`) and Telegram routing per zone.

Q: Can you temporarily stop notifications without losing setup?
A: Yes, Stop/Resume keeps chat registration and pauses only signals.

Q: What if Telegram test fails?
A: API returns reason details; common fix is starting chat with the bot on mobile.

## 14) File Map for Demo Explanation

Backend core:

- `backend/server.py`
- `backend/telegram_notify.py`

Frontend core:

- `frontend/web/src/App.jsx`

Run scripts:

- `run_backend.ps1`
- `run_frontend.ps1`

Model/data:

- `models/best.pt`
- `dataset/data.yaml`
- `data/alerts/`

## 15) One-Minute Pitch

"CoastVision is a multi-zone AI surveillance platform for water safety. It runs YOLO-based detection on live video feeds, streams annotated video to a React dashboard, logs incidents, and sends zone-specific Telegram alerts to lifeguards. It is GPU-optimized for near real-time performance, has fallback streaming for reliability, and includes operational controls like test, stop/resume, and analytics for faster emergency response."