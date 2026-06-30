"""CoastVision backend: serves annotated zone frames, alerts, and analysis."""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import sys
import platform
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    except Exception as e:
        pass  # Silently ignore .env loading errors

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request, send_from_directory
try:
    from PIL import Image
except Exception:
    Image = None
from flask_cors import CORS
from ultralytics import YOLO
from flask_socketio import SocketIO, emit

# Import Telegram notification system
from telegram_notify import notifier

# --- NEW: GPU perf toggles (safe for RTX 3050) ---
COASTVISION_TF32 = os.environ.get("COASTVISION_TF32", "1").strip().lower() not in {"0", "false", "no"}
COASTVISION_CUDNN_BENCHMARK = os.environ.get("COASTVISION_CUDNN_BENCHMARK", "1").strip().lower() not in {"0", "false", "no"}

# NEW: if set, backend will refuse to start unless CUDA is usable
COASTVISION_REQUIRE_CUDA = os.environ.get("COASTVISION_REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes", "on"}

# Disable autograd globally for inference server
try:
    torch.set_grad_enabled(False)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        if COASTVISION_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        if COASTVISION_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# NEW: CUDA smoke test (catches "CUDA available but broken driver/runtime" cases)
CUDA_SMOKE_OK: bool = False
CUDA_SMOKE_ERROR: Optional[str] = None

def _cuda_smoke_test() -> tuple[bool, Optional[str]]:
    try:
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False (CPU-only torch or no CUDA runtime)"
        # init + tiny allocation
        torch.cuda.init()
        _ = torch.empty((1,), device="cuda")
        torch.cuda.synchronize()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _torch_cuda_build_info() -> dict:
    # Torch can be installed without CUDA support (torch.version.cuda == None)
    built_cuda = getattr(torch.version, "cuda", None) is not None
    try:
        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        device_count = None

    names = []
    if torch.cuda.is_available():
        try:
            for i in range(int(device_count or 0)):
                names.append(torch.cuda.get_device_name(i))
        except Exception:
            pass

    return {
        "torch_built_with_cuda": bool(built_cuda),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": device_count,
        "torch_cuda_device_names": names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }

# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parent

_VIDEO_DIR_ENV = os.environ.get("COASTVISION_VIDEO_DIR", "").strip()
VIDEO_DIR_CANDIDATES = [
    Path(_VIDEO_DIR_ENV) if _VIDEO_DIR_ENV else None,
    (ROOT / ".." / "frontend" / "dashboard" / "videos"),
    (ROOT / ".." / "data" / "raw_videos"),
]


import re

# Supported video file extensions
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts"}

# Stable mapping: filename -> zone ID (persists across rescans so IDs don't shuffle)
_video_name_to_zid: Dict[str, int] = {}
_next_auto_zid: int = 1

# Historical person count tracking: zid -> deque of (timestamp, count) tuples
_zone_person_history: Dict[int, deque] = {}
# Custom zone names: zid -> custom name
_zone_custom_names: Dict[int, str] = {}

# Max history points to keep per zone (last 24 hours at 1 point per minute = 1440)
MAX_HISTORY_POINTS = 1440


def _find_zone_ids(video_dir: Path) -> list:
    """Return sorted list of all zone IDs for ALL video files in the video dir.
    
    - zone{N}.mp4 files get zone ID = N  (backward compatible)
    - Any other video file gets a stable auto-assigned ID
    """
    global _next_auto_zid

    # Discover all video files
    all_videos: Dict[str, Path] = {}
    if video_dir.exists():
        for p in video_dir.iterdir():
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS:
                all_videos[p.name] = p

    # First pass: assign IDs for zone{N}.* files (backward compat)
    assigned: Dict[str, int] = {}
    used_ids: set = set()
    for name in sorted(all_videos.keys()):
        m = re.match(r"zone(\d+)\.\w+$", name, re.IGNORECASE)
        if m:
            zid = int(m.group(1))
            assigned[name] = zid
            used_ids.add(zid)

    # Second pass: assign stable IDs for other video files
    for name in sorted(all_videos.keys()):
        if name in assigned:
            continue
        # Check if we already assigned an ID in a previous scan
        if name in _video_name_to_zid:
            zid = _video_name_to_zid[name]
            assigned[name] = zid
            used_ids.add(zid)
        else:
            # Find next available ID
            while _next_auto_zid in used_ids:
                _next_auto_zid += 1
            assigned[name] = _next_auto_zid
            used_ids.add(_next_auto_zid)
            _next_auto_zid += 1

    # Update stable mapping
    _video_name_to_zid.clear()
    _video_name_to_zid.update({name: zid for name, zid in assigned.items()})

    # Update _next_auto_zid to be beyond all used IDs
    if used_ids:
        _next_auto_zid = max(max(used_ids) + 1, _next_auto_zid)

    return sorted(assigned.values())


# Also maintain a reverse map: zid -> filename for display
def _zid_to_filename() -> Dict[int, str]:
    return {zid: name for name, zid in _video_name_to_zid.items()}


def _record_person_count(zid: int, count: int):
    """Record person count for a zone at current timestamp.
    
    Records on every call but throttles to prevent flooding:
    - Always record if count changed from last value
    - Record at least every 10 seconds even if unchanged (smooth chart)
    - Collapse consecutive identical values older than 10s
    """
    timestamp = time.time()
    if zid not in _zone_person_history:
        _zone_person_history[zid] = deque(maxlen=MAX_HISTORY_POINTS)
    
    history = _zone_person_history[zid]
    if not history:
        history.append((timestamp, count))
        # Check crowd density
        _check_crowd_density(zid, count)
        return
    
    last_ts, last_count = history[-1]
    elapsed = timestamp - last_ts
    
    if count != last_count:
        # Count changed: always record immediately
        history.append((timestamp, count))
        # Check crowd density
        _check_crowd_density(zid, count)
    elif elapsed >= 10:
        # Same count but 10s passed: record a fresh data point for smooth lines
        history.append((timestamp, count))
        # Check crowd density periodically
        _check_crowd_density(zid, count)


def _check_crowd_density(zid: int, person_count: int):
    """Check if zone exceeds crowd density threshold and generate alert if needed."""
    threshold = CROWD_THRESHOLDS.get(zid, 50)
    now = time.time()
    
    # Update crowd status
    with _crowd_lock:
        CROWD_STATUS[zid] = {
            "count": person_count,
            "threshold": threshold,
            "status": "crowded" if person_count > threshold else "normal",
            "last_check": now,
            "exceeded": person_count > threshold
        }
        
        # Check if we should trigger a crowding alert
        if person_count > threshold:
            # Check cooldown to avoid spam
            last_alert_time = CROWD_ALERT_LAST_TIME.get(zid, 0)
            if now - last_alert_time >= CROWD_ALERT_COOLDOWN:
                # Generate crowd alert
                severity = "low" if person_count <= threshold * 1.2 else "medium" if person_count <= threshold * 1.5 else "high"
                alert = {
                    "zone": zid,
                    "person_count": person_count,
                    "threshold": threshold,
                    "severity": severity,
                    "timestamp": now,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "label": "Crowd Density"
                }
                
                CROWD_ALERT_HISTORY.appendleft(alert)
                CROWD_ALERT_LAST_TIME[zid] = now
                
                # Log to CSV
                try:
                    with open(CROWD_ALERTS_CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now(timezone.utc).isoformat(),
                            zid,
                            person_count,
                            threshold,
                            severity
                        ])
                except Exception as e:
                    print(f"[crowd] Error logging crowd alert: {e}")
                
                print(f"[crowd] Zone {zid} CROWDING ALERT! {person_count} people (threshold: {threshold}, severity: {severity})")

                # Send Telegram crowd alert only to lifeguards mapped to this
                # zone (lifeguard_<zoneId>), and keep the wording consistent
                # with drowning/emergency alerts ("Zone X"), to avoid random
                # location names like "South Beach" showing up on unrelated
                # lifeguards.
                try:
                    zone_name = f"Zone {zid}"
                    users = getattr(notifier, "users", {}) or {}
                    for tg_lg_id in list(users.keys()):
                        target_zone = None
                        if isinstance(tg_lg_id, str) and tg_lg_id.startswith("lifeguard_"):
                            try:
                                target_zone = int(tg_lg_id.split("_", 1)[1])
                            except Exception:
                                target_zone = None

                        if target_zone is not None and target_zone == zid:
                            # Treat crowd density like a special detection type
                            # so the message stays compact, eg:
                            # "⚠️ Crowd density detected in Zone 4 (150.0%)".
                            try:
                                # Map person_count vs threshold into a % style
                                # "confidence" for the existing send_alert API.
                                if threshold > 0:
                                    crowd_ratio = min(1.5, person_count / float(threshold))
                                    crowd_percent = crowd_ratio * 100.0
                                else:
                                    crowd_percent = 100.0
                            except Exception:
                                crowd_percent = 100.0

                            try:
                                notifier.send_alert(tg_lg_id, zone_name, "Crowd density", crowd_percent)
                            except Exception as e:
                                print(f"[telegram] Error sending crowd alert to {tg_lg_id}: {e}")
                except Exception as e:
                    print(f"[telegram] Crowd routing error: {e}")


def _get_zone_display_name(zid: int) -> str:
    """Get display name for zone (custom name or default)."""
    if zid in _zone_custom_names:
        return _zone_custom_names[zid]
    return f"Zone {zid}"


def _set_zone_name(zid: int, name: str) -> bool:
    """Set custom name for a zone. Returns True if successful."""
    if not name or not name.strip():
        # Remove custom name if empty
        _zone_custom_names.pop(zid, None)
        return True
    _zone_custom_names[zid] = name.strip()
    return True


ZONE_IDS = _find_zone_ids(
    Path(_VIDEO_DIR_ENV) if _VIDEO_DIR_ENV else (ROOT / ".." / "frontend" / "dashboard" / "videos")
)

# Detection
CONF_THRES = float(os.environ.get("COASTVISION_CONF", "0.35"))
PERSON_CONF_THRES = float(os.environ.get("COASTVISION_PERSON_CONF", "0.25"))
COASTVISION_IOU = float(os.environ.get("COASTVISION_IOU", "0.45"))
COASTVISION_MAX_DET = int(os.environ.get("COASTVISION_MAX_DET", "200"))
# Alerts are stricter than overlays (helps precision for drowning/emergency)
COASTVISION_ALERT_CONF = float(os.environ.get("COASTVISION_ALERT_CONF", "0.55"))

# Performance (optimized for smooth playback with RTX 3050)
COASTVISION_MAX_SIDE = int(os.environ.get("COASTVISION_MAX_SIDE", "1280"))
COASTVISION_FPS = int(os.environ.get("COASTVISION_FPS", "24"))  # Smooth 24fps playback
COASTVISION_INFER_EVERY = int(os.environ.get("COASTVISION_INFER_EVERY", "2"))  # Detect every 2 frames for smoother boxes
COASTVISION_IMGSZ = int(os.environ.get("COASTVISION_IMGSZ", "640"))  # Optimal for YOLOv8

# Grid playback: serve a smaller cached JPEG to reduce bandwidth and stutter
COASTVISION_GRID_MAX_W = int(os.environ.get("COASTVISION_GRID_MAX_W", "640"))
COASTVISION_GRID_JPEG_QUALITY = int(os.environ.get("COASTVISION_GRID_JPEG_QUALITY", "80"))  # Higher quality for crisp boxes

COASTVISION_DEVICE = os.environ.get("COASTVISION_DEVICE", "").strip()
COASTVISION_ALERT_COOLDOWN_S = float(os.environ.get("COASTVISION_ALERT_COOLDOWN_S", "4"))
COASTVISION_DET_HOLD_S = float(os.environ.get("COASTVISION_DET_HOLD_S", "1.5"))  # Hold detections longer (1.5s) to prevent flickering
COASTVISION_OVERLAY_STYLE = os.environ.get("COASTVISION_OVERLAY_STYLE", "pro").strip().lower()

# HLS Streaming (hardware-accelerated via FFmpeg + NVENC)
COASTVISION_HLS_ENABLED = os.environ.get("COASTVISION_HLS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
COASTVISION_HLS_SEGMENT_S = float(os.environ.get("COASTVISION_HLS_SEGMENT_S", "1"))  # short segments for low latency
COASTVISION_HLS_LIST_SIZE = int(os.environ.get("COASTVISION_HLS_LIST_SIZE", "4"))  # sliding window of segments in playlist
COASTVISION_HLS_BITRATE = os.environ.get("COASTVISION_HLS_BITRATE", "2M").strip()  # target bitrate
COASTVISION_HLS_PRESET = os.environ.get("COASTVISION_HLS_PRESET", "p4").strip()  # NVENC preset (p1-p7, p4=balanced)

COASTVISION_ENABLE_PERSON_DET = os.environ.get("COASTVISION_ENABLE_PERSON_DET", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

_ALERT_CLASSES_ENV = os.environ.get("COASTVISION_ALERT_CLASSES", "").strip()
ALERT_CLASSES = {s.strip().lower() for s in _ALERT_CLASSES_ENV.split(",") if s.strip()} if _ALERT_CLASSES_ENV else None

ALERT_HISTORY = deque(maxlen=400)

ALERTS_DIR = (ROOT / ".." / "data" / "alerts").resolve()
ALERTS_IMAGES_DIR = (ALERTS_DIR / "images").resolve()
ALERTS_CSV_PATH = (ALERTS_DIR / "alerts.csv").resolve()
_alerts_lock = threading.Lock()

# Response time tracking
RESPONSE_TIMES_CSV_PATH = (ALERTS_DIR / "response_times.csv").resolve()
_response_lock = threading.Lock()

# Create response times CSV headers if it doesn't exist
def _init_response_times_csv():
    if not RESPONSE_TIMES_CSV_PATH.exists():
        try:
            with open(RESPONSE_TIMES_CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "zone", "lifeguard_id", "lifeguard_name", "response_time_seconds", "alert_sent_at", "responded_at"])
        except Exception as e:
            print(f"[response] Error creating response times CSV: {e}")

# Call on startup
_init_response_times_csv()

# In-memory tracking of sent alerts with timestamps (for response time calculation)
ALERT_SENT_TIMES: Dict[str, float] = {}  # alert_id -> timestamp when sent to lifeguards

# ===================== CROWD DENSITY MANAGEMENT =====================
CROWD_THRESHOLDS_FILE = (ALERTS_DIR / "crowd_thresholds.json").resolve()
CROWD_ALERTS_CSV_PATH = (ALERTS_DIR / "crowd_alerts.csv").resolve()

# Default crowd thresholds (people per zone)
CROWD_THRESHOLDS: Dict[int, int] = {}  # zone_id -> threshold person count

# Track last crowd alert time per zone (to avoid spam)
CROWD_ALERT_LAST_TIME: Dict[int, float] = {}  # zone_id -> last alert timestamp
CROWD_ALERT_COOLDOWN = 60  # Minimum seconds between crowd alerts for same zone

# Load/save crowd thresholds
def _load_crowd_thresholds():
    """Load crowd thresholds from file."""
    global CROWD_THRESHOLDS
    if CROWD_THRESHOLDS_FILE.exists():
        try:
            with open(CROWD_THRESHOLDS_FILE, "r") as f:
                raw = json.load(f)
                normalized = {}
                for key, value in (raw or {}).items():
                    try:
                        zid = int(key)
                    except Exception:
                        continue
                    try:
                        normalized[zid] = int(value)
                    except Exception:
                        normalized[zid] = 50
                CROWD_THRESHOLDS = normalized
            print(f"[crowd] Loaded thresholds: {CROWD_THRESHOLDS}")
        except Exception as e:
            print(f"[crowd] Error loading thresholds: {e}")
            CROWD_THRESHOLDS = {}

def _save_crowd_thresholds():
    """Persist crowd thresholds to file."""
    try:
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CROWD_THRESHOLDS_FILE, "w") as f:
            json.dump(CROWD_THRESHOLDS, f, indent=2)
    except Exception as e:
        print(f"[crowd] Error saving thresholds: {e}")

def _init_crowd_thresholds():
    """Initialize crowd thresholds with defaults if needed."""
    global CROWD_THRESHOLDS
    _load_crowd_thresholds()
    # Ensure all zones have thresholds
    for zid in ZONE_IDS:
        if zid not in CROWD_THRESHOLDS:
            CROWD_THRESHOLDS[zid] = 50  # Default threshold: 50 people
    _save_crowd_thresholds()

# Create crowd alerts CSV headers if not exists
def _init_crowd_alerts_csv():
    if not CROWD_ALERTS_CSV_PATH.exists():
        try:
            with open(CROWD_ALERTS_CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "zone", "person_count", "threshold", "severity"])
        except Exception as e:
            print(f"[crowd] Error creating crowd alerts CSV: {e}")

_init_crowd_alerts_csv()
_init_crowd_thresholds()

# Crowd alert history in memory (for dashboard)
CROWD_ALERT_HISTORY = deque(maxlen=200)  # Store recent crowd alerts

# Current crowd status per zone
CROWD_STATUS: Dict[int, Dict[str, Any]] = {}  # zone_id -> {count, threshold, status, last_check}
_crowd_lock = threading.Lock()

# ----------------- LIFEGUARD MANAGEMENT -----------------
import json
import uuid
import queue

# In-memory lifeguard registry (for demo; use DB in production)
LIFEGUARDS: Dict[str, Dict[str, Any]] = {}  # id -> {id, name, phone, zones: [], online: bool, last_seen}
LIFEGUARD_SESSIONS: Dict[str, str] = {}  # session_token -> lifeguard_id
LIFEGUARD_ALERTS: Dict[str, deque] = {}  # lifeguard_id -> deque of alerts
LIFEGUARD_SSE_QUEUES: Dict[str, queue.Queue] = {}  # lifeguard_id -> SSE queue for real-time push
_lifeguard_lock = threading.Lock()

# File to persist lifeguards
LIFEGUARDS_FILE = (ALERTS_DIR / "lifeguards.json").resolve()

def _load_lifeguards():
    """Load lifeguards from file on startup."""
    global LIFEGUARDS
    if LIFEGUARDS_FILE.exists():
        try:
            with open(LIFEGUARDS_FILE, "r") as f:
                LIFEGUARDS = json.load(f)
            print(f"[lifeguard] Loaded {len(LIFEGUARDS)} lifeguards from {LIFEGUARDS_FILE}")
        except Exception as e:
            print(f"[lifeguard] Error loading lifeguards: {e}")
            LIFEGUARDS = {}

def _save_lifeguards():
    """Persist lifeguards to file."""
    try:
        LIFEGUARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LIFEGUARDS_FILE, "w") as f:
            json.dump(LIFEGUARDS, f, indent=2)
    except Exception as e:
        print(f"[lifeguard] Error saving lifeguards: {e}")

_load_lifeguards()

def _broadcast_alert_to_lifeguards(alert: dict):
    """Send alert to all lifeguards assigned to the zone."""
    zone = alert.get("zone")
    
    # Create unique alert ID if not present
    if "alert_id" not in alert:
        alert["alert_id"] = f"{zone}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    # Track when alert is sent (for response time calculation)
    alert_id = alert["alert_id"]
    ALERT_SENT_TIMES[alert_id] = time.time()
    
    with _lifeguard_lock:
        for lg_id, lg in LIFEGUARDS.items():
            # Send to lifeguards assigned to this zone OR all zones (empty list = all)
            assigned = lg.get("zones", [])
            if not assigned or zone in assigned:
                # Add to their alert queue
                if lg_id not in LIFEGUARD_ALERTS:
                    LIFEGUARD_ALERTS[lg_id] = deque(maxlen=100)
                LIFEGUARD_ALERTS[lg_id].appendleft(alert)
                # Push to SSE if connected
                if lg_id in LIFEGUARD_SSE_QUEUES:
                    try:
                        LIFEGUARD_SSE_QUEUES[lg_id].put_nowait(alert)
                    except:
                        pass

    # Telegram routing: use registered Telegram users whose IDs follow the
    # pattern "lifeguard_<zoneId>" so that each lifeguard only receives
    # alerts for their own zone.
    try:
        detection_type = alert.get("label", alert.get("class", "Detection"))
        confidence = alert.get("conf", 0)
        zone_name = f"Zone {zone}" if isinstance(zone, int) else str(zone)

        users = getattr(notifier, "users", {}) or {}
        for tg_lg_id in list(users.keys()):
            target_zone = None
            if isinstance(tg_lg_id, str) and tg_lg_id.startswith("lifeguard_"):
                try:
                    target_zone = int(tg_lg_id.split("_", 1)[1])
                except Exception:
                    target_zone = None

            # Only send if this lifeguard is mapped to the alert's zone
            if target_zone is not None and target_zone == zone:
                try:
                    notifier.send_alert(tg_lg_id, zone_name, detection_type, confidence)
                except Exception as e:
                    print(f"[telegram] Error sending alert to {tg_lg_id}: {e}")
    except Exception as e:
        print(f"[telegram] Routing error: {e}")
    
    # NEW: Broadcast to WebSocket clients (dashboard + mobile app) for real-time sync
    try:
        # socketio.emit broadcasts to all clients by default when called outside a handler
        socketio.emit('new_alert', {
            'alert_id': alert_id,
            'zone': zone,
            'type': detection_type,
            'confidence': float(confidence),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'label': detection_type,
        })
        print(f"[WS] Alert broadcast: {alert_id} ({detection_type})")
    except Exception as e:
        print(f"[WS] Broadcast error: {e}")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in {"0", "false", "no", "off"}


COASTVISION_HALF = _env_bool("COASTVISION_HALF", True)

# One GPU -> avoid concurrent predict() across threads
_INFER_LOCK = threading.Lock()


# ----------------- MODEL -----------------

def _pick_model_path() -> Path:
    candidates = [
        ROOT / ".." / "models" / "best.pt",
        ROOT / "best.pt",
        ROOT / ".." / "best.pt",
        ROOT / ".." / "yolov8n.pt",
        ROOT / ".." / "yolo11n.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise RuntimeError("No model weights found (expected best.pt or yolov8n.pt).")


def _pick_person_model_path() -> Optional[Path]:
    candidates = [
        ROOT / ".." / "yolov8n.pt",
        ROOT / ".." / "yolo11n.pt",
        ROOT / "yolov8n.pt",
        ROOT / "yolo11n.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


MODEL_PATH = _pick_model_path()

_cuda_available = bool(torch.cuda.is_available())

REQUESTED_DEVICE = COASTVISION_DEVICE or ("cuda:0" if _cuda_available else "cpu")

# Decide effective device + run smoke test before loading YOLO
if str(REQUESTED_DEVICE).startswith("cuda"):
    ok, err = _cuda_smoke_test()
    CUDA_SMOKE_OK, CUDA_SMOKE_ERROR = ok, err
    if not ok:
        build_info = _torch_cuda_build_info()
        msg = (
            "[init][GPU] Requested CUDA but CUDA smoke test failed.\n"
            f"[init][GPU] torch_built_with_cuda={build_info.get('torch_built_with_cuda')} "
            f"torch_version={getattr(torch,'__version__','?')} torch_cuda_version={build_info.get('torch_cuda_version')}\n"
            f"[init][GPU] smoke_error={err}\n"
            "[init][GPU] Fix: install CUDA-enabled torch in THIS python/venv (python -m pip ... cu121), "
            "and ensure NVIDIA driver works (nvidia-smi)."
        )
        if COASTVISION_REQUIRE_CUDA:
            raise RuntimeError(msg)
        print(msg + " Falling back to CPU.")
        DEVICE = "cpu"
    else:
        DEVICE = REQUESTED_DEVICE
else:
    DEVICE = REQUESTED_DEVICE

# Ultralytics expects device=0 for cuda:0
PREDICT_DEVICE = 0 if (str(DEVICE).startswith("cuda") and bool(torch.cuda.is_available())) else "cpu"

print(f"[init] Requested device={REQUESTED_DEVICE} | Effective device={DEVICE} | predict_device={PREDICT_DEVICE}")
print(f"[init] Loading model from {MODEL_PATH} on {DEVICE}")
MODEL = YOLO(str(MODEL_PATH)).to(DEVICE)
try:
    MODEL.fuse()
except Exception:
    pass

# Optional COCO person detector if main model lacks 'person'
PERSON_MODEL = None
try:
    main_names = MODEL.names
    has_person = False
    if isinstance(main_names, dict):
        has_person = any(str(v).lower() == "person" for v in main_names.values())
    elif isinstance(main_names, list):
        has_person = any(str(v).lower() == "person" for v in main_names)

    if COASTVISION_ENABLE_PERSON_DET and (not has_person):
        person_path = _pick_person_model_path()
        if person_path:
            print(f"[init] Main model has no 'person' class; loading person model {person_path} on {DEVICE}")
            PERSON_MODEL = YOLO(str(person_path)).to(DEVICE)
            try:
                PERSON_MODEL.fuse()
            except Exception:
                pass
        else:
            print("[warn] COASTVISION_ENABLE_PERSON_DET=1 but no yolov8n.pt/yolo11n.pt found for person detection")
except Exception as e:
    print(f"[warn] Person model init failed: {type(e).__name__}: {e}")


def _names_to_list(names_obj) -> List[str]:
    """Return a stable list of class names for debugging/health."""
    try:
        if isinstance(names_obj, dict):
            out: List[str] = []
            for k in sorted(names_obj.keys()):
                out.append(str(names_obj[k]))
            return out
        if isinstance(names_obj, list):
            return [str(x) for x in names_obj]
    except Exception:
        pass
    return []


def _cls_to_label(names_obj, cls: int) -> str:
    """Map a class id to a readable label for both dict and list name formats."""
    if isinstance(names_obj, dict):
        return str(names_obj.get(cls, f"class_{cls}"))
    if isinstance(names_obj, list):
        if 0 <= cls < len(names_obj):
            return str(names_obj[cls])
        return f"class_{cls}"
    return f"class_{cls}"


# ----------------- HELPERS -----------------

def _placeholder_jpeg(title: str, subtitle: str = "") -> bytes:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, title, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (46, 230, 255), 3, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 190, 220), 2, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return jpg.tobytes() if ok else b""


def _find_video_dir() -> Path:
    for candidate in VIDEO_DIR_CANDIDATES:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / ".." / "frontend" / "dashboard" / "videos").resolve()


VIDEO_DIR = _find_video_dir()


def _open_capture(path: Path) -> Optional[cv2.VideoCapture]:
    # Prefer FFMPEG on Windows for MP4 stability
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        # reduce decode buffering / latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _resize_for_speed(frame):
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= COASTVISION_MAX_SIDE:
        return frame
    scale = COASTVISION_MAX_SIDE / float(m)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _ensure_alert_dirs():
    ALERTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not ALERTS_CSV_PATH.exists():
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        with ALERTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "ts_utc", "zone", "label", "conf", "x1", "y1", "x2", "y2", "image_path"])


def _append_alert_row(row):
    with _alerts_lock:
        _ensure_alert_dirs()
        with ALERTS_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row)


def _draw_detections(frame, dets: List[Dict[str, Any]]):
    if not dets:
        return

    h, w = frame.shape[:2]
    base = max(1, min(h, w))
    
    # THICK, highly visible boxes - easy to see in dashboard grid
    # 4-6px thick for good visibility at all scales
    thickness = max(4, min(6, int(round(base / 150))))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # LARGE font - must be readable from dashboard grid view
    font_scale = max(0.8, min(1.2, (base / 720) * 1.0))
    text_th = max(2, min(3, thickness // 2))  # Bold text

    for d in dets:
        x1, y1, x2, y2 = d.get("bbox") or [0, 0, 0, 0]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        color = tuple(d.get("color") or (0, 220, 0))
        label = str(d.get("label") or "")
        conf = float(d.get("conf") or 0.0)

        # Draw THICK bounding box - black outline + colored main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)  # Black outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # Colored box

        # Format label text - decimal confidence format (e.g., 0.85)
        text = f"{label} {conf:.2f}"
        if not text.strip():
            continue

        (tw, th_text), baseline = cv2.getTextSize(text, font, font_scale, text_th)
        pad_x = 10
        pad_y = 8
        label_h = th_text + pad_y * 2
        label_w = tw + pad_x * 2
        
        # Smart label positioning - prefer above box, but go inside if no room
        if y1 - label_h >= 0:
            y_text_top = y1 - label_h
            y_text_bottom = y1
            x_text_left = x1
        else:
            y_text_top = y1
            y_text_bottom = y1 + label_h
            x_text_left = x1
        
        # Ensure label doesn't go off edges
        if x_text_left + label_w > w:
            x_text_left = max(0, w - label_w)
        x_text_right = min(w, x_text_left + label_w)

        # WHITE/LIGHT background for label - highly readable
        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), (255, 255, 255), -1)
        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), color, 3)  # Colored border
        
        # DARK text on light background - maximum readability
        text_x = x_text_left + pad_x
        text_y = y_text_bottom - pad_y
        # Shadow for extra clarity
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, font_scale, (150, 150, 150), text_th, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)


# ----------------- STATE -----------------


@dataclass
class ZoneState:
    zid: int
    path: Path
    cap: cv2.VideoCapture
    lock: threading.Lock
    last_jpeg: Optional[bytes] = None
    last_jpeg_grid: Optional[bytes] = None
    last_ts: float = 0.0
    last_error: Optional[str] = None
    frame_i: int = 0
    last_alert_time_s: float = 0.0
    last_dets: Optional[List[Dict[str, Any]]] = None
    last_dets_ts: float = 0.0
    # HLS streaming state
    hls_proc: Optional[Any] = None  # FFmpeg subprocess
    hls_dir: Optional[Path] = None  # Temp dir for .ts segments + .m3u8
    hls_frame_w: int = 0
    hls_frame_h: int = 0
    hls_ok: bool = False


# ----- HLS Infrastructure -----

_HLS_BASE_DIR: Optional[Path] = None
_FFMPEG_PATH: Optional[str] = None
_HLS_USE_NVENC: bool = False


def _find_ffmpeg() -> Optional[str]:
    """Locate FFmpeg binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Common Windows install locations
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"),
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _check_nvenc(ffmpeg: str) -> bool:
    """Test if FFmpeg can actually use h264_nvenc (not just listed — needs driver support)."""
    try:
        # Actually try to initialize the encoder — checking -encoders is not enough
        # because the binary may list nvenc even if the driver is too old.
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _init_hls():
    """Initialize HLS temp directory and probe FFmpeg capabilities."""
    global _HLS_BASE_DIR, _FFMPEG_PATH, _HLS_USE_NVENC

    _FFMPEG_PATH = _find_ffmpeg()
    if not _FFMPEG_PATH:
        print("[HLS] FFmpeg not found — HLS streaming disabled")
        return

    _HLS_USE_NVENC = _check_nvenc(_FFMPEG_PATH)
    print(f"[HLS] FFmpeg: {_FFMPEG_PATH}")
    print(f"[HLS] NVENC available: {_HLS_USE_NVENC}")

    _HLS_BASE_DIR = Path(tempfile.mkdtemp(prefix="coastvision_hls_"))
    print(f"[HLS] Segment directory: {_HLS_BASE_DIR}")


def _start_hls_encoder(st: ZoneState, width: int, height: int, fps: int):
    """Launch FFmpeg process to encode raw BGR frames into HLS segments."""
    if not _FFMPEG_PATH or not _HLS_BASE_DIR or not COASTVISION_HLS_ENABLED:
        return

    # Rate-limit restarts: don't restart within 3 seconds of last attempt
    now = time.time()
    last_attempt = getattr(st, '_hls_last_start', 0.0)
    if now - last_attempt < 3.0:
        return
    st._hls_last_start = now

    # Track consecutive failures to fall back from NVENC to libx264
    fail_count = getattr(st, '_hls_nvenc_fails', 0)

    # Create zone-specific dir
    zdir = _HLS_BASE_DIR / f"zone{st.zid}"
    zdir.mkdir(parents=True, exist_ok=True)
    st.hls_dir = zdir
    st.hls_frame_w = width
    st.hls_frame_h = height

    playlist = str(zdir / "stream.m3u8")
    segment_pattern = str(zdir / "seg%05d.ts")

    # Determine encoder: use NVENC unless we've had repeated failures (session limit)
    use_nvenc = _HLS_USE_NVENC and fail_count < 2

    # Build encoder args
    if use_nvenc:
        enc_args = [
            "-c:v", "h264_nvenc",
            "-preset", COASTVISION_HLS_PRESET,
            "-tune", "ll",
            "-rc", "cbr",
            "-b:v", COASTVISION_HLS_BITRATE,
            "-maxrate", COASTVISION_HLS_BITRATE,
            "-bufsize", "1M",
            "-profile:v", "main",
            "-g", str(fps),
            "-sc_threshold", "0",
        ]
    else:
        enc_args = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", COASTVISION_HLS_BITRATE,
            "-maxrate", COASTVISION_HLS_BITRATE,
            "-bufsize", "1M",
            "-profile:v", "main",
            "-g", str(fps),
            "-sc_threshold", "0",
        ]

    cmd = [
        _FFMPEG_PATH,
        "-hide_banner", "-loglevel", "error",
        # Input: raw BGR frames from stdin pipe
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        # Convert to yuv420p (required for H.264 main/high profile)
        "-pix_fmt", "yuv420p",
        # Encoder
        *enc_args,
        # HLS output
        "-f", "hls",
        "-hls_time", str(COASTVISION_HLS_SEGMENT_S),
        "-hls_list_size", str(COASTVISION_HLS_LIST_SIZE),
        "-hls_flags", "delete_segments+append_list+independent_segments",
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", segment_pattern,
        "-y",
        playlist,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=width * height * 3 * 2,
        )
        st.hls_proc = proc
        st.hls_ok = True
        encoder_name = 'NVENC' if use_nvenc else 'libx264'
        print(f"[HLS] Started encoder for zone {st.zid} ({width}x{height} @ {fps}fps, {encoder_name})")
    except Exception as e:
        print(f"[HLS] Failed to start encoder for zone {st.zid}: {e}")
        st.hls_ok = False


def _feed_hls_frame(st: ZoneState, frame):
    """Write a raw BGR frame to the FFmpeg stdin pipe."""
    if not st.hls_proc or not st.hls_ok:
        return
    try:
        h, w = frame.shape[:2]
        # If frame dimensions changed, restart encoder
        if w != st.hls_frame_w or h != st.hls_frame_h:
            _stop_hls_encoder(st)
            _start_hls_encoder(st, w, h, COASTVISION_FPS)
        if st.hls_proc and st.hls_proc.stdin:
            st.hls_proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError):
        # Read FFmpeg stderr for diagnostics
        stderr_text = ""
        try:
            if st.hls_proc and st.hls_proc.stderr:
                stderr_text = st.hls_proc.stderr.read(4096).decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        fail_count = getattr(st, '_hls_nvenc_fails', 0) + 1
        st._hls_nvenc_fails = fail_count
        st.hls_ok = False
        msg = f"[HLS] Encoder pipe broken for zone {st.zid} (fail #{fail_count})"
        if stderr_text:
            msg += f" — FFmpeg: {stderr_text[:200]}"
        print(msg)
    except Exception as e:
        st.hls_ok = False
        print(f"[HLS] Feed error zone {st.zid}: {e}")


def _stop_hls_encoder(st: ZoneState):
    """Gracefully stop FFmpeg process for a zone."""
    proc = st.hls_proc
    st.hls_proc = None
    st.hls_ok = False
    if proc:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _cleanup_hls():
    """Stop all encoders and remove temp dir on shutdown."""
    for st in _zones.values():
        _stop_hls_encoder(st)
    if _HLS_BASE_DIR and _HLS_BASE_DIR.exists():
        try:
            shutil.rmtree(_HLS_BASE_DIR, ignore_errors=True)
        except Exception:
            pass

import atexit
atexit.register(_cleanup_hls)


VIDEO_PATHS: Dict[int, Path] = {}
_zones: Dict[int, ZoneState] = {}
_zone_threads: Dict[int, threading.Thread] = {}



def _open_zone_caps():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    global ZONE_IDS
    ZONE_IDS = _find_zone_ids(VIDEO_DIR)
    name_map = _zid_to_filename()
    for zid in ZONE_IDS:
        filename = name_map.get(zid, f"zone{zid}.mp4")
        p = VIDEO_DIR / filename
        VIDEO_PATHS[zid] = p
        if not p.exists():
            print(f"[warn] Missing video for zone {zid}: {p}")
            continue
        # Keep existing zone if it is already active.
        if zid in _zones:
            continue
        cap = _open_capture(p)
        if not cap:
            print(f"[warn] Could not open video for zone {zid}: {p}")
            continue
        _zones[zid] = ZoneState(zid=zid, path=p, cap=cap, lock=threading.Lock())


def _ensure_zone_thread(zid: int):
    if zid in _zone_threads:
        return
    st = _zones.get(zid)
    if not st:
        return
    th = threading.Thread(target=_zone_worker, args=(zid,), daemon=True)
    th.start()
    _zone_threads[zid] = th
    print(f"[zones] Started worker for zone {zid}")


def _record_alerts(alerts):
    for a in alerts:
        ALERT_HISTORY.appendleft(a)
        # Broadcast to lifeguards in real-time
        _broadcast_alert_to_lifeguards(a)


def _persist_alerts(zid: int, alerts, annotated_bgr_frame):
    if not alerts:
        return
    st = _zones.get(zid)
    if not st:
        return

    now_s = time.time()
    if COASTVISION_ALERT_COOLDOWN_S > 0 and (now_s - st.last_alert_time_s) < COASTVISION_ALERT_COOLDOWN_S:
        return

    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
    image_name = f"zone{zid}_{ts_str}.jpg"
    image_path = (ALERTS_IMAGES_DIR / image_name).resolve()
    try:
        _ensure_alert_dirs()
        cv2.imwrite(str(image_path), annotated_bgr_frame)
    except Exception:
        image_path = None

    for a in alerts:
        x1, y1, x2, y2 = (a.get("bbox") or [None, None, None, None])
        event_id = f"{ts_str}_z{zid}_{a.get('label','')}_{int((a.get('conf') or 0) * 1000)}"
        _append_alert_row(
            [
                event_id,
                a.get("ts"),
                zid,
                a.get("label"),
                a.get("conf"),
                x1,
                y1,
                x2,
                y2,
                str(image_path) if image_path else "",
            ]
        )
        a["event_id"] = event_id
        if image_path:
            a["image_path"] = str(image_path)

    st.last_alert_time_s = now_s


def _annotate(frame, zid: int):
    # IMPORTANT: serialize GPU inference across threads
    with _INFER_LOCK:
        results = MODEL.predict(
            frame,
            verbose=False,
            conf=CONF_THRES,
            iou=COASTVISION_IOU,
            max_det=COASTVISION_MAX_DET,
            device=PREDICT_DEVICE,
            imgsz=COASTVISION_IMGSZ,
            half=(COASTVISION_HALF and str(DEVICE).startswith("cuda")),
        )

    alerts = []
    dets: List[Dict[str, Any]] = []
    names = MODEL.names
    h, w = frame.shape[:2]

    # 1) Person detections (if enabled) to ensure every person gets its own box
    if PERSON_MODEL is not None:
        try:
            with _INFER_LOCK:
                pres = PERSON_MODEL.predict(
                    frame,
                    verbose=False,
                    conf=PERSON_CONF_THRES,
                    iou=COASTVISION_IOU,
                    max_det=COASTVISION_MAX_DET,
                    device=PREDICT_DEVICE,
                    imgsz=COASTVISION_IMGSZ,
                    classes=[0],
                    half=(COASTVISION_HALF and str(DEVICE).startswith("cuda")),
                )
            for pr in pres:
                for box in pr.boxes:
                    pconf = float(box.conf[0]) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({"bbox": [x1, y1, x2, y2], "label": "person", "conf": pconf, "color": (0, 220, 0)})
        except Exception:
            pass

    # 2) Main model detections (drowning/emergency etc)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            label = _cls_to_label(names, cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            label_l = str(label).lower()
            if COASTVISION_OVERLAY_STYLE in {"green", "pro"}:
                color = (0, 220, 0)
                if "drown" in label_l or "emerg" in label_l:
                    color = (0, 165, 255)
            else:
                color = (0, 220, 0)

            det = {"bbox": [x1, y1, x2, y2], "label": label, "conf": conf, "color": color}
            dets.append(det)

            # Stricter rule for alerts (precision): only high-confidence detections become events.
            if conf >= COASTVISION_ALERT_CONF and ((ALERT_CLASSES is None) or (str(label).lower() in ALERT_CLASSES)):
                alerts.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "zone": zid,
                        "label": label,
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                        "msg": f"{label} detected",
                    }
                )

    _draw_detections(frame, dets)
    return frame, alerts, dets


def _zone_worker(zid: int, fps: int = COASTVISION_FPS):
    interval = 1.0 / max(1, fps)
    st = _zones[zid]

    while True:
        # If the zone was removed (video deleted), stop this worker.
        if zid not in _zones:
            break
        t0 = time.time()
        try:
            with st.lock:
                ok, frame = st.cap.read()
                if not ok:
                    st.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = st.cap.read()

                if ok and frame is not None:
                    st.frame_i += 1
                    frame = _resize_for_speed(frame)

                    did_infer = COASTVISION_INFER_EVERY <= 1 or (st.frame_i % COASTVISION_INFER_EVERY) == 0
                    if did_infer:
                        frame, alerts, dets = _annotate(frame, zid)
                        now_det = time.time()
                        # Always record person count (including 0) for accurate timeline
                        person_count = len([d for d in dets if str(d.get('label', '')).lower() == 'person'])
                        _record_person_count(zid, person_count)
                        if dets:
                            st.last_dets = dets
                            st.last_dets_ts = now_det
                        elif st.last_dets:
                            # No detections this frame: clear stale dets after hold period
                            hold_time = now_det - (st.last_dets_ts or 0.0)
                            if hold_time > COASTVISION_DET_HOLD_S * 2:
                                st.last_dets = []
                                st.last_dets_ts = now_det
                        if alerts:
                            _record_alerts(alerts)
                            _persist_alerts(zid, alerts, frame)
                    else:
                        # ALWAYS draw cached detections on non-inference frames for stable boxes
                        if st.last_dets:
                            hold_time = time.time() - (st.last_dets_ts or 0.0)
                            if hold_time <= COASTVISION_DET_HOLD_S:
                                _draw_detections(frame, st.last_dets)
                            # Even after hold time expires, keep drawing for smoothness
                            # until new detection replaces old ones
                            elif hold_time <= COASTVISION_DET_HOLD_S * 2:
                                _draw_detections(frame, st.last_dets)

                    # --- HLS: feed annotated frame to FFmpeg encoder ---
                    if COASTVISION_HLS_ENABLED and _FFMPEG_PATH:
                        if not st.hls_ok:
                            h, w = frame.shape[:2]
                            _start_hls_encoder(st, w, h, fps)
                        _feed_hls_frame(st, frame)

                    # Higher JPEG quality (88) for crisp detection boxes and smooth video
                    ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ok2:
                        st.last_jpeg = jpg.tobytes()

                        # Cache a smaller JPEG for the grid to keep 12+ zones smooth.
                        try:
                            gf = frame
                            gh, gw = gf.shape[:2]
                            if COASTVISION_GRID_MAX_W > 0 and gw > COASTVISION_GRID_MAX_W:
                                scale = COASTVISION_GRID_MAX_W / float(gw)
                                gf = cv2.resize(
                                    gf,
                                    (int(gw * scale), int(gh * scale)),
                                    interpolation=cv2.INTER_LINEAR,  # Better quality resize
                                )
                            okg, jpg_g = cv2.imencode(
                                ".jpg",
                                gf,
                                [int(cv2.IMWRITE_JPEG_QUALITY), COASTVISION_GRID_JPEG_QUALITY],
                            )
                            if okg:
                                st.last_jpeg_grid = jpg_g.tobytes()
                        except Exception:
                            pass

                        st.last_ts = time.time()
                        st.last_error = None
                else:
                    st.last_error = "read_failed"
        except Exception as e:
            st.last_error = f"{type(e).__name__}: {e}"

        dt = time.time() - t0
        time.sleep(max(0.0, interval - dt))

    # Clean up HLS encoder when worker exits
    _stop_hls_encoder(st)
    _zone_threads.pop(zid, None)


# ----------------- API (Flask) -----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB max upload
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize WebSocket for real-time dashboard updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)
print("[init] WebSocket initialized for real-time alert broadcasting")

@socketio.on('connect')
def handle_connect(auth=None):
    """Handle WebSocket client connection"""
    client_type = 'unknown'
    if auth:
        client_type = auth.get('client_type', 'unknown')
    print(f"[WS] Client connected (type={client_type})")
    emit('connection_response', {'data': 'Connected to CoastVision backend'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket client disconnection"""
    print("[WS] Client disconnected")


@app.errorhandler(413)
def _too_large(e):
    return jsonify({"error": "File too large. Max upload size is 10 GB."}), 413


@app.errorhandler(500)
def _internal(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500


_workers_started = False
_workers_lock = threading.Lock()


def _start_workers_once():
    global _workers_started
    if _workers_started:
        return
    with _workers_lock:
        if _workers_started:
            return
        # Initialize HLS infrastructure
        if COASTVISION_HLS_ENABLED:
            _init_hls()
        _open_zone_caps()
        for zid in list(_zones.keys()):
            _ensure_zone_thread(zid)
        _workers_started = True


@app.before_request
def _ensure_started():
    _start_workers_once()


@app.route("/api/health", methods=["GET"])
def health():
    gpu_name = None
    gpu_vram_gb = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = round(float(props.total_memory) / (1024 ** 3), 2)
        except Exception:
            gpu_vram_gb = None

    cuda_info = _torch_cuda_build_info()

    return jsonify(
        {
            "status": "ok",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "requested_device": str(REQUESTED_DEVICE),
            "device": str(DEVICE),
            # NEW: definitive CUDA build/runtime info
            **cuda_info,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram_gb,
            "cuda_smoke_ok": bool(CUDA_SMOKE_OK),
            "cuda_smoke_error": CUDA_SMOKE_ERROR,
            "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
            "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None) if torch.cuda.is_available() else None,
            "zones": len(_zones),
            "alerts_cached": len(ALERT_HISTORY),
            "conf": CONF_THRES,
            "person_conf": PERSON_CONF_THRES,
            "iou": COASTVISION_IOU,
            "max_det": COASTVISION_MAX_DET,
            "alert_conf": COASTVISION_ALERT_CONF,
            "imgsz": COASTVISION_IMGSZ,
            "max_side": COASTVISION_MAX_SIDE,
            "fps": COASTVISION_FPS,
            "infer_every": COASTVISION_INFER_EVERY,
            "hls_enabled": COASTVISION_HLS_ENABLED and _FFMPEG_PATH is not None,
            "hls_nvenc": _HLS_USE_NVENC,
        }
    )


@app.route("/api/zones", methods=["GET"])

def zones():
    # Always rescan for ALL video files before reporting zones
    global ZONE_IDS
    ZONE_IDS = _find_zone_ids(VIDEO_DIR)
    name_map = _zid_to_filename()
    for zid in ZONE_IDS:
        if zid not in VIDEO_PATHS:
            filename = name_map.get(zid, f"zone{zid}.mp4")
            p = VIDEO_DIR / filename
            VIDEO_PATHS[zid] = p
            if p.exists():
                cap = _open_capture(p)
                if cap:
                    _zones[zid] = ZoneState(zid=zid, path=p, cap=cap, lock=threading.Lock())
                    _ensure_zone_thread(zid)
    # Remove zones for deleted videos
    for zid in list(VIDEO_PATHS.keys()):
        if zid not in ZONE_IDS:
            VIDEO_PATHS.pop(zid, None)
            zs = _zones.pop(zid, None)
            if zs:
                _stop_hls_encoder(zs)
                if zs.cap:
                    try:
                        zs.cap.release()
                    except Exception:
                        pass
            _zone_threads.pop(zid, None)
    items = []
    now = time.time()
    zid_names = _zid_to_filename()
    for zid, p in sorted(VIDEO_PATHS.items()):
        st = _zones.get(zid)
        items.append(
            {
                "id": zid,
                "name": _get_zone_display_name(zid),
                "filename": zid_names.get(zid, p.name),
                "path": str(p),
                "exists": bool(p.exists()),
                "active": zid in _zones,
                "is_opened": bool(st.cap.isOpened()) if st else False,
                "last_frame_age_s": (now - st.last_ts) if (st and st.last_ts) else None,
                "last_error": st.last_error if st else None,
            }
        )
    return jsonify({"items": items, "video_dir": str(VIDEO_DIR)})

# Optional: endpoint to force reload all videos (for UI button)
@app.route("/api/zones/reload", methods=["POST"])
def reload_zones():
    _open_zone_caps()
    for zid in list(_zones.keys()):
        _ensure_zone_thread(zid)
    return jsonify({"ok": True, "zones": [z for z in ZONE_IDS]})


# --------- VIDEO MANAGEMENT API ---------

@app.route("/api/videos", methods=["GET"])
def list_videos():
    """List all video files in VIDEO_DIR with their zone assignment."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    name_map = _zid_to_filename()
    zid_for_name = {v: k for k, v in name_map.items()}
    files = []
    for p in sorted(VIDEO_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS:
            zid = zid_for_name.get(p.name)
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            files.append({
                "filename": p.name,
                "size_mb": size_mb,
                "zone_id": zid,
                "active": zid in _zones if zid else False,
            })
    return jsonify({"items": files, "video_dir": str(VIDEO_DIR), "supported_extensions": list(_VIDEO_EXTENSIONS)})


@app.route("/api/videos/upload", methods=["POST"])
def upload_video():
    """Upload one or more video files. Auto-assigns zone IDs on next scan."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided. Use form field 'file'."}), 400

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        uploaded = []
        files = request.files.getlist("file")
        for f in files:
            if not f.filename:
                continue
            # Sanitize filename
            safe_name = Path(f.filename).name
            # Check extension
            ext = Path(safe_name).suffix.lower()
            if ext not in _VIDEO_EXTENSIONS:
                uploaded.append({"filename": safe_name, "ok": False, "error": f"Unsupported extension '{ext}'"})
                continue
            dest = VIDEO_DIR / safe_name
            # If file exists, add a suffix
            if dest.exists():
                stem = dest.stem
                i = 1
                while dest.exists():
                    dest = VIDEO_DIR / f"{stem}_{i}{ext}"
                    i += 1
            f.save(str(dest))
            uploaded.append({"filename": dest.name, "ok": True, "size_mb": round(dest.stat().st_size / (1024 * 1024), 2)})

        # Auto-reload zones to pick up new files
        _open_zone_caps()
        for zid in list(_zones.keys()):
            _ensure_zone_thread(zid)

        return jsonify({"uploaded": uploaded, "zones": [z for z in ZONE_IDS]})
    except Exception as e:
        print(f"[upload] Error: {type(e).__name__}: {e}")
        return jsonify({"error": f"Upload failed: {type(e).__name__}: {e}"}), 500


@app.route("/api/videos/<filename>", methods=["DELETE"])
def delete_video(filename: str):
    """Delete a video file and stop its zone."""
    safe_name = Path(filename).name
    p = VIDEO_DIR / safe_name
    if not p.exists():
        return jsonify({"error": f"File '{safe_name}' not found"}), 404

    # Find and stop corresponding zone
    name_map = _zid_to_filename()
    zid_for_name = {v: k for k, v in name_map.items()}
    zid = zid_for_name.get(safe_name)
    if zid:
        zs = _zones.pop(zid, None)
        if zs:
            _stop_hls_encoder(zs)
            if zs.cap:
                try:
                    zs.cap.release()
                except Exception:
                    pass
        VIDEO_PATHS.pop(zid, None)
        _zone_threads.pop(zid, None)
        _video_name_to_zid.pop(safe_name, None)

    try:
        p.unlink()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    global ZONE_IDS
    ZONE_IDS = _find_zone_ids(VIDEO_DIR)
    return jsonify({"ok": True, "deleted": safe_name, "zones": [z for z in ZONE_IDS]})

@app.route("/api/zones/<int:zid>/frame.jpg", methods=["GET"])
def zone_frame(zid: int):
    st = _zones.get(zid)
    if not st:
        jpg = _placeholder_jpeg(f"Zone {zid} unavailable", "Video file missing or cannot open")
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    if st.last_jpeg is None:
        jpg = _placeholder_jpeg(f"Zone {zid}", "Loading frames...")
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    w = request.args.get("w", "").strip()
    if w:
        jpg = st.last_jpeg_grid or st.last_jpeg
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    return Response(st.last_jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/zones/<int:zid>/stream.mjpg", methods=["GET"])
def zone_stream(zid: int):
    st = _zones.get(zid)
    if not st:
        abort(404)

    boundary = "frame"
    frame_interval_s = 1.0 / max(1, COASTVISION_FPS)

    def gen():
        while True:
            try:
                jpg = st.last_jpeg
                if not jpg:
                    jpg = _placeholder_jpeg(f"Zone {zid}", "Loading frames...")
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                    + jpg
                    + b"\r\n"
                )
            except GeneratorExit:
                break
            except Exception:
                pass
            time.sleep(frame_interval_s)

    return Response(
        gen(),
        mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store"},
    )


# ----- HLS Streaming Endpoints -----

@app.route("/api/zones/<int:zid>/hls/stream.m3u8", methods=["GET"])
def zone_hls_playlist(zid: int):
    """Serve the HLS playlist (.m3u8) for a zone."""
    st = _zones.get(zid)
    if not st or not st.hls_dir:
        abort(404)
    p = st.hls_dir / "stream.m3u8"
    if not p.exists():
        abort(404)
    data = p.read_bytes()
    return Response(
        data,
        mimetype="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/zones/<int:zid>/hls/<path:filename>", methods=["GET"])
def zone_hls_segment(zid: int, filename: str):
    """Serve HLS .ts segments for a zone."""
    st = _zones.get(zid)
    if not st or not st.hls_dir:
        abort(404)
    # Security: only allow .ts files, no path traversal
    safe_name = Path(filename).name
    if not safe_name.endswith(".ts"):
        abort(400)
    p = st.hls_dir / safe_name
    if not p.exists():
        abort(404)
    data = p.read_bytes()
    return Response(
        data,
        mimetype="video/mp2t",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/hls/status", methods=["GET"])
def hls_status():
    """Return HLS streaming status for all zones."""
    zones_status = {}
    for zid, st in _zones.items():
        playlist_exists = False
        segment_count = 0
        if st.hls_dir and st.hls_dir.exists():
            playlist_exists = (st.hls_dir / "stream.m3u8").exists()
            segment_count = len(list(st.hls_dir.glob("*.ts")))
        zones_status[str(zid)] = {
            "hls_ok": st.hls_ok,
            "encoder_running": st.hls_proc is not None and st.hls_proc.poll() is None,
            "playlist_ready": playlist_exists,
            "segments": segment_count,
            "resolution": f"{st.hls_frame_w}x{st.hls_frame_h}" if st.hls_frame_w else None,
        }
    return jsonify({
        "hls_enabled": COASTVISION_HLS_ENABLED and _FFMPEG_PATH is not None,
        "nvenc": _HLS_USE_NVENC,
        "ffmpeg": _FFMPEG_PATH,
        "zones": zones_status,
    })


@app.route("/api/zones/<int:zid>/detections", methods=["GET"])
def zone_detections(zid: int):
    st = _zones.get(zid)
    if not st:
        return jsonify({"zone": zid, "count": 0, "age_s": None, "items": []})
    age = (time.time() - st.last_dets_ts) if st.last_dets_ts else None
    items = st.last_dets or []
    return jsonify({"zone": zid, "count": len(items), "age_s": age, "items": items})


@app.route("/api/zones/<int:zid>/timeline", methods=["GET"])
def zone_timeline(zid: int):
    """Get person count timeline data for a zone."""
    if zid not in _zone_person_history:
        return jsonify({"zone": zid, "timeline": []})
    
    history = list(_zone_person_history[zid])
    # Convert to frontend-friendly format
    timeline = [{"timestamp": ts, "count": count} for ts, count in history]
    return jsonify({"zone": zid, "timeline": timeline})


@app.route("/api/zones/<int:zid>/name", methods=["GET", "POST"])
def zone_name(zid: int):
    """Get or set custom name for a zone."""
    if request.method == "GET":
        return jsonify({
            "zone": zid,
            "name": _get_zone_display_name(zid),
            "is_custom": zid in _zone_custom_names
        })
    
    elif request.method == "POST":
        data = request.get_json() or {}
        new_name = data.get("name", "").strip()
        
        if _set_zone_name(zid, new_name):
            return jsonify({
                "success": True,
                "zone": zid,
                "name": _get_zone_display_name(zid),
                "is_custom": zid in _zone_custom_names
            })
        else:
            return jsonify({"success": False, "error": "Invalid name"}), 400


@app.route("/api/analytics/timeline", methods=["GET"])
def analytics_timeline():
    """Get person count timeline data for all zones."""
    data = {}
    for zid in _zone_person_history:
        history = list(_zone_person_history[zid])
        data[str(zid)] = {
            "name": _get_zone_display_name(zid),
            "timeline": [{"timestamp": ts, "count": count} for ts, count in history]
        }
    return jsonify(data)


@app.route("/api/alerts", methods=["GET"])
def alerts():
    limit = int(request.args.get("limit", "120"))
    zone = request.args.get("zone", "").strip()
    out = []
    for a in list(ALERT_HISTORY):
        if zone and str(a.get("zone")) != str(zone):
            continue
        out.append(a)
        if len(out) >= limit:
            break
    return jsonify({"items": out})


@app.route("/api/analysis", methods=["GET"])
def analysis():
    zone = request.args.get("zone", "").strip()
    items = list(ALERT_HISTORY)
    if zone:
        items = [a for a in items if str(a.get("zone")) == str(zone)]

    by_zone = Counter(str(a.get("zone")) for a in items)
    by_label = Counter(str(a.get("label")) for a in items)
    return jsonify(
        {
            "alerts_total": len(items),
            "alerts_by_zone": dict(by_zone),
            "alerts_by_label": dict(by_label),
        }
    )


@app.route("/api/analytics/response-times", methods=["GET"])
def response_times_analytics():
    """Get lifeguard response time metrics."""
    limit = int(request.args.get("limit", "100"))
    zone_filter = request.args.get("zone", "").strip()
    
    response_times = []
    recent_responses = []
    by_zone = {}
    by_lifeguard = {}
    
    try:
        with open(RESPONSE_TIMES_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    response_sec = float(row.get("response_time_seconds", 0))
                    zone = row.get("zone", "")
                    lg_id = row.get("lifeguard_id", "")
                    lg_name = row.get("lifeguard_name", "")
                    
                    # Filter by zone if specified
                    if zone_filter and zone != zone_filter:
                        continue
                    
                    response_times.append(response_sec)
                    recent_responses.append({
                        "timestamp": row.get("timestamp", ""),
                        "zone": zone,
                        "lifeguard_id": lg_id,
                        "lifeguard_name": lg_name,
                        "response_time_seconds": response_sec,
                        "alert_sent_at": row.get("alert_sent_at", ""),
                        "responded_at": row.get("responded_at", "")
                    })
                    
                    # Group by zone
                    if zone not in by_zone:
                        by_zone[zone] = []
                    by_zone[zone].append(response_sec)
                    
                    # Group by lifeguard
                    if lg_id not in by_lifeguard:
                        by_lifeguard[lg_id] = {"name": lg_name, "times": []}
                    by_lifeguard[lg_id]["times"].append(response_sec)
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    
    # Calculate statistics
    avg_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0
    min_response_time = round(min(response_times), 2) if response_times else 0
    max_response_time = round(max(response_times), 2) if response_times else 0
    
    # Calculate by zone
    zone_stats = {}
    for zone, times in by_zone.items():
        zone_stats[zone] = {
            "count": len(times),
            "avg": round(sum(times) / len(times), 2),
            "min": round(min(times), 2),
            "max": round(max(times), 2)
        }
    
    # Calculate by lifeguard
    lifeguard_stats = {}
    for lg_id, data in by_lifeguard.items():
        times = data["times"]
        lifeguard_stats[lg_id] = {
            "name": data["name"],
            "count": len(times),
            "avg": round(sum(times) / len(times), 2),
            "min": round(min(times), 2),
            "max": round(max(times), 2)
        }
    
    return jsonify({
        "overall": {
            "total_responses": len(response_times),
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time
        },
        "by_zone": zone_stats,
        "by_lifeguard": lifeguard_stats,
        "recent": recent_responses[-limit:] if recent_responses else []
    })


@app.route("/api/zones/<int:zid>/crowd-status", methods=["GET"])
def zone_crowd_status(zid: int):
    """Get current crowd density status for a zone."""
    with _crowd_lock:
        status = CROWD_STATUS.get(zid, {
            "count": 0,
            "threshold": CROWD_THRESHOLDS.get(zid, 50),
            "status": "normal",
            "exceeded": False
        })
    
    return jsonify({
        "zone": zid,
        "zone_name": _get_zone_display_name(zid),
        "person_count": status.get("count", 0),
        "threshold": status.get("threshold", 50),
        "status": status.get("status", "normal"),
        "exceeded": status.get("exceeded", False),
        "safety_percentage": round((1 - min(1, status.get("count", 0) / max(1, status.get("threshold", 50)))) * 100, 1)
    })


@app.route("/api/analytics/crowd-status", methods=["GET"])
def crowd_status_all():
    """Get crowd density status for all zones."""
    zones_status = {}
    
    with _crowd_lock:
        for zid in ZONE_IDS:
            status = CROWD_STATUS.get(zid, {
                "count": 0,
                "threshold": CROWD_THRESHOLDS.get(zid, 50),
                "status": "normal",
                "exceeded": False
            })
            person_count = status.get("count", 0)
            threshold = status.get("threshold", 50)
            
            zones_status[str(zid)] = {
                "zone_name": _get_zone_display_name(zid),
                "person_count": person_count,
                "threshold": threshold,
                "status": status.get("status", "normal"),
                "exceeded": status.get("exceeded", False),
                "safety_percentage": round((1 - min(1, person_count / max(1, threshold))) * 100, 1),
                "crowding_level": round((person_count / max(1, threshold)) * 100, 1)
            }
    
    # Count crowded zones
    crowded_zones = sum(1 for z in zones_status.values() if z["exceeded"])
    
    return jsonify({
        "zones": zones_status,
        "crowded_zones_count": crowded_zones,
        "total_zones": len(ZONE_IDS),
        "overall_safety": "safe" if crowded_zones == 0 else "warning" if crowded_zones <= len(ZONE_IDS) // 2 else "critical"
    })


@app.route("/api/zones/<int:zid>/crowd-threshold", methods=["GET", "POST"])
def crowd_threshold(zid: int):
    """Get or set crowd threshold for a zone."""
    if request.method == "GET":
        threshold = CROWD_THRESHOLDS.get(zid, 50)
        return jsonify({"zone": zid, "threshold": threshold})
    
    elif request.method == "POST":
        data = request.get_json() or {}
        new_threshold = data.get("threshold")
        
        if new_threshold is None or not isinstance(new_threshold, (int, float)):
            return jsonify({"error": "threshold must be a number"}), 400
        
        new_threshold = int(new_threshold)
        if new_threshold < 1:
            return jsonify({"error": "threshold must be at least 1"}), 400
        
        with _crowd_lock:
            CROWD_THRESHOLDS[zid] = new_threshold
            _save_crowd_thresholds()
        
        return jsonify({
            "zone": zid,
            "threshold": new_threshold,
            "message": f"Zone {zid} crowd threshold updated to {new_threshold} people"
        })


@app.route("/api/analytics/crowd-alerts", methods=["GET"])
def crowd_alerts():
    """Get crowd alert history."""
    try:
        limit = int(request.args.get("limit", "100"))
        zone_filter = request.args.get("zone", "").strip()
        
        with _crowd_lock:
            alerts = []
            for alert in list(CROWD_ALERT_HISTORY):
                if zone_filter and str(alert.get("zone")) != str(zone_filter):
                    continue
                alerts.append({
                    "timestamp": alert.get("ts"),
                    "zone": alert.get("zone"),
                    "zone_name": _get_zone_display_name(alert.get("zone")),
                    "person_count": alert.get("person_count"),
                    "threshold": alert.get("threshold"),
                    "severity": alert.get("severity"),
                })
                if len(alerts) >= limit:
                    break
        
        # Count by severity
        severity_counts = {
            "low": sum(1 for a in alerts if a["severity"] == "low"),
            "medium": sum(1 for a in alerts if a["severity"] == "medium"),
            "high": sum(1 for a in alerts if a["severity"] == "high")
        }
        thresholds = {str(zid): threshold for zid, threshold in CROWD_THRESHOLDS.items()}
        
        return jsonify({
            "alerts": alerts,
            "total": len(alerts),
            "severity_counts": severity_counts,
            "thresholds": thresholds
        })
    except Exception as e:
        print("[crowd] crowd_alerts error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal crowd-alerts error", "details": str(e)}), 500


# ----------------- LIFEGUARD API ENDPOINTS -----------------

@app.route("/api/lifeguards/register", methods=["POST"])
def register_lifeguard():
    """Register a new lifeguard. Returns session token."""
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    with _lifeguard_lock:
        # Check if phone already registered
        for lg_id, lg in LIFEGUARDS.items():
            if phone and lg.get("phone") == phone:
                # Return existing session
                session_token = str(uuid.uuid4())
                LIFEGUARD_SESSIONS[session_token] = lg_id
                lg["online"] = True
                lg["last_seen"] = time.time()
                _save_lifeguards()
                return jsonify({
                    "id": lg_id,
                    "name": lg["name"],
                    "phone": lg.get("phone"),
                    "zones": lg.get("zones", []),
                    "session_token": session_token,
                    "message": "Welcome back!"
                })
        
        # Create new lifeguard
        lg_id = str(uuid.uuid4())[:8]
        LIFEGUARDS[lg_id] = {
            "id": lg_id,
            "name": name,
            "phone": phone,
            "zones": [],  # Empty = all zones
            "online": True,
            "last_seen": time.time(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        session_token = str(uuid.uuid4())
        LIFEGUARD_SESSIONS[session_token] = lg_id
        LIFEGUARD_ALERTS[lg_id] = deque(maxlen=100)
        _save_lifeguards()
        
    return jsonify({
        "id": lg_id,
        "name": name,
        "phone": phone,
        "zones": [],
        "session_token": session_token,
        "message": "Registration successful!"
    }), 201


def _get_lifeguard_id_from_token():
    """Extract lifeguard id from Authorization bearer token."""
    auth_header = request.headers.get("Authorization", "") or ""
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        return LIFEGUARD_SESSIONS.get(token), token
    return None, None


@app.route("/api/lifeguards/login", methods=["POST"])
def lifeguard_login():
    """Log in an existing lifeguard by phone and return a session token."""
    data = request.get_json() or {}
    phone = str(data.get("phone", "")).strip()
    if not phone:
        return jsonify({"error": "Phone number is required"}), 400

    with _lifeguard_lock:
        lg_id = None
        for lid, lg in LIFEGUARDS.items():
            if str(lg.get("phone", "")).strip() == phone:
                lg_id = lid
                break

        if lg_id is None:
            return jsonify({"error": "Lifeguard not found"}), 404

        lg = LIFEGUARDS[lg_id]
        session_token = str(uuid.uuid4())
        LIFEGUARD_SESSIONS[session_token] = lg_id
        lg["online"] = True
        lg["last_seen"] = time.time()
        _save_lifeguards()

    return jsonify({
        "id": lg_id,
        "name": lg.get("name"),
        "phone": lg.get("phone"),
        "zones": lg.get("zones", []),
        "session_token": session_token,
        "message": "Login successful"
    })


@app.route("/api/lifeguards/me", methods=["GET"])
def lifeguard_me():
    """Return the currently authenticated lifeguard profile."""
    lg_id, _token = _get_lifeguard_id_from_token()
    if not lg_id:
        return jsonify({"error": "Unauthorized"}), 401

    with _lifeguard_lock:
        lg = LIFEGUARDS.get(lg_id)
        if not lg:
            return jsonify({"error": "Lifeguard not found"}), 404

        lg_copy = dict(lg)
        host = request.host_url.rstrip("/")
        avatar = lg_copy.get("avatar")
        thumb = lg_copy.get("avatar_thumb")
        if avatar and isinstance(avatar, str) and avatar.startswith("/"):
            lg_copy["avatar"] = f"{host}{avatar}"
        if thumb and isinstance(thumb, str) and thumb.startswith("/"):
            lg_copy["avatar_thumb"] = f"{host}{thumb}"

        # Only return common profile fields (include avatar URLs when present)
        return jsonify({
            "id": lg_id,
            "name": lg_copy.get("name"),
            "phone": lg_copy.get("phone"),
            "zones": lg_copy.get("zones", []),
            "online": lg_copy.get("online", False),
            "last_seen": lg_copy.get("last_seen"),
            "avatar": lg_copy.get("avatar"),
            "avatar_thumb": lg_copy.get("avatar_thumb"),
        })


@app.route("/api/lifeguards/logout", methods=["POST"])
def lifeguard_logout():
    """Log out the current lifeguard session."""
    lg_id, token = _get_lifeguard_id_from_token()
    if not token:
        return jsonify({"status": "ok"})

    with _lifeguard_lock:
        if token in LIFEGUARD_SESSIONS:
            del LIFEGUARD_SESSIONS[token]
        if lg_id and lg_id in LIFEGUARDS:
            LIFEGUARDS[lg_id]["online"] = False
            _save_lifeguards()

    return jsonify({"status": "ok"})


@app.route("/api/lifeguards", methods=["GET"])
def list_lifeguards():
    """List all registered lifeguards (admin view)."""
    with _lifeguard_lock:
        # Return absolute URLs for avatars so mobile clients can load them
        host = request.host_url.rstrip("/")
        lifeguards_out = []
        for lg in LIFEGUARDS.values():
            lg_copy = dict(lg)
            avatar = lg_copy.get("avatar")
            thumb = lg_copy.get("avatar_thumb")
            if avatar and avatar.startswith("/"):
                lg_copy["avatar"] = f"{host}{avatar}"
            if thumb and thumb.startswith("/"):
                lg_copy["avatar_thumb"] = f"{host}{thumb}"
            lifeguards_out.append(lg_copy)
        return jsonify({
            "lifeguards": lifeguards_out,
            "count": len(lifeguards_out)
        })


@app.route("/api/lifeguards/<lg_id>", methods=["GET"])
def get_lifeguard(lg_id: str):
    """Get lifeguard details."""
    with _lifeguard_lock:
        lg = LIFEGUARDS.get(lg_id)
        if not lg:
            return jsonify({"error": "Lifeguard not found"}), 404
        lg_copy = dict(lg)
        host = request.host_url.rstrip("/")
        avatar = lg_copy.get("avatar")
        thumb = lg_copy.get("avatar_thumb")
        if avatar and avatar.startswith("/"):
            lg_copy["avatar"] = f"{host}{avatar}"
        if thumb and thumb.startswith("/"):
            lg_copy["avatar_thumb"] = f"{host}{thumb}"
        return jsonify(lg_copy)


@app.route("/api/lifeguards/<lg_id>/assign", methods=["POST"])
def assign_lifeguard_zones(lg_id: str):
    """Assign zones to a lifeguard. Empty list = all zones."""
    data = request.get_json() or {}
    zones = data.get("zones", [])
    
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        LIFEGUARDS[lg_id]["zones"] = [int(z) for z in zones]
        _save_lifeguards()
        
    return jsonify({
        "id": lg_id,
        "zones": LIFEGUARDS[lg_id]["zones"],
        "message": f"Assigned to zones: {zones if zones else 'ALL'}"
    })


@app.route("/api/lifeguards/<lg_id>/avatar", methods=["POST"])
def upload_lifeguard_avatar(lg_id: str):
    """Upload and store a lifeguard avatar image. Returns public URL."""
    if 'avatar' not in request.files:
        return jsonify({"error": "No file part 'avatar'"}), 400
    file = request.files['avatar']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404

        # Ensure avatars directory exists
        avatars_dir = (ALERTS_DIR / "lifeguard_avatars").resolve()
        try:
            avatars_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return jsonify({"error": f"Could not create avatars dir: {e}"}), 500

        # Build safe filename
        import time, werkzeug
        fname = werkzeug.utils.secure_filename(file.filename)
        ext = Path(fname).suffix or ".jpg"
        out_name = f"{lg_id}_{int(time.time())}{ext}"
        out_path = avatars_dir / out_name

        try:
            file.save(str(out_path))
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {e}"}), 500

        # Optionally create a thumbnail if PIL available
        thumb_name = None
        if Image is not None:
            try:
                img = Image.open(out_path)
                img.thumbnail((320, 320))
                thumb_name = f"{lg_id}_{int(time.time())}_thumb{ext}"
                thumb_path = avatars_dir / thumb_name
                img.save(thumb_path)
            except Exception as e:
                print(f"[avatar] Thumbnail generation failed: {e}")

        # Save relative URL into lifeguard record
        public_url = f"/lifeguard_avatars/{out_name}"
        LIFEGUARDS[lg_id]["avatar"] = public_url
        if thumb_name:
            LIFEGUARDS[lg_id]["avatar_thumb"] = f"/lifeguard_avatars/{thumb_name}"
        _save_lifeguards()

    # Return both paths (relative) for frontend convenience
    resp = {"avatar_url": public_url}
    if thumb_name:
        resp["avatar_thumb_url"] = f"/lifeguard_avatars/{thumb_name}"
    return jsonify(resp), 201


@app.route("/api/lifeguards/<lg_id>", methods=["DELETE"])
def delete_lifeguard(lg_id: str):
    """Delete a lifeguard and associated avatar files."""
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        # delete avatar files if present
        avatars_dir = (ALERTS_DIR / "lifeguard_avatars").resolve()
        lg = LIFEGUARDS[lg_id]
        for key in ("avatar", "avatar_thumb"):
            path = lg.get(key)
            if path and isinstance(path, str) and path.startswith("/lifeguard_avatars/"):
                fname = path.split("/lifeguard_avatars/", 1)[-1]
                try:
                    p = avatars_dir / fname
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    print(f"[lifeguard] Failed to remove avatar file {fname}: {e}")
        # remove lifeguard
        del LIFEGUARDS[lg_id]
        LIFEGUARD_ALERTS.pop(lg_id, None)
        LIFEGUARD_SSE_QUEUES.pop(lg_id, None)
        # remove any sessions pointing to this id
        for token, lid in list(LIFEGUARD_SESSIONS.items()):
            if lid == lg_id:
                del LIFEGUARD_SESSIONS[token]
        _save_lifeguards()
    return jsonify({"status": "deleted"})


@app.route('/lifeguard_avatars/<path:filename>')
def serve_lifeguard_avatar(filename: str):
    avatars_dir = (ALERTS_DIR / "lifeguard_avatars").resolve()
    if not (avatars_dir / filename).exists():
        abort(404)
    return send_from_directory(str(avatars_dir), filename)


@app.route("/api/lifeguards/<lg_id>/alerts", methods=["GET"])
def get_lifeguard_alerts(lg_id: str):
    """Get alerts for a lifeguard's assigned zones."""
    limit = int(request.args.get("limit", "50"))
    
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        
        lg = LIFEGUARDS[lg_id]
        assigned_zones = lg.get("zones", [])
        
        # Get alerts from history for assigned zones
        alerts = []
        for a in list(ALERT_HISTORY):
            if not assigned_zones or a.get("zone") in assigned_zones:
                alerts.append(a)
                if len(alerts) >= limit:
                    break
        
        return jsonify({
            "lifeguard_id": lg_id,
            "assigned_zones": assigned_zones,
            "alerts": alerts,
            "count": len(alerts)
        })


@app.route("/api/lifeguards/<lg_id>/respond", methods=["POST"])
def lifeguard_respond(lg_id: str):
    """Mark that lifeguard is responding to an alert."""
    data = request.get_json() or {}
    alert_id = data.get("alert_id")
    zone = data.get("zone")
    
    response_time_seconds = None
    
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        
        lg = LIFEGUARDS[lg_id]
        responded_at = datetime.now(timezone.utc)
        
        # Calculate response time if we have the alert sent time
        if alert_id and alert_id in ALERT_SENT_TIMES:
            alert_sent_time = ALERT_SENT_TIMES[alert_id]
            response_time_seconds = time.time() - alert_sent_time
            
            # Log to CSV
            try:
                with open(RESPONSE_TIMES_CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now(timezone.utc).isoformat(),
                        zone,
                        lg_id,
                        lg["name"],
                        round(response_time_seconds, 2),
                        datetime.fromtimestamp(alert_sent_time, tz=timezone.utc).isoformat(),
                        responded_at.isoformat()
                    ])
            except Exception as e:
                print(f"[response] Error logging response time: {e}")
        
        response_record = {
            "lifeguard_id": lg_id,
            "lifeguard_name": lg["name"],
            "alert_id": alert_id,
            "zone": zone,
            "responded_at": responded_at.isoformat(),
            "response_time_seconds": response_time_seconds
        }
        
        # Log response
        print(f"[lifeguard] {lg['name']} responding to zone {zone}" + 
              (f" (response time: {response_time_seconds:.1f}s)" if response_time_seconds else ""))
        
        return jsonify({
            "message": f"{lg['name']} is responding to zone {zone}",
            "response": response_record,
            "response_time_seconds": response_time_seconds
        })


@app.route("/api/lifeguards/<lg_id>/heartbeat", methods=["POST"])
def lifeguard_heartbeat(lg_id: str):
    """Update lifeguard online status."""
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        LIFEGUARDS[lg_id]["online"] = True
        LIFEGUARDS[lg_id]["last_seen"] = time.time()
        _save_lifeguards()
    return jsonify({"status": "ok"})


@app.route("/api/lifeguards/<lg_id>/stream", methods=["GET"])
def lifeguard_alert_stream(lg_id: str):
    """Server-Sent Events stream for real-time alerts to lifeguard."""
    with _lifeguard_lock:
        if lg_id not in LIFEGUARDS:
            return jsonify({"error": "Lifeguard not found"}), 404
        
        # Create SSE queue for this lifeguard
        if lg_id not in LIFEGUARD_SSE_QUEUES:
            LIFEGUARD_SSE_QUEUES[lg_id] = queue.Queue(maxsize=50)
        q = LIFEGUARD_SSE_QUEUES[lg_id]
    
    def generate():
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'lifeguard_id': lg_id})}\n\n"
        
        while True:
            try:
                # Wait for alert with timeout (sends keepalive)
                try:
                    alert = q.get(timeout=30)
                    yield f"data: {json.dumps({'type': 'alert', 'alert': alert})}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            except GeneratorExit:
                # Client disconnected
                with _lifeguard_lock:
                    if lg_id in LIFEGUARD_SSE_QUEUES:
                        del LIFEGUARD_SSE_QUEUES[lg_id]
                break
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/admin/broadcast", methods=["POST"])
def admin_broadcast():
    """Admin: Send manual alert to all lifeguards or specific zones."""
    data = request.get_json() or {}
    message = data.get("message", "Manual alert from admin")
    zones = data.get("zones", [])  # Empty = all
    
    alert = {
        "type": "admin_alert",
        "message": message,
        "zones": zones,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": str(uuid.uuid4())[:8]
    }
    
    with _lifeguard_lock:
        for lg_id, lg in LIFEGUARDS.items():
            assigned = lg.get("zones", [])
            # Send if no specific zones, or lifeguard is assigned to one of the zones
            if not zones or not assigned or any(z in assigned for z in zones):
                if lg_id in LIFEGUARD_SSE_QUEUES:
                    try:
                        LIFEGUARD_SSE_QUEUES[lg_id].put_nowait(alert)
                    except:
                        pass
    
    return jsonify({"message": "Broadcast sent", "alert": alert})


# =============== TELEGRAM NOTIFICATION ENDPOINTS ===============

@app.route("/api/telegram/status", methods=["GET"])
def telegram_status():
    """Get Telegram bot connection status."""
    return jsonify(notifier.test_connection())


@app.route("/api/telegram/register", methods=["POST"])
def telegram_register():
    """Register a lifeguard's Telegram chat ID.
    
    Expected JSON:
    {
        "lifeguard_id": "abc12345",
        "chat_id": 123456789,
        "username": "@username" (optional)
    }
    """
    data = request.get_json() or {}
    lg_id = data.get("lifeguard_id", "").strip()
    chat_id = data.get("chat_id")
    username = data.get("username", "").strip()
    
    if not lg_id or not chat_id:
        return jsonify({"error": "Missing lifeguard_id or chat_id"}), 400
    
    success = notifier.register_user(lg_id, chat_id, username)
    
    if success:
        return jsonify({
            "status": "registered",
            "lifeguard_id": lg_id,
            "chat_id": chat_id,
            "message": f"Telegram registered for lifeguard {lg_id}"
        }), 201
    else:
        return jsonify({
            "status": "failed",
            "message": "Failed to register Telegram"
        }), 400


@app.route("/api/telegram/unregister/<lg_id>", methods=["POST"])
def telegram_unregister(lg_id: str):
    """Unregister a lifeguard's Telegram."""
    success = notifier.unregister_user(lg_id)
    
    if success:
        return jsonify({
            "status": "unregistered",
            "lifeguard_id": lg_id,
            "message": f"Telegram unregistered"
        })
    else:
        return jsonify({
            "error": "Not registered"
        }), 404


@app.route("/api/telegram/<lg_id>", methods=["GET"])
def telegram_get_user(lg_id: str):
    """Get Telegram info for a lifeguard."""
    user_info = notifier.get_user(lg_id)
    
    if user_info:
        # Include both the raw telegram info and a top-level chat_id field
        # so frontend clients can easily consume this without knowing the
        # internal structure of user_info.
        chat_id = user_info.get("chat_id")
        return jsonify({
            "lifeguard_id": lg_id,
            "telegram": user_info,
            "chat_id": chat_id
        })
    else:
        return jsonify({
            "error": "Not registered"
        }), 404


@app.route("/api/telegram/<lg_id>/test", methods=["POST"])
def telegram_test(lg_id: str):
    """Send a test drowning alert for *this* lifeguard only.

    Mapping rules (simple and deterministic):
    - If the lifeguard ID looks like "lifeguard_<N>", always use Zone N.
      (So lifeguard_1 -> Zone 1, lifeguard_3 -> Zone 3, etc.)
    - Otherwise fall back to Zone 1.

    This ignores the separate LIFEGUARDS registry on purpose so that
    Telegram tests from the Lifeguards tab never jump to random zones.
    """
    if not notifier.enabled:
        return jsonify({
            "status": "error",
            "message": "❌ Telegram bot not configured. Set COASTVISION_TELEGRAM_BOT_TOKEN environment variable."
        }), 400

    user_info = notifier.get_user(lg_id)
    if not user_info:
        return jsonify({
            "status": "failed",
            "message": "Lifeguard is not registered for Telegram.",
            "lifeguard_id": lg_id,
        }), 404

    if user_info.get("paused"):
        return jsonify({
            "status": "paused",
            "message": "Notifications are stopped for this lifeguard. Click Resume first.",
            "lifeguard_id": lg_id,
        }), 409

    import random

    # Infer zone directly from the lifeguard ID pattern lifeguard_<N>.
    # This keeps tests 1:1 with the zone cards shown in the dashboard.
    zid = 1
    if isinstance(lg_id, str) and lg_id.startswith("lifeguard_"):
        try:
            parsed = int(lg_id.split("_", 1)[1])
            if parsed > 0:
                zid = parsed
        except Exception:
            zid = 1

    zone_name = f"Zone {zid}"

    detection_type = "Drowning"
    confidence = random.uniform(78, 95)  # Realistic confidence range

    success = notifier.send_alert(lg_id, zone_name, detection_type, confidence)

    if success:
        return jsonify({
            "status": "sent",
            "message": f"Test alert sent: {detection_type} in {zone_name} ({confidence:.1f}%)",
            "lifeguard_id": lg_id,
            "zone": zone_name,
            "confidence": f"{confidence:.1f}%",
            "detection_type": detection_type
        }), 200
    else:
        # Likely causes: wrong chat ID, user has not started a conversation
        # with the bot, or Telegram API rejected the chat.
        last_error = ""
        try:
            last_error = notifier.get_last_error(lg_id)
        except Exception:
            last_error = ""
        detail = f" Details: {last_error}" if last_error else ""
        return jsonify({
            "status": "failed",
            "message": "Failed to send test alert. Please verify the chat ID and make sure this Telegram account has started a chat with the bot." + detail,
            "lifeguard_id": lg_id,
            "zone": zone_name,
            "detection_type": detection_type,
            "error_detail": last_error,
        }), 400


@app.route("/api/telegram/<lg_id>/pause", methods=["POST"])
def telegram_pause(lg_id: str):
    """Pause Telegram notifications for a lifeguard."""
    success = notifier.set_paused(lg_id, True)
    if not success:
        return jsonify({"error": "Not registered"}), 404
    return jsonify({
        "status": "paused",
        "lifeguard_id": lg_id,
        "message": "Notifications stopped for this lifeguard"
    })


@app.route("/api/telegram/<lg_id>/resume", methods=["POST"])
def telegram_resume(lg_id: str):
    """Resume Telegram notifications for a lifeguard."""
    success = notifier.set_paused(lg_id, False)
    if not success:
        return jsonify({"error": "Not registered"}), 404
    return jsonify({
        "status": "active",
        "lifeguard_id": lg_id,
        "message": "Notifications resumed for this lifeguard"
    })


@app.route("/api/telegram/alert", methods=["POST"])
def telegram_send_alert():
    """Manually send a Telegram alert (for testing).
    
    Expected JSON:
    {
        "lifeguard_id": "abc12345",
        "zone": "Zone 3",
        "detection_type": "Drowning",
        "confidence": 95.5
    }
    """
    data = request.get_json() or {}
    lg_id = data.get("lifeguard_id", "").strip()
    zone = data.get("zone", "Unknown Zone")
    detection_type = data.get("detection_type", "Detection")
    confidence = float(data.get("confidence", 0))
    
    if not lg_id:
        return jsonify({"error": "Missing lifeguard_id"}), 400
    
    success = notifier.send_alert(lg_id, zone, detection_type, confidence)
    
    status_code = 200 if success else 400
    return jsonify({
        "status": "sent" if success else "failed",
        "lifeguard_id": lg_id,
        "zone": zone,
        "detection_type": detection_type,
        "confidence": confidence
    }), status_code


@app.route("/api/telegram/crowd-alert", methods=["POST"])
def telegram_crowd_alert():
    """Send a crowd density alert via Telegram.
    
    Expected JSON:
    {
        "zone": "Zone 3",
        "person_count": 150,
        "threshold": 100
    }
    """
    data = request.get_json() or {}
    zone = data.get("zone", "Unknown Zone")
    person_count = int(data.get("person_count", 0))
    threshold = int(data.get("threshold", 0))
    
    success = notifier.send_crowd_alert(zone, person_count, threshold)
    
    status_code = 200 if success else 400
    return jsonify({
        "status": "sent" if success else "failed",
        "zone": zone,
        "person_count": person_count,
        "threshold": threshold
    }), status_code



if __name__ == "__main__":
    # Dev convenience only. For Windows stability and concurrency, prefer:
    #   .\run_backend.ps1  (Waitress + correct venv python)
    host = os.environ.get("COASTVISION_HOST", "127.0.0.1")
    port = int(os.environ.get("COASTVISION_PORT", "8000"))
    print(f"[main] Starting Flask server with WebSocket support on http://{host}:{port} (device={DEVICE})")
    print("[main] WebSocket enabled for real-time dashboard updates")
    print("[main] Tip: use run_backend.ps1 for production-like serving.")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
