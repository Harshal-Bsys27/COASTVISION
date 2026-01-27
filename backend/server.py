"""CoastVision backend: serves annotated zone frames, alerts, and analysis."""

from __future__ import annotations

import csv
import os
import threading
import time
import sys
import platform
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

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

def _find_zone_ids(video_dir: Path) -> list:
    """Return sorted list of all zone IDs (ints) for zone*.mp4 in the video dir."""
    ids = set()
    for p in video_dir.glob("zone*.mp4"):
        m = re.match(r"zone(\d+)\.mp4", p.name, re.IGNORECASE)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)

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

# Performance
COASTVISION_MAX_SIDE = int(os.environ.get("COASTVISION_MAX_SIDE", "1280"))
COASTVISION_FPS = int(os.environ.get("COASTVISION_FPS", "10"))
COASTVISION_INFER_EVERY = int(os.environ.get("COASTVISION_INFER_EVERY", "2"))
COASTVISION_IMGSZ = int(os.environ.get("COASTVISION_IMGSZ", "640"))

# Grid playback: serve a smaller cached JPEG to reduce bandwidth and stutter
COASTVISION_GRID_MAX_W = int(os.environ.get("COASTVISION_GRID_MAX_W", "640"))
COASTVISION_GRID_JPEG_QUALITY = int(os.environ.get("COASTVISION_GRID_JPEG_QUALITY", "72"))

COASTVISION_DEVICE = os.environ.get("COASTVISION_DEVICE", "").strip()
COASTVISION_ALERT_COOLDOWN_S = float(os.environ.get("COASTVISION_ALERT_COOLDOWN_S", "4"))
COASTVISION_DET_HOLD_S = float(os.environ.get("COASTVISION_DET_HOLD_S", "0.75"))
COASTVISION_OVERLAY_STYLE = os.environ.get("COASTVISION_OVERLAY_STYLE", "pro").strip().lower()

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
    thickness = max(4, int(round(base / 200)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.95, min(1.25, (base / 720) * 0.95))
    text_th = max(2, thickness - 1)

    for d in dets:
        x1, y1, x2, y2 = d.get("bbox") or [0, 0, 0, 0]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        color = tuple(d.get("color") or (0, 220, 0))
        label = str(d.get("label") or "")
        conf = float(d.get("conf") or 0.0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {conf:.2f}".strip()
        if not text:
            continue

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_th)
        pad_x = 12
        pad_y = 10
        y_text_top = max(0, y1 - th - baseline - pad_y)
        y_text_bottom = min(h - 1, y_text_top + th + baseline + pad_y)
        x_text_left = max(0, x1)
        x_text_right = min(w - 1, x_text_left + tw + pad_x)

        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), color, -1)
        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), (0, 0, 0), 2)

        org = (x_text_left + 10, y_text_bottom - baseline - 5)
        cv2.putText(frame, text, org, font, font_scale, (255, 255, 255), text_th + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)


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


VIDEO_PATHS: Dict[int, Path] = {}
_zones: Dict[int, ZoneState] = {}
_zone_threads: Dict[int, threading.Thread] = {}



def _open_zone_caps():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    global ZONE_IDS
    ZONE_IDS = _find_zone_ids(VIDEO_DIR)
    for zid in ZONE_IDS:
        p = VIDEO_DIR / f"zone{zid}.mp4"
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
                        if dets:
                            st.last_dets = dets
                            st.last_dets_ts = now_det
                        if alerts:
                            _record_alerts(alerts)
                            _persist_alerts(zid, alerts, frame)
                    else:
                        if st.last_dets and (time.time() - (st.last_dets_ts or 0.0)) <= COASTVISION_DET_HOLD_S:
                            _draw_detections(frame, st.last_dets)

                    ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
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
                                    interpolation=cv2.INTER_AREA,
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

    _zone_threads.pop(zid, None)


# ----------------- API (Flask) -----------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

_workers_started = False
_workers_lock = threading.Lock()


def _start_workers_once():
    global _workers_started
    if _workers_started:
        return
    with _workers_lock:
        if _workers_started:
            return
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
        }
    )


@app.route("/api/zones", methods=["GET"])

def zones():
    # Always rescan for new zone*.mp4 files before reporting zones
    global ZONE_IDS
    ZONE_IDS = _find_zone_ids(VIDEO_DIR)
    for zid in ZONE_IDS:
        if zid not in VIDEO_PATHS:
            p = VIDEO_DIR / f"zone{zid}.mp4"
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
            if zs and zs.cap:
                try:
                    zs.cap.release()
                except Exception:
                    pass
            _zone_threads.pop(zid, None)
    items = []
    now = time.time()
    for zid, p in VIDEO_PATHS.items():
        st = _zones.get(zid)
        items.append(
            {
                "id": zid,
                "path": str(p),
                "exists": bool(p.exists()),
                "active": zid in _zones,
                "is_opened": bool(st.cap.isOpened()) if st else False,
                "last_frame_age_s": (now - st.last_ts) if (st and st.last_ts) else None,
                "last_error": st.last_error if st else None,
            }
        )
    return jsonify({"items": items})

# Optional: endpoint to force reload all videos (for UI button)
@app.route("/api/zones/reload", methods=["POST"])
def reload_zones():
    _open_zone_caps()
    for zid in list(_zones.keys()):
        _ensure_zone_thread(zid)
    return jsonify({"ok": True, "zones": [z for z in ZONE_IDS]})


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


@app.route("/api/zones/<int:zid>/detections", methods=["GET"])
def zone_detections(zid: int):
    st = _zones.get(zid)
    if not st:
        return jsonify({"zone": zid, "count": 0, "age_s": None, "items": []})
    age = (time.time() - st.last_dets_ts) if st.last_dets_ts else None
    items = st.last_dets or []
    return jsonify({"zone": zid, "count": len(items), "age_s": age, "items": items})


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


if __name__ == "__main__":
    # Dev convenience only. For Windows stability and concurrency, prefer:
    #   .\run_backend.ps1  (Waitress + correct venv python)
    host = os.environ.get("COASTVISION_HOST", "127.0.0.1")
    port = int(os.environ.get("COASTVISION_PORT", "8000"))
    print(f"[main] Starting Flask dev server on http://{host}:{port} (device={DEVICE})")
    print("[main] Tip: use run_backend.ps1 for production-like serving.")
    app.run(host=host, port=port, threaded=True)
