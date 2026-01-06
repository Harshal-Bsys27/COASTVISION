"""CoastVision backend: serves annotated zone frames, alerts, and analysis."""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import threading
import time
from typing import Any, Dict, List, Optional
import csv

import cv2
import torch
import numpy as np  # <-- add (used by placeholder jpeg)
from flask import Flask, Response, abort, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO


# ----------------- CONFIG -----------------
ROOT = Path(__file__).resolve().parent
_VIDEO_DIR_ENV = os.environ.get("COASTVISION_VIDEO_DIR", "").strip()
VIDEO_DIR_CANDIDATES = [
    Path(_VIDEO_DIR_ENV) if _VIDEO_DIR_ENV else None,
    ROOT / ".." / "frontend" / "dashboard" / "videos",
    ROOT / ".." / "data" / "raw_videos",
]
ZONE_IDS = list(range(1, 7))
CONF_THRES = float(os.environ.get("COASTVISION_CONF", "0.35"))
PERSON_CONF_THRES = float(os.environ.get("COASTVISION_PERSON_CONF", "0.25"))
COASTVISION_IOU = float(os.environ.get("COASTVISION_IOU", "0.45"))
COASTVISION_MAX_DET = int(os.environ.get("COASTVISION_MAX_DET", "200"))
COASTVISION_ALERT_CONF = float(os.environ.get("COASTVISION_ALERT_CONF", "0.55"))
COASTVISION_MAX_SIDE = int(os.environ.get("COASTVISION_MAX_SIDE", "1280"))
COASTVISION_FPS = int(os.environ.get("COASTVISION_FPS", "10"))
COASTVISION_INFER_EVERY = int(os.environ.get("COASTVISION_INFER_EVERY", "2"))
COASTVISION_IMGSZ = int(os.environ.get("COASTVISION_IMGSZ", "640"))
COASTVISION_DEVICE = os.environ.get("COASTVISION_DEVICE", "").strip()
COASTVISION_ALERT_COOLDOWN_S = float(os.environ.get("COASTVISION_ALERT_COOLDOWN_S", "4"))
COASTVISION_DET_HOLD_S = float(os.environ.get("COASTVISION_DET_HOLD_S", "0.75"))
COASTVISION_OVERLAY_STYLE = os.environ.get("COASTVISION_OVERLAY_STYLE", "pro").strip().lower()
COASTVISION_ENABLE_PERSON_DET = os.environ.get("COASTVISION_ENABLE_PERSON_DET", "1").strip().lower() not in {"0", "false", "no"}
_ALERT_CLASSES_ENV = os.environ.get("COASTVISION_ALERT_CLASSES", "").strip()
ALERT_CLASSES = {s.strip().lower() for s in _ALERT_CLASSES_ENV.split(",") if s.strip()} if _ALERT_CLASSES_ENV else None
ALERT_HISTORY = deque(maxlen=400)

ALERTS_DIR = (ROOT / ".." / "data" / "alerts").resolve()
ALERTS_IMAGES_DIR = (ALERTS_DIR / "images").resolve()
ALERTS_CSV_PATH = (ALERTS_DIR / "alerts.csv").resolve()
_alerts_lock = threading.Lock()


# ----------------- PERF / SAFETY -----------------
def _env_bool(name: str, default: bool) -> bool:
	v = os.environ.get(name, "").strip().lower()
	if not v:
		return default
	return v not in {"0", "false", "no", "off"}

# Prefer stable defaults for 6 zones (2 of them are 4K)
COASTVISION_MAX_SIDE = int(os.environ.get("COASTVISION_MAX_SIDE", "1280"))
COASTVISION_FPS = int(os.environ.get("COASTVISION_FPS", "10"))
COASTVISION_INFER_EVERY = int(os.environ.get("COASTVISION_INFER_EVERY", "2"))
COASTVISION_IMGSZ = int(os.environ.get("COASTVISION_IMGSZ", "640"))
COASTVISION_HALF = _env_bool("COASTVISION_HALF", True)

# One GPU -> avoid concurrent predict() across threads (this is the #1 cause of stalls/crashes)
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
if COASTVISION_DEVICE:
    DEVICE = COASTVISION_DEVICE
else:
    DEVICE = "cuda:0" if _cuda_available else "cpu"
PREDICT_DEVICE = 0 if (str(DEVICE).startswith("cuda") and _cuda_available) else "cpu"
print(f"[init] Loading model from {MODEL_PATH} on {DEVICE} (predict_device={PREDICT_DEVICE})")
MODEL = YOLO(str(MODEL_PATH)).to(DEVICE)
try:
    MODEL.fuse()
except Exception:
    pass

# If the main model doesn't have a 'person' class (common for custom drowning models),
# load a lightweight COCO model to detect each person separately.
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


# ----------------- PLACEHOLDER JPEG -----------------
def _placeholder_jpeg(title: str, subtitle: str = "") -> bytes:
    # IMPORTANT: keep this dead-simple; it must NEVER throw
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, title, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (46, 230, 255), 3, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 190, 220), 2, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return jpg.tobytes() if ok else b""


# ----------------- STATE -----------------
@dataclass
class ZoneState:
    zid: int
    path: Path
    cap: cv2.VideoCapture
    lock: threading.Lock
    last_jpeg: Optional[bytes] = None
    last_ts: float = 0.0
    last_error: Optional[str] = None
    frame_i: int = 0
    last_alert_time_s: float = 0.0
    last_dets: Optional[List[Dict[str, Any]]] = None
    last_dets_ts: float = 0.0


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

        # Outline + main stroke for better legibility
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

        # label background SAME as box color + black border
        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), color, -1)
        cv2.rectangle(frame, (x_text_left, y_text_top), (x_text_right, y_text_bottom), (0, 0, 0), 2)

        org = (x_text_left + 10, y_text_bottom - baseline - 5)
        # outlined text (white underlay + black foreground) for colored background
        cv2.putText(frame, text, org, font, font_scale, (255, 255, 255), text_th + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, font_scale, (0, 0, 0), text_th, cv2.LINE_AA)


VIDEO_PATHS: Dict[int, Path] = {}
_zones: Dict[int, ZoneState] = {}


def _find_video_dir() -> Path:
    for candidate in VIDEO_DIR_CANDIDATES:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate.resolve()
    # fallback to ROOT
    return ROOT / "videos"


VIDEO_DIR = _find_video_dir()
print(f"[init] Using video directory: {VIDEO_DIR}")


def _open_capture(path: Path) -> Optional[cv2.VideoCapture]:
	# FFMPEG path first (more reliable for mp4 on Windows)
	cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
	if cap.isOpened():
		try:
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		except Exception:
			pass
		return cap
	cap.release()
	cap = cv2.VideoCapture(str(path))
	if cap.isOpened():
		try:
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		except Exception:
			pass
		return cap
	cap.release()
	return None


def _open_zone_caps():
    for zid in ZONE_IDS:
        p = VIDEO_DIR / f"zone{zid}.mp4"
        VIDEO_PATHS[zid] = p
        if not p.exists():
            print(f"[warn] Missing video for zone {zid}: {p}")
            continue
        cap = _open_capture(p)
        if not cap:
            print(f"[warn] Could not open video for zone {zid}: {p}")
            continue
        _zones[zid] = ZoneState(zid=zid, path=p, cap=cap, lock=threading.Lock())


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
        with ALERTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "ts_utc", "zone", "label", "conf", "x1", "y1", "x2", "y2", "image_path"])


def _append_alert_row(row):
    with _alerts_lock:
        _ensure_alert_dirs()
        with ALERTS_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row)


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
					conf = float(box.conf[0]) if box.conf is not None else 0.0
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					dets.append({
						"bbox": [x1, y1, x2, y2],
						"label": "person",
						"conf": conf,
						"color": (0, 220, 0),
					})
		except Exception:
			# don't break video if person model fails
			pass

	# 2) Main model detections (drowning/emergency etc)
	for r in results:
		for box in r.boxes:
			cls = int(box.cls[0]) if box.cls is not None else -1
			conf = float(box.conf[0]) if box.conf is not None else 0.0
			label = names.get(cls, f"class_{cls}") if isinstance(names, dict) else str(cls)
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			x1 = max(0, min(w - 1, x1))
			y1 = max(0, min(h - 1, y1))
			x2 = max(0, min(w - 1, x2))
			y2 = max(0, min(h - 1, y2))

			# High-contrast styling for readability
			label_l = str(label).lower()
			# Requested style: green for normal, orange for drowning/emergency.
			if COASTVISION_OVERLAY_STYLE in {"green", "pro"}:
				color = (0, 220, 0)  # bright green (BGR)
				if "drown" in label_l or "emerg" in label_l:
					color = (0, 165, 255)  # orange (BGR)
			else:
				# fallback
				color = (0, 220, 0)

			det = {"bbox": [x1, y1, x2, y2], "label": label, "conf": conf, "color": color}
            dets.append(det)
            # For project context: keep overlays flexible, but make ALERTS stricter to reduce false alarms.
            if conf >= COASTVISION_ALERT_CONF and ((ALERT_CLASSES is None) or (str(label).lower() in ALERT_CLASSES)):
                alerts.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "zone": zid,
                    "label": label,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "msg": f"{label} detected",
                })
	_draw_detections(frame, dets)
	return frame, alerts, dets


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

    # write one CSV row per box
    for a in alerts:
        x1, y1, x2, y2 = (a.get("bbox") or [None, None, None, None])
        event_id = f"{ts_str}_z{zid}_{a.get('label','')}_{int((a.get('conf') or 0)*1000)}"
        _append_alert_row([
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
        ])
        a["event_id"] = event_id
        if image_path:
            a["image_path"] = str(image_path)

    st.last_alert_time_s = now_s


def _zone_worker(zid: int, fps: int = COASTVISION_FPS):
    interval = 1.0 / max(1, fps)
    st = _zones[zid]
    while True:
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
                        # IMPORTANT: do not wipe cached dets when the model misses on a frame.
                        if dets:
                            st.last_dets = dets
                            st.last_dets_ts = now_det
                        if alerts:
                            _record_alerts(alerts)
                            _persist_alerts(zid, alerts, frame)
                    else:
                        # Reduce flicker: keep drawing last detections for a short hold window.
                        if st.last_dets and (time.time() - (st.last_dets_ts or 0.0)) <= COASTVISION_DET_HOLD_S:
                            _draw_detections(frame, st.last_dets)
                    ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok2:
                        st.last_jpeg = jpg.tobytes()
                        st.last_ts = time.time()
                        st.last_error = None
                else:
                    st.last_error = "read_failed"
        except Exception as e:
            st.last_error = f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        time.sleep(max(0.0, interval - dt))


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
            th = threading.Thread(target=_zone_worker, args=(zid,), daemon=True)
            th.start()
            print(f"[init] Started worker for zone {zid}")
        _workers_started = True


@app.before_request
def _ensure_started():
    # Ensure video readers + workers are running before serving endpoints.
    _start_workers_once()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "model_path": str(MODEL_PATH),
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
    })


@app.route("/api/zones", methods=["GET"])
def zones():
    items = []
    now = time.time()
    for zid, p in VIDEO_PATHS.items():
        st = _zones.get(zid)
        items.append({
            "id": zid,
            "path": str(p),
            "exists": bool(p.exists()),
            "active": zid in _zones,
            "is_opened": bool(st.cap.isOpened()) if st else False,
            "last_frame_age_s": (now - st.last_ts) if (st and st.last_ts) else None,
            "last_error": st.last_error if st else None,
        })
    return jsonify({"items": items})


@app.route("/api/zones/<int:zid>/frame.jpg", methods=["GET"])
def zone_frame(zid: int):
	st = _zones.get(zid)
	if not st:
		abort(404, description="Zone not available")

	if st.last_jpeg is None:
		return Response(
			_placeholder_jpeg(f"Zone {zid}", "Loading frames..."),
			mimetype="image/jpeg",
			headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
		)

	return Response(
		st.last_jpeg,
		mimetype="image/jpeg",
		headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
	)


@app.route("/api/zones/<int:zid>/stream.mjpg", methods=["GET"])
def zone_stream(zid: int):
	st = _zones.get(zid)
	if not st:
		abort(404, description="Zone not available")

	boundary = "frame"
	frame_interval_s = 1.0 / max(1, COASTVISION_FPS)

	def gen():
		last_sent = None
		while True:
			jpg = st.last_jpeg
			if not jpg:
				ph = _placeholder_jpeg(f"Zone {zid}", "Loading frames...")
				yield (
					f"--{boundary}\r\n"
					"Content-Type: image/jpeg\r\n"
					f"Content-Length: {len(ph)}\r\n\r\n"
				).encode("utf-8") + ph + b"\r\n"
				time.sleep(frame_interval_s)
				continue

			if jpg is not last_sent:
				last_sent = jpg
				yield (
					f"--{boundary}\r\n"
					"Content-Type: image/jpeg\r\n"
					f"Content-Length: {len(jpg)}\r\n\r\n"
				).encode("utf-8") + jpg + b"\r\n"
			time.sleep(frame_interval_s)

	return Response(
		gen(),
		mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
		headers={"Cache-Control": "no-store", "Pragma": "no-cache", "Expires": "0"},
	)


@app.route("/api/zones/<int:zid>/detections", methods=["GET"])
def zone_detections(zid: int):
    st = _zones.get(zid)
    if not st:
        abort(404, description="Zone not available")
    now = time.time()
    age_s = (now - st.last_dets_ts) if st.last_dets_ts else None
    items = st.last_dets or []
    # Only return cached detections if they are recent, otherwise return empty.
    if age_s is None or age_s > max(0.1, COASTVISION_DET_HOLD_S * 2.5):
        items = []
    return jsonify({"zone": zid, "count": len(items), "age_s": age_s, "items": items})


@app.route("/api/alerts", methods=["GET"])
def alerts():
    limit = int(request.args.get("limit", "100"))
    zone = request.args.get("zone", "").strip()
    if zone:
        try:
            zid = int(zone)
        except Exception:
            zid = None
    else:
        zid = None

    items = list(ALERT_HISTORY)
    if zid is not None:
        items = [a for a in items if a.get("zone") == zid]
    limit = max(1, min(limit, len(items)))
    return jsonify({"items": items[:limit]})


@app.route("/api/analysis", methods=["GET"])
def analysis():
    zone = request.args.get("zone", "").strip()
    if zone:
        try:
            zid = int(zone)
        except Exception:
            zid = None
    else:
        zid = None

    by_zone = Counter()
    by_label = Counter()
    items = list(ALERT_HISTORY)
    if zid is not None:
        items = [a for a in items if a.get("zone") == zid]

    for a in items:
        by_zone[a["zone"]] += 1
        by_label[a["label"]] += 1
    return jsonify({
        "alerts_total": len(items),
        "alerts_by_zone": dict(by_zone),
        "alerts_by_label": dict(by_label),
        "device": str(DEVICE),
        "predict_device": str(PREDICT_DEVICE),
        "imgsz": COASTVISION_IMGSZ,
        "fps": COASTVISION_FPS,
        "infer_every": COASTVISION_INFER_EVERY,
        "alerts_csv": str(ALERTS_CSV_PATH),
    })


if __name__ == "__main__":
    _start_workers_once()
    # Use threaded server so multiple zone streams can be served concurrently.
    # Note: For production on Windows, prefer running behind waitress.
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)