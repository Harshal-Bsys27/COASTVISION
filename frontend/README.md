# Frontend - PyQt Dashboard

PyQt-based 6-zone monitor that plays prototype videos and highlights people (YOLO) with a simple immobility check for possible drowning. Zoomable per-zone popup (mouse/touchpad), pause/play, and alert log.

## Structure
```
frontend/
  dashboard/
    main_dashboard.py   # entry point
    models/             # best.pt preferred; falls back to yolov8n.pt
    videos/             # zone1.mp4 ... zone6.mp4
    alerts.csv          # rolling alert log
  legacy_te_proj/       # archived original files (kept for reference)
  requirements.txt
```

## Setup
```powershell
cd "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\frontend"
pip install -r requirements.txt
```

## Run (GPU-aware)
```powershell
# activate venv first if you use one
& "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\venv\Scripts\Activate.ps1"
cd "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\frontend"
python dashboard\main_dashboard.py
```
- Console should print `Model running on: CUDA` when CUDA torch is installed (e.g., torch 2.2.2+cu121).
- In the UI header, GPU name will be shown when running on CUDA.

## Models
- Drop your trained `best.pt` into `frontend/dashboard/models/` (auto-picked). If absent, it uses `yolov8n.pt` in the same folder.
- Root `models/` can store checkpoints; copy the one you want into `frontend/dashboard/models/`.

## Videos
- Put `zone1.mp4` ... `zone6.mp4` under `frontend/dashboard/videos/`.
- Restart the app after replacing videos.
