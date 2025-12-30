# CoastVision: AI-Powered Beach Surveillance System

## Project Overview
CoastVision detects people in water, flags possible drowning behavior, and surfaces alerts for lifeguards. A PyQt desktop dashboard shows six zones with zoomable popouts and alerts. Future work: Android app for lifeguard notifications.

## Objectives
- Detect humans entering water zones from camera feeds.
- Flag potential distress (immobility heuristic for now; extendable to richer behavior models).
- Alert with zone context and maintain a rolling log.
- Enable heatmap/analytics for risky areas (planned).

## Tech Stack
- Python, PyTorch, Ultralytics YOLOv8/YOLOv5, OpenCV
- PyQt dashboard (current), Flask/Streamlit legacy prototype
- Roboflow / LabelImg for annotation
- Training on Colab or local GPU (RTX-class recommended)

## Getting Started
```bash
git clone https://github.com/yourusername/coastvision.git
pip install -r requirements.txt
```

## Training (YOLOv8)
- Local: run `scripts/train_yolov8.py`; copy the resulting `best.pt` into `models/` (root) and into `frontend/dashboard/models/` for the UI.
- Colab: follow [docs/colab_training_full_example.md](docs/colab_training_full_example.md) (short guides: [docs/colab_training.md](docs/colab_training.md), [docs/colab_training_with_auto_backup.md](docs/colab_training_with_auto_backup.md)).

## Dashboard (PyQt, GPU-aware)
```powershell
# activate venv if you use one
& "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\venv\Scripts\Activate.ps1"
cd "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\frontend"
pip install -r requirements.txt
python dashboard\main_dashboard.py
```
- Videos: place `zone1.mp4` … `zone6.mp4` in `frontend/dashboard/videos/`.
- Model: auto-loads `frontend/dashboard/models/best.pt`, else `yolov8n.pt`.
- GPU: use a CUDA Torch build (e.g., torch 2.2.2+cu121); console will show `Model running on: CUDA`.

## Folder Structure
- `/backend` — backend API/streaming (if used)
- `/frontend` — PyQt dashboard (`dashboard/`), legacy prototype (`legacy_te_proj/`)
- `/models` — trained weights storage
- `/data` and `/dataset` — raw/annotated data (not tracked in git)
- `/docs` — guides (Colab training, project notes)
- `/scripts` — training/inference utilities

## License
MIT License (add LICENSE file as needed)


