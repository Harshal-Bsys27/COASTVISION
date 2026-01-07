# CoastVision: AI-Powered Beach Surveillance System

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Harshal-Bsys27/COASTVISION?quickstart=1)

**Quick Start**: Click the badge above to run this project instantly in your browser with zero setup!

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

## 🚀 Quick Start with GitHub Codespaces (No Setup Required!)

**Run this project instantly in your browser without any local setup:**

1. Click the **Code** button on GitHub
2. Select **Codespaces** tab
3. Click **Create codespace on main** (or your branch)
4. Wait for the environment to build (3-5 minutes first time)
5. Once ready, run one command:
   ```bash
   ./start_all.sh
   ```
6. Open the forwarded ports (VS Code will notify you):
   - Backend API: `http://localhost:8000`
   - Frontend Dashboard: `http://localhost:5173`

**That's it! No dependencies, no Python setup, no Node.js installation needed!**

### Codespaces Features
- ✅ **Zero Setup**: Pre-configured with Python, Node.js, and all dependencies
- ✅ **CPU Mode**: Optimized for CPU inference (no GPU required)
- ✅ **Auto-Install**: Dependencies installed automatically
- ✅ **Sample Model**: YOLOv8n model downloaded automatically
- ✅ **Port Forwarding**: Automatic port forwarding for backend and frontend
- ✅ **VS Code Extensions**: Pre-installed Python and JavaScript extensions

### Codespaces Commands
```bash
# Start both backend and frontend
./start_all.sh

# Or start them separately:
./start_backend.sh    # Backend only (port 8000)
./start_frontend.sh   # Frontend only (port 5173)
```

### Codespaces Notes
- The environment runs in **CPU mode** (Codespaces doesn't provide GPU)
- Performance is optimized for CPU inference
- Sample video files can be placed in `frontend/dashboard/videos/`
- Without videos, placeholder frames are shown
- Free tier: 60 hours/month for 2-core instances

---

## Getting Started (Local Development)
```bash
git clone https://github.com/yourusername/coastvision.git
pip install -r requirements.txt
```

## Training (YOLOv8)
- Local: run `scripts/train_yolov8.py`; copy the resulting `best.pt` into `models/` (root) and into `frontend/dashboard/models/` for the UI.
- Colab: follow [docs/colab_training_full_example.md](docs/colab_training_full_example.md) (short guides: [docs/colab_training.md](docs/colab_training.md), [docs/colab_training_with_auto_backup.md](docs/colab_training_with_auto_backup.md)).

## Local Training (YOLOv8 Drowning Detection)

1. Activate your virtual environment:
    ```powershell
    & "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\venv\Scripts\Activate.ps1"
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```bash
    python scripts/train_yolov8.py
    ```

4. After training completes, copy the trained model:
    ```bash
    copy runs\detect\trainX\weights\best.pt models\best.pt
    ```

5. Use `models/best.pt` for inference or dashboard integration.

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

## Performance (RTX 3050 6GB recommended settings)
If you use 4K videos (zone1/zone2), run backend with:

```powershell
$env:COASTVISION_MAX_SIDE="1280"
$env:COASTVISION_FPS="10"
$env:COASTVISION_INFER_EVERY="2"
$env:COASTVISION_IMGSZ="640"
$env:COASTVISION_HALF="1"
python backend\server.py
```

Notes:
- Backend serializes GPU inference across zones for stability.
- Frontend uses MJPEG streaming; if a stream fails, it auto-falls back to `frame.jpg` polling.

## Folder Structure
- `/backend` — backend API/streaming (if used)
- `/frontend` — PyQt dashboard (`dashboard/`), legacy prototype (`legacy_te_proj/`)
- `/models` — trained weights storage
- `/data` and `/dataset` — raw/annotated data (not tracked in git)
- `/docs` — guides (Colab training, project notes)
- `/scripts` — training/inference utilities

## License
MIT License (add LICENSE file as needed)


