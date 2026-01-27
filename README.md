# CoastVision: AI-Powered Beach Surveillance System
( Currently In Progress )

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

## GPU setup (RTX 3050 6GB, Windows) — do this exactly

### 0) NVIDIA driver check (required)
1. Install latest NVIDIA driver (Game Ready / Studio), reboot.
2. Verify driver works:
   ```powershell
   nvidia-smi
   ```
   If this fails, CUDA PyTorch will not work (fix driver first).

### 1) Activate the SAME Python/venv used by the backend
```powershell
cd "c:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION"
& ".\venv\Scripts\Activate.ps1"
where python
python -m pip --version
```

### 2) Install project deps, then install CUDA PyTorch (important order)
If your `requirements.txt` installs torch (CPU), it can overwrite CUDA torch. So do:

```powershell
# Install everything else first
python -m pip install -r requirements.txt

# Then force CUDA torch install in THIS SAME venv
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 3) Verify CUDA PyTorch is actually working
```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_ver=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```
Expected:
- `cuda_available= True`
- `gpu= NVIDIA GeForce RTX 3050 ...`

### 4) Run backend and FAIL FAST if it falls back to CPU
```powershell
$env:COASTVISION_DEVICE="cuda:0"
$env:COASTVISION_REQUIRE_CUDA="1"

# perf
$env:COASTVISION_HALF="1"
$env:COASTVISION_TF32="1"
$env:COASTVISION_CUDNN_BENCHMARK="1"

# smooth multi-zone defaults
$env:COASTVISION_MAX_SIDE="1280"
$env:COASTVISION_FPS="10"
$env:COASTVISION_INFER_EVERY="2"
$env:COASTVISION_IMGSZ="640"

python backend\server.py
```

### 5) Confirm from the API
Open:
- http://127.0.0.1:8000/api/health

Must show:
- `requested_device = "cuda:0"`
- `device = "cuda:0"`
- `torch_cuda_available = true`
- `cuda_smoke_ok = true`

If `cuda_smoke_ok=false`, read `cuda_smoke_error` — it tells you exactly what’s wrong.

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

## If backend still runs on CPU (most common causes)
### 1) You installed CUDA torch into a different Python than the backend uses
Run these in the SAME terminal before starting the backend:

```powershell
# verify which python/pip you are using
where python
python -m pip --version
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_ver=', torch.version.cuda)"
```

If `cuda_available` is False, install CUDA torch in this exact environment:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 2) Force CUDA and fail fast (recommended for debugging)
```powershell
$env:COASTVISION_DEVICE="cuda:0"
$env:COASTVISION_REQUIRE_CUDA="1"
python backend\server.py
```

Then check:
- http://127.0.0.1:8000/api/health
- Look at: `device`, `torch_cuda_available`, `cuda_smoke_ok`, `cuda_smoke_error`

## Important: venv name must match what you actually run
If you activate `.venv` but earlier installed CUDA torch in `venv` (or vice-versa), the backend will run CPU.

Use whichever exists on your machine:

```powershell
# Option A
& ".\venv\Scripts\Activate.ps1"

# Option B
& ".\.venv\Scripts\Activate.ps1"
```

Then verify this exact python has CUDA torch:
```powershell
where python
python -c "import torch; print('torch=',torch.__version__); print('torch_built_with_cuda=', torch.version.cuda is not None); print('cuda_available=', torch.cuda.is_available()); print('cuda_ver=', torch.version.cuda)"
```

Expected:
- `torch_built_with_cuda = True`
- `cuda_available = True`

If `torch_built_with_cuda=False`, you installed CPU-only torch in that environment.

## Folder Structure
- `/backend` — backend API/streaming (if used)
- `/frontend` — PyQt dashboard (`dashboard/`), legacy prototype (`legacy_te_proj/`)
- `/models` — trained weights storage
- `/data` and `/dataset` — raw/annotated data (not tracked in git)
- `/docs` — guides (Colab training, project notes)
- `/scripts` — training/inference utilities

## License
MIT License (add LICENSE file as needed)


