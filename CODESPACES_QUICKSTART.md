# 🚀 Run CoastVision in GitHub Codespaces - Zero Setup!

Want to run this project **right now** without installing anything? Use GitHub Codespaces!

## Quick Start (2 Steps)

### 1. Create Codespace
- Click the green **Code** button above
- Select **Codespaces** tab
- Click **Create codespace on main**

### 2. Run the App
Wait for setup to complete (3-5 minutes), then run:
```bash
./start_all.sh
```

**That's it!** The app will start automatically. Click the port notifications to open:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## What Gets Installed Automatically?
- ✅ Python 3.11 + all dependencies
- ✅ Node.js 20 + npm packages
- ✅ YOLOv8 model (CPU-optimized)
- ✅ OpenCV and system dependencies
- ✅ All necessary directories
- ✅ Startup scripts

## No GPU? No Problem!
Codespaces runs on CPU, and we've optimized everything for CPU inference:
- Smaller image sizes (640px)
- Lower FPS (5 fps)
- Frame skipping for efficiency
- CPU-friendly YOLO model

## Want More Details?
See the full guide: [docs/codespaces_guide.md](docs/codespaces_guide.md)

## Prefer Local Setup?
Check the main [README.md](README.md) for local installation instructions.

---

**Free Tier**: GitHub provides 60 hours/month of Codespaces for free (2-core machine).
