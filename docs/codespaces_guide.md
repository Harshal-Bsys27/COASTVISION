# GitHub Codespaces Setup Guide

This guide explains how to run CoastVision in GitHub Codespaces with **zero local setup**.

## What is GitHub Codespaces?

GitHub Codespaces provides a complete development environment in your browser. No need to install Python, Node.js, or any dependencies on your local machine.

## Quick Start

### 1. Launch Codespace

1. Go to the GitHub repository
2. Click the green **Code** button
3. Click the **Codespaces** tab
4. Click **Create codespace on main** (or select your branch)

### 2. Wait for Setup

The first time you create a codespace, it will:
- Build the container (2-3 minutes)
- Install Python dependencies
- Install Node.js dependencies
- Download YOLOv8n model
- Create necessary directories
- Set up startup scripts

You'll see the progress in the terminal. When you see "✅ Setup Complete!", you're ready to go!

### 3. Start the Application

Run the all-in-one startup script:
```bash
./start_all.sh
```

This starts both the backend and frontend servers.

**Alternative: Start services separately**
```bash
# Terminal 1: Start backend
./start_backend.sh

# Terminal 2: Start frontend (in a new terminal)
./start_frontend.sh
```

### 4. Access the Application

VS Code will show port forwarding notifications. Click to open:
- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Health**: http://localhost:8000/api/health

You can also view forwarded ports in the **PORTS** tab at the bottom of VS Code.

## Features in Codespaces

### ✅ Pre-configured Environment
- Python 3.11 with all dependencies
- Node.js 20 with npm
- OpenCV system dependencies
- YOLOv8 model pre-downloaded

### ✅ CPU-Optimized Settings
The environment is configured for CPU inference:
```bash
COASTVISION_DEVICE="cpu"
COASTVISION_MAX_SIDE="640"
COASTVISION_FPS="5"
COASTVISION_INFER_EVERY="3"
COASTVISION_IMGSZ="640"
COASTVISION_HALF="0"
```

### ✅ VS Code Extensions
Pre-installed extensions:
- Python + Pylance
- ESLint + Prettier
- TypeScript
- Jupyter

### ✅ Port Forwarding
Automatic forwarding for:
- Port 8000: Backend API
- Port 5173: Frontend Dev Server

## Adding Video Files

To test with real video files:

1. Upload videos to `frontend/dashboard/videos/`:
   - zone1.mp4
   - zone2.mp4
   - zone3.mp4
   - zone4.mp4
   - zone5.mp4
   - zone6.mp4

2. Restart the backend:
   ```bash
   # Stop the backend (Ctrl+C)
   ./start_backend.sh
   ```

**Tip**: You can upload files by:
- Dragging and dropping into VS Code file explorer
- Using the Upload Files option in the file explorer
- Using terminal commands: `wget`, `curl`, etc.

## Environment Variables

The backend startup script sets these environment variables for Codespaces:

| Variable | Value | Description |
|----------|-------|-------------|
| `COASTVISION_DEVICE` | `cpu` | Use CPU for inference |
| `COASTVISION_MAX_SIDE` | `640` | Resize frames to max 640px |
| `COASTVISION_FPS` | `5` | Process 5 frames per second |
| `COASTVISION_INFER_EVERY` | `3` | Run inference every 3rd frame |
| `COASTVISION_IMGSZ` | `640` | YOLO input image size |
| `COASTVISION_HALF` | `0` | Disable FP16 (CPU doesn't support) |

You can modify these in `start_backend.sh` if needed.

## API Endpoints

Once the backend is running, try these endpoints:

```bash
# Health check
curl http://localhost:8000/api/health

# List zones
curl http://localhost:8000/api/zones

# Get zone frame
curl http://localhost:8000/api/zones/1/frame.jpg --output zone1.jpg

# Get detections for zone 1
curl http://localhost:8000/api/zones/1/detections

# Get recent alerts
curl http://localhost:8000/api/alerts

# Get analysis
curl http://localhost:8000/api/analysis
```

## Manual Commands

If you prefer to run commands manually:

### Backend
```bash
export COASTVISION_DEVICE="cpu"
export COASTVISION_MAX_SIDE="640"
export COASTVISION_FPS="5"
python backend/server.py
```

### Frontend
```bash
cd frontend/web
npm run dev
```

## Troubleshooting

### Port Already in Use
If you see "port already in use" error:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:5173 | xargs kill -9  # Frontend
```

### Model Not Found
If the model isn't downloaded:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

### Dependencies Missing
Re-run the setup:
```bash
bash .devcontainer/postCreateCommand.sh
```

### Codespace Performance
For better performance:
- Use a larger machine type (4-core or 8-core)
- Reduce FPS in `start_backend.sh`
- Increase `COASTVISION_INFER_EVERY` to skip more frames

## Development Workflow

### Making Changes

1. Edit code in VS Code
2. Changes are automatically saved
3. Backend: Restart the backend script
4. Frontend: Vite will hot-reload automatically

### Testing

```bash
# Backend health check
curl http://localhost:8000/api/health

# Frontend (open in browser)
http://localhost:5173
```

### Committing Changes

```bash
git add .
git commit -m "Your message"
git push
```

## Codespaces Limits

GitHub provides free Codespaces hours:
- **Free tier**: 120 core-hours/month
- 2-core machine: 60 hours/month
- 4-core machine: 30 hours/month

**Tip**: Stop your codespace when not in use to save hours!

## Stopping Codespace

### Stop Services
Press `Ctrl+C` in the terminal running the servers.

### Stop Codespace
- Click the Codespaces icon in bottom-left corner
- Select "Stop Current Codespace"
- Or: Go to GitHub → Codespaces → Stop

### Delete Codespace
- Go to GitHub → Your repositories
- Click "Code" → "Codespaces" tab
- Click "..." → "Delete"

## Advanced Configuration

### Custom Codespace Settings

Edit `.devcontainer/devcontainer.json` to:
- Change base image
- Add more VS Code extensions
- Modify port forwarding
- Add prebuild commands

### Environment Customization

Edit `.devcontainer/postCreateCommand.sh` to:
- Install additional packages
- Download different models
- Configure custom settings

## Next Steps

- **Add your own videos**: Upload to `frontend/dashboard/videos/`
- **Train custom models**: See [docs/colab_training.md](colab_training.md)
- **Customize detection**: Modify environment variables
- **Develop features**: Edit code and test in real-time

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. View logs in the terminal
3. Open an issue on GitHub
4. Check Codespaces logs: Command Palette → "Codespaces: View Creation Log"

## Resources

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [CoastVision Main README](../README.md)
- [Colab Training Guide](colab_training_full_example.md)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
