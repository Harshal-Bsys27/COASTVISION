# GitHub Codespaces Setup Validation

This document helps verify that your Codespaces environment is set up correctly.

## Automated Checks

Run this validation script after Codespaces setup completes:

```bash
#!/bin/bash
echo "======================================"
echo "CoastVision Environment Validation"
echo "======================================"
echo ""

# Check Python
echo "✓ Checking Python..."
python --version
echo ""

# Check Node.js
echo "✓ Checking Node.js..."
node --version
npm --version
echo ""

# Check Python dependencies
echo "✓ Checking Python dependencies..."
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
echo ""

# Check directories
echo "✓ Checking directories..."
test -d models && echo "models/ exists" || echo "❌ models/ missing"
test -d data/alerts/images && echo "data/alerts/images/ exists" || echo "❌ data/alerts/images/ missing"
test -d frontend/dashboard/videos && echo "frontend/dashboard/videos/ exists" || echo "❌ frontend/dashboard/videos/ missing"
echo ""

# Check model
echo "✓ Checking YOLO model..."
test -f models/yolov8n.pt && echo "models/yolov8n.pt exists" || echo "❌ models/yolov8n.pt missing"
echo ""

# Check startup scripts
echo "✓ Checking startup scripts..."
test -x start_backend.sh && echo "start_backend.sh is executable" || echo "❌ start_backend.sh not executable"
test -x start_frontend.sh && echo "start_frontend.sh is executable" || echo "❌ start_frontend.sh not executable"
test -x start_all.sh && echo "start_all.sh is executable" || echo "❌ start_all.sh not executable"
echo ""

# Check frontend dependencies
echo "✓ Checking frontend dependencies..."
test -d frontend/web/node_modules && echo "node_modules installed" || echo "❌ node_modules missing"
echo ""

echo "======================================"
echo "Validation Complete!"
echo "======================================"
```

## Manual Checks

### 1. Python Environment
```bash
python --version  # Should be 3.11.x
pip list | grep -E "(torch|opencv|ultralytics|flask)"
```

### 2. Node.js Environment
```bash
node --version  # Should be 20.x
npm --version
cd frontend/web && npm list
```

### 3. Test Backend
```bash
# In one terminal
export COASTVISION_DEVICE="cpu"
python backend/server.py
```

Then in another terminal or browser:
```bash
curl http://localhost:8000/api/health
```

Should return JSON with status "ok".

### 4. Test Frontend
```bash
cd frontend/web
npm run dev
```

Should start Vite dev server on port 5173.

## Common Issues

### Issue: "Module not found" errors
**Solution**: Re-run the setup script
```bash
bash .devcontainer/postCreateCommand.sh
```

### Issue: Port already in use
**Solution**: Kill existing processes
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

### Issue: Model not found
**Solution**: Download manually
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

### Issue: Frontend won't start
**Solution**: Reinstall dependencies
```bash
cd frontend/web
rm -rf node_modules package-lock.json
npm install
```

### Issue: Backend crashes on startup
**Solution**: Check environment variables
```bash
export COASTVISION_DEVICE="cpu"
export COASTVISION_HALF="0"
python backend/server.py
```

## Expected Behavior

### After Setup Completes
- ✅ All Python packages installed
- ✅ All npm packages installed
- ✅ YOLOv8n model downloaded to `models/`
- ✅ Startup scripts created and executable
- ✅ Directories created

### When Running Backend
- ✅ Starts on port 8000
- ✅ Health endpoint responds: `http://localhost:8000/api/health`
- ✅ Console shows: "Model running on: cpu"
- ✅ Zones endpoint accessible: `http://localhost:8000/api/zones`

### When Running Frontend
- ✅ Vite dev server starts on port 5173
- ✅ React app loads in browser
- ✅ Can see zone grid interface
- ✅ No console errors

## Performance Expectations

In Codespaces (CPU-only):
- Backend starts in ~5-10 seconds
- Frontend starts in ~3-5 seconds
- Frame processing: ~0.5-1 FPS per zone (optimized for CPU)
- Model inference: ~200-500ms per frame

## Quick Validation Command

Copy and run this one-liner:
```bash
python --version && node --version && test -f models/yolov8n.pt && echo "✅ Environment ready!" || echo "❌ Setup incomplete"
```

## Getting Help

If validation fails:
1. Check the [Troubleshooting section](codespaces_guide.md#troubleshooting) in the guide
2. Review setup logs in the terminal
3. Re-run postCreateCommand.sh
4. Open an issue on GitHub with error logs
