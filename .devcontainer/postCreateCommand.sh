#!/bin/bash
set -e

echo "================================================"
echo "Setting up CoastVision Development Environment"
echo "================================================"

# Install system dependencies for OpenCV
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend/web
npm install
cd ../..

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models
mkdir -p data/alerts/images
mkdir -p data/raw_videos
mkdir -p frontend/dashboard/videos
mkdir -p frontend/dashboard/models

# Download a lightweight YOLO model for demo (CPU-friendly)
echo "Downloading YOLOv8 nano model for CPU usage..."
if [ ! -f models/yolov8n.pt ]; then
    python -c "
from ultralytics import YOLO
import shutil
model = YOLO('yolov8n.pt')
shutil.move('yolov8n.pt', 'models/yolov8n.pt')
"
fi

# Copy model to frontend dashboard if it doesn't exist
if [ ! -f frontend/dashboard/models/yolov8n.pt ] && [ -f models/yolov8n.pt ]; then
    cp models/yolov8n.pt frontend/dashboard/models/yolov8n.pt
fi

# Create sample video placeholder info
cat > data/raw_videos/README.md << 'EOF'
# Video Files

Place your zone video files here:
- zone1.mp4
- zone2.mp4
- zone3.mp4
- zone4.mp4
- zone5.mp4
- zone6.mp4

For testing in Codespaces without video files, the backend will show placeholder frames.

You can also use sample videos or webcam feeds for testing.
EOF

# Create startup scripts
echo "Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "Starting CoastVision Backend..."
echo "Backend will be available at http://localhost:8000"
echo "API Health check: http://localhost:8000/api/health"
export COASTVISION_DEVICE="cpu"
export COASTVISION_MAX_SIDE="640"
export COASTVISION_FPS="5"
export COASTVISION_INFER_EVERY="3"
export COASTVISION_IMGSZ="640"
export COASTVISION_HALF="0"
python backend/server.py
EOF

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "Starting CoastVision Frontend..."
echo "Frontend will be available at http://localhost:5173"
cd frontend/web
npm run dev
EOF

# Combined startup script
cat > start_all.sh << 'EOF'
#!/bin/bash
echo "Starting CoastVision (Backend + Frontend)..."
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Starting backend in background..."
bash start_backend.sh &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Waiting 5 seconds for backend to initialize..."
sleep 5

echo "Starting frontend..."
bash start_frontend.sh
EOF

chmod +x start_backend.sh start_frontend.sh start_all.sh

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Quick Start Commands:"
echo "  - Start backend only:  ./start_backend.sh"
echo "  - Start frontend only: ./start_frontend.sh"
echo "  - Start both:          ./start_all.sh"
echo ""
echo "Manual Commands:"
echo "  Backend:  python backend/server.py"
echo "  Frontend: cd frontend/web && npm run dev"
echo ""
echo "Note: Running in CPU mode (no GPU in Codespaces)"
echo "================================================"
