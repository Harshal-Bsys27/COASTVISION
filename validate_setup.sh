#!/bin/bash
# CoastVision Environment Validation Script
# Run this after Codespaces setup to verify everything is configured correctly

set +e  # Don't exit on errors, we want to see all checks

echo "======================================"
echo "CoastVision Environment Validation"
echo "======================================"
echo ""

ERRORS=0

# Check Python
echo "🔍 Checking Python..."
if python --version 2>&1 | grep -q "3\."; then
    echo "✅ Python: $(python --version)"
else
    echo "❌ Python not found or wrong version"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check Node.js
echo "🔍 Checking Node.js..."
if node --version 2>&1 | grep -q "v"; then
    echo "✅ Node.js: $(node --version)"
    echo "✅ npm: $(npm --version)"
else
    echo "❌ Node.js not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
if python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" 2>&1; then
    :
else
    echo "❌ OpenCV not installed"
    ERRORS=$((ERRORS + 1))
fi

if python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>&1; then
    :
else
    echo "❌ PyTorch not installed"
    ERRORS=$((ERRORS + 1))
fi

if python -c "from ultralytics import YOLO; print('✅ Ultralytics: OK')" 2>&1; then
    :
else
    echo "❌ Ultralytics not installed"
    ERRORS=$((ERRORS + 1))
fi

if python -c "import flask; print(f'✅ Flask: {flask.__version__}')" 2>&1; then
    :
else
    echo "❌ Flask not installed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check directories
echo "🔍 Checking directories..."
test -d models && echo "✅ models/ exists" || { echo "❌ models/ missing"; ERRORS=$((ERRORS + 1)); }
test -d data/alerts/images && echo "✅ data/alerts/images/ exists" || { echo "❌ data/alerts/images/ missing"; ERRORS=$((ERRORS + 1)); }
test -d frontend/dashboard/videos && echo "✅ frontend/dashboard/videos/ exists" || { echo "❌ frontend/dashboard/videos/ missing"; ERRORS=$((ERRORS + 1)); }
test -d frontend/web && echo "✅ frontend/web/ exists" || { echo "❌ frontend/web/ missing"; ERRORS=$((ERRORS + 1)); }
echo ""

# Check model
echo "🔍 Checking YOLO model..."
if test -f models/yolov8n.pt; then
    MODEL_SIZE=$(du -h models/yolov8n.pt | cut -f1)
    echo "✅ models/yolov8n.pt exists (${MODEL_SIZE})"
else
    echo "❌ models/yolov8n.pt missing"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check startup scripts
echo "🔍 Checking startup scripts..."
test -f start_backend.sh && echo "✅ start_backend.sh exists" || { echo "❌ start_backend.sh missing"; ERRORS=$((ERRORS + 1)); }
test -f start_frontend.sh && echo "✅ start_frontend.sh exists" || { echo "❌ start_frontend.sh missing"; ERRORS=$((ERRORS + 1)); }
test -f start_all.sh && echo "✅ start_all.sh exists" || { echo "❌ start_all.sh missing"; ERRORS=$((ERRORS + 1)); }
test -x start_backend.sh && echo "✅ start_backend.sh is executable" || { echo "⚠️  start_backend.sh not executable"; }
test -x start_frontend.sh && echo "✅ start_frontend.sh is executable" || { echo "⚠️  start_frontend.sh not executable"; }
test -x start_all.sh && echo "✅ start_all.sh is executable" || { echo "⚠️  start_all.sh not executable"; }
echo ""

# Check frontend dependencies
echo "🔍 Checking frontend dependencies..."
if test -d frontend/web/node_modules; then
    PACKAGE_COUNT=$(ls frontend/web/node_modules | wc -l)
    echo "✅ node_modules installed (${PACKAGE_COUNT} packages)"
else
    echo "❌ node_modules missing"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "======================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ Validation Complete! All checks passed."
    echo "======================================"
    echo ""
    echo "🚀 Ready to start! Run one of these:"
    echo "   ./start_all.sh      # Start both backend and frontend"
    echo "   ./start_backend.sh  # Start backend only"
    echo "   ./start_frontend.sh # Start frontend only"
    echo ""
else
    echo "❌ Validation Failed! Found ${ERRORS} error(s)."
    echo "======================================"
    echo ""
    echo "Try running the setup again:"
    echo "   bash .devcontainer/postCreateCommand.sh"
    echo ""
fi
