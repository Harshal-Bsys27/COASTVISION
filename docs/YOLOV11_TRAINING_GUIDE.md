# YOLOv11 Training Readiness Checklist

## ✓ Prerequisites Met
- [x] Dataset: 20,280 training images + labels
- [x] GPU: RTX 3050 (6GB VRAM)
- [x] PyTorch: 2.9.1 with CUDA support
- [x] Ultralytics: 8.3.241 (supports YOLOv11)
- [x] Python: 3.8+
- [x] Storage: ~50GB available (for training + models)

## System Specifications
```
GPU: NVIDIA RTX 3050 (6GB)
CPU: Intel Core i7/i9
RAM: 16GB+
PyTorch Version: 2.9.1
CUDA Support: Yes
```

## Expected Training Performance
- **Model**: YOLOv11n (5.5MB, nano variant)
- **Training Time**: 2-3 hours for 100 epochs
- **Batch Size**: 16 (optimal for 6GB VRAM)
- **Image Size**: 640x640 (balanced speed/accuracy)
- **Learning Rate**: 0.01 (initial)

## Pre-Training Steps

### 1. Verify Dataset
```bash
# Check dataset structure
ls -la dataset/
# Should show: train/, valid/, test/, data.yaml
```

### 2. Check GPU Memory
```bash
python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### 3. Backup Current Model
```bash
# Save current model as fallback
cp models/best.pt models/best.pt.backup
```

## Training Commands

### Standard Training (Recommended)
```bash
python scripts/train_yolov11.py \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --imgsz 640
```

**Duration**: ~2-3 hours
**Expected mAP50**: 0.85-0.90

### Quick Validation (for testing)
```bash
python scripts/train_yolov11.py \
  --epochs 10 \
  --batch 16 \
  --device 0 \
  --imgsz 640
```

**Duration**: ~15-20 minutes
**Use case**: Verify setup before full training

### High-Precision Training (for best accuracy)
```bash
python scripts/train_yolov11.py \
  --epochs 150 \
  --batch 16 \
  --device 0 \
  --imgsz 768 \
  --lr0 0.005
```

**Duration**: ~4-5 hours
**Expected mAP50**: 0.88-0.92

### Resume Training (if interrupted)
```bash
python scripts/train_yolov11.py \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --resume
```

## Post-Training Deployment

### 1. Verify Model Quality
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/yolov11_drowning/weights/best.pt')
print(f'Model loaded: {model.model}')
"
```

### 2. Run Validation
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/yolov11_drowning/weights/best.pt')
results = model.val()
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
"
```

### 3. Deploy to Backend
```bash
# Copy best model to active directory
cp runs/detect/yolov11_drowning/weights/best.pt models/best.pt

# Verify backend loads it
python backend/server.py
# Check logs for: "[INIT] YOLO model loaded: models/best.pt"
```

### 4. Test Inference
```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/best.pt')

# Test on sample image
results = model.predict('dataset/test/images/sample.jpg', conf=0.5)
print(f'Detections: {len(results[0].boxes)}')
"
```

## Monitoring During Training

### Real-time Monitoring
The training script will log:
- Epoch progress
- Loss metrics
- Validation mAP
- Learning rate adjustments

### Check Training Progress
```bash
# In another terminal:
ls -la runs/detect/yolov11_drowning/weights/
# Watch best.pt and last.pt update in real-time
```

## Troubleshooting

### Out of Memory Error
**Solution**: Reduce batch size
```bash
python scripts/train_yolov11.py --epochs 100 --batch 8 --device 0
```

### Slow Training
**Solution**: Disable augmentation or reduce image size
```bash
python scripts/train_yolov11.py \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --imgsz 512 \
  --no-augment
```

### GPU Not Detected
**Solution**: Verify CUDA
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU available: {torch.cuda.is_available()}')
"
```

## Data Augmentation Strategy

During training, YOLOv11 applies:
- **Mosaic**: 100% (default)
- **Mixup**: 10%
- **HSV shifts**: Color variations
- **Rotation**: ±10°
- **Flips**: 50% horizontal/vertical
- **Scale**: 0.5x to 2.0x

These augmentations help generalize across:
- Different water conditions
- Various camera angles
- Different lighting
- Crowded and sparse scenes

## Expected Results

### Baseline (YOLOv8n before training)
- mAP50: ~0.70-0.75
- Speed: 25-30ms/image

### After YOLOv11n Training
- mAP50: ~0.85-0.90 ✓ Improvement: +15-20%
- Speed: 30-40ms/image (slightly slower, better accuracy trade-off)

### After Dataset Expansion + Training
- mAP50: ~0.90-0.95 (achievable with Person_out expansion)
- Recall: +25-30% (fewer missed drownings)
- Precision: +10-15% (fewer false positives)

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Verify Setup | 5 min | Ready ✓ |
| 2 | Train YOLOv11 | 2-3 hrs | **READY TO START** |
| 3 | Validate Results | 10 min | Post-training |
| 4 | Deploy to Backend | 2 min | Post-training |
| 5 | Dataset Expansion (optional) | 1-2 days | Optional |
| 6 | Retrain with Expanded Data | 2-3 hrs | Optional |

## Next Steps

1. **Now**: Start training
   ```bash
   python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0
   ```

2. **After 2-3 hours**: Validate results
   ```bash
   # Check runs/detect/yolov11_drowning/ for metrics
   ```

3. **Optional**: Expand Person_out dataset and retrain for +10-15% mAP

## Resources

- YOLOv11 Docs: https://docs.ultralytics.com/
- Training Guide: https://docs.ultralytics.com/modes/train/
- Performance Tips: https://docs.ultralytics.com/guides/training-tips-best-practices/

## Important Notes

⚠️ **GPU Memory**: RTX 3050 (6GB) is at the edge with batch 16. If OOM errors occur:
- Reduce batch to 8
- Reduce imgsz to 512
- Enable gradient accumulation

⚠️ **Power**: Training will use 100% GPU for 2-3 hours. Ensure adequate cooling.

⚠️ **Interruption**: Training checkpoints every 10 epochs. Can resume with `--resume`.
