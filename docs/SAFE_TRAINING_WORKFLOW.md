# Safe YOLOv11 Model Training Workflow

## Architecture: Non-Disruptive Training

```
┌─────────────────────────────────────────────────────────┐
│  PRODUCTION SYSTEM (RUNNING)                           │
│  ├─ Backend Server: backend/server.py                  │
│  ├─ Current Model: models/best.pt (YOLOv8)            │
│  └─ Dashboard: Live alerts, real-time detections       │
└─────────────────────────────────────────────────────────┘
                        ↓ (Untouched)
┌─────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (PARALLEL)                          │
│  ├─ Training Script: scripts/train_yolov11.py          │
│  ├─ Staging Location: models/staging/yolov11_best.pt  │
│  ├─ Duration: 2-3 hours (separate process)            │
│  └─ Status: Does NOT affect running system            │
└─────────────────────────────────────────────────────────┘
                        ↓ (After training)
┌─────────────────────────────────────────────────────────┐
│  VALIDATION PHASE (Safe)                               │
│  ├─ Validate Script: scripts/validate_and_deploy.py   │
│  ├─ Compare: New vs Current model performance          │
│  ├─ Decision: Proceed only if +5% improvement         │
│  └─ Risk: ZERO (still on current model)                │
└─────────────────────────────────────────────────────────┘
                        ↓ (If approved)
┌─────────────────────────────────────────────────────────┐
│  DEPLOYMENT PHASE (Controlled)                         │
│  ├─ Backup: Auto-backup current model                 │
│  ├─ Replace: Copy new model to models/best.pt         │
│  ├─ Restart: Manual restart of backend server         │
│  └─ Rollback: Simple restore if issues occur          │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Process

### STEP 1: Start Training (System Still Running)

**Command**:
```bash
python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0
```

**What happens**:
- ✓ Training runs in separate process
- ✓ Your current system continues working
- ✓ Alerts keep flowing in real-time
- ✓ Dashboard remains unaffected
- ✓ New model saved to: `models/staging/yolov11_best.pt`

**During training** (2-3 hours):
- Logs show training progress
- No interruption to live system
- You can stop/pause if needed (just Ctrl+C)
- Checkpoints saved every 10 epochs (can resume)

---

### STEP 2: Validate New Model (No Risk)

**After training completes**, validate:

```bash
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt
```

**What it checks**:
- ✓ Model loads correctly
- ✓ Runs on test dataset
- ✓ Calculates mAP50, mAP50-95, Precision, Recall
- ✓ Shows validation metrics

**Output**:
```
VALIDATION RESULTS:
  mAP50:     0.8765
  mAP50-95:  0.6234
  Precision: 0.9012
  Recall:    0.8567
```

**Risk Level**: ZERO (no changes to system)

---

### STEP 3: Compare Models (Decision Point)

**Option A - Quick Validation**:
```bash
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --compare
```

**Compares**:
```
mAP50:
  Current: 0.7234
  New:     0.8765
  Change:  +21.1% ✓ BETTER
```

**Threshold for deployment**:
- **≥ +5% improvement**: Safe to deploy ✓
- **0 to +5%**: Minor improvement (optional)
- **< 0% (worse)**: Do NOT deploy ✗

**Risk Level**: ZERO (comparison only)

---

### STEP 4: Deploy to Production (Controlled)

**When ready**:
```bash
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --deploy
```

**What happens**:
1. ✓ Auto-backups current model: `models/best.pt.backup/best_TIMESTAMP.pt`
2. ✓ Copies new model to: `models/best.pt`
3. ✓ Deployment complete

**Risk Level**: LOW (backup in place)

**Current system**: Still running with old model

---

### STEP 5: Restart Backend (Activation)

**After deployment**, restart backend:

```bash
# Kill current backend (Ctrl+C if running)
# Or in new terminal:
python backend/server.py
```

**What happens**:
- ✓ Backend loads new model from `models/best.pt`
- ✓ All WebSocket connections re-established
- ✓ Dashboard shows improved detection
- ✓ Real-time alerts with better accuracy

**Monitoring**:
- Check logs for: `[INIT] YOLO model loaded: models/best.pt`
- Should see improvements in detections immediately

---

### STEP 6: Rollback (If Needed)

**If anything goes wrong**:

```bash
# Restore from backup
cp models/best.pt.backup/best_TIMESTAMP.pt models/best.pt

# Restart backend
python backend/server.py
```

**Duration**: < 2 minutes
**Risk**: MINIMAL (previous working version restored)

---

## Parallel Workflow Example

### Terminal 1 - Current System (Keep Running)
```bash
# Start production backend
python backend/server.py
# Runs continuously, serving alerts
```

### Terminal 2 - Training (New Process)
```bash
# Start YOLOv11 training in separate window
python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0
# Runs for 2-3 hours
# Does NOT affect Terminal 1
```

### Terminal 3 - Validation (After Training)
```bash
# Once training finishes:
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --compare

# If results look good:
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --deploy
```

### Terminal 1 - Restart Backend
```bash
# Stop current backend (Ctrl+C)
# Restart to load new model
python backend/server.py
```

---

## File Organization

```
models/
├── best.pt                          (Current production model - YOLOv8)
├── best.pt.backup/                  (Backups folder)
│   └── best_20260531_143022.pt     (Auto-backup on deployment)
└── staging/
    └── yolov11_best.pt             (New model - training output)
    
runs/detect/
└── yolov11_drowning/               (Training logs & metrics)
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    └── plots/                       (Training visualizations)
```

---

## Dataset Expansion (Optional, Parallel)

While training runs, you can optionally expand dataset:

```bash
# Terminal 4 - Dataset Expansion
python scripts/augment_person_out.py --count 1000 --output dataset/augmented

# Then manually move to dataset/train/labels and dataset/train/images
# And retrain with expanded dataset (will improve accuracy further)
```

---

## Timeline

| Phase | Duration | Action | System Impact |
|-------|----------|--------|----------------|
| 1. Training | 2-3 hrs | `train_yolov11.py` | ✓ None (parallel) |
| 2. Validation | 10 min | `validate_and_deploy.py` | ✓ None (read-only) |
| 3. Deployment | 1 min | Copy best.pt | ✓ None (before restart) |
| 4. Restart | 30 sec | Kill & restart server | ⚠️ Brief downtime |
| **Total** | **2-3.5 hrs** | | **30 sec downtime** |

---

## Safety Checklist

✓ Current model backed up before deployment
✓ New model tested before deployment
✓ Validation script compares vs current
✓ Deployment can be rolled back in seconds
✓ Backups stored with timestamps
✓ Training doesn't interfere with production
✓ Separate staging directory for new model
✓ Quick comparison metrics before committing

---

## Commands Reference

```bash
# 1. Start training (system keeps running)
python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0

# 2. Validate new model (no system changes)
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt

# 3. Compare new vs current (view improvement)
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --compare

# 4. Deploy (backup + replace)
python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --deploy

# 5. Restart backend (activate new model)
python backend/server.py

# 6. Rollback if needed
cp models/best.pt.backup/best_TIMESTAMP.pt models/best.pt
python backend/server.py
```

---

## Key Points

🔒 **Safety First**: Current system never interrupted until you choose to restart
📊 **Validation Required**: New model must show improvement before deployment
🔄 **Rollback Ready**: One command to restore previous model
⏱️ **Minimal Downtime**: ~30 seconds when restarting backend
📈 **Measurable Improvement**: Compare metrics before/after
🎯 **No Risk Deployment**: Staging area completely separate
