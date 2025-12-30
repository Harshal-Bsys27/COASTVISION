# CoastVision YOLOv8 Training on Colab with Auto-Backup

This workflow automatically saves checkpoints to Google Drive during training to prevent data loss.

---

## Setup and Training with Auto-Backup

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Unzip dataset
!unzip -q "/content/drive/MyDrive/dataset.zip" -d /content/

# 3. Install Ultralytics
!pip install ultralytics

# 4. Create a callback to save to Drive every 10 epochs
from ultralytics import YOLO
from ultralytics.utils.callbacks import Callbacks
import shutil
import os

def save_to_drive_callback(trainer):
    """Callback to save checkpoints to Drive every save_period epochs"""
    if trainer.epoch > 0 and trainer.epoch % 10 == 0:  # Save every 10 epochs
        src = '/content/runs'
        dst = '/content/drive/MyDrive/runs_backup'
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"✅ Checkpoint saved to Drive at epoch {trainer.epoch}")

# 5. Train with callback
DATA_YAML = '/content/dataset/data.yaml'

model = YOLO('yolov8n.pt')

# Add custom callback
model.add_callback("on_train_epoch_end", save_to_drive_callback)

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    save_period=10  # YOLOv8 saves checkpoint every 10 epochs
)

# 6. Final save after training completes
shutil.copytree('/content/runs', '/content/drive/MyDrive/runs', dirs_exist_ok=True)
print("✅ Training complete! All results saved to Drive.")
```

---

## Resume Training from Drive Backup

If training is interrupted, resume from your Drive backup:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

# Copy backup to /content/runs
!cp -r /content/drive/MyDrive/runs_backup /content/runs

# Check for last checkpoint
!ls /content/runs/detect/train*/weights/

# Resume from last.pt
from ultralytics import YOLO

LAST_PT = '/content/runs/detect/train2/weights/last.pt'  # Update path as needed
model = YOLO(LAST_PT)

# Add callback for continued auto-backup
def save_to_drive_callback(trainer):
    if trainer.epoch > 0 and trainer.epoch % 10 == 0:
        import shutil
        shutil.copytree('/content/runs', '/content/drive/MyDrive/runs_backup', dirs_exist_ok=True)
        print(f"✅ Checkpoint saved to Drive at epoch {trainer.epoch}")

model.add_callback("on_train_epoch_end", save_to_drive_callback)

results = model.train(resume=True, device=0)

# Final save
import shutil
shutil.copytree('/content/runs', '/content/drive/MyDrive/runs', dirs_exist_ok=True)
print("✅ Training resumed and completed. Results saved to Drive.")
```

---

## Benefits

- Checkpoints are automatically backed up to Drive **every 10 epochs**.
- If GPU quota is exceeded or session disconnects, you can resume from the last backup.
- No manual intervention needed during training.
