# CoastVision YOLOv8 Training on Google Colab

## 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2. Unzip Your Dataset

```python
# Update the path if your zip is in a subfolder or has a different name
!unzip -q "/content/drive/MyDrive/dataset.zip" -d /content/
```

---

## 3. Check Extracted Files

```python
import os
print(os.listdir('/content/dataset'))
# Should show: ['train', 'valid', 'test', 'data.yaml']
```

---

## 4. Install Ultralytics (YOLOv8)

```python
!pip install ultralytics
```

---

## 5. (Optional) Check GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

---

## 6. Train YOLOv8 Model with Automatic Checkpoint Saving

```python
from ultralytics import YOLO
import shutil

DATA_YAML = '/content/dataset/data.yaml'

model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,  # Use GPU
    save_period=10  # Save checkpoint every 10 epochs
)

# After training completes, copy all results to Google Drive
shutil.copytree('/content/runs', '/content/drive/MyDrive/runs', dirs_exist_ok=True)
print("Training complete and all results saved to Drive!")
```

---

## 7. (Optional) Manually Save Progress During Training

If you want to save progress while training is still running, open a new cell and run:

```python
!cp -r /content/runs /content/drive/MyDrive/
print("Progress saved to Drive!")
```

Run this periodically (every 15-20 minutes) to back up checkpoints.

---

## 8. Resume Training from Last Checkpoint

If your session disconnects and you want to resume training:

```python
from google.colab import drive
drive.mount('/content/drive')

# Install Ultralytics
!pip install ultralytics

# Check for saved checkpoint
!ls /content/drive/MyDrive/runs/detect/

# Resume from last checkpoint (update path as needed)
from ultralytics import YOLO

LAST_PT = '/content/drive/MyDrive/runs/detect/train2/weights/last.pt'  # Update folder name if different

model = YOLO(LAST_PT)
results = model.train(
    resume=True,
    device=0  # Use GPU
)

# After training, save results to Drive
import shutil
shutil.copytree('/content/runs', '/content/drive/MyDrive/runs', dirs_exist_ok=True)
print("Training resumed and completed. Results saved to Drive!")
```

---

## 9. Download Your Model

- After training, download `best.pt` and results from `/content/drive/MyDrive/runs/detect/trainX/weights/`.

---

## Tips

- Always save `/content/runs` to Drive after training or periodically during training to prevent data loss.
- Use `save_period=10` to automatically save checkpoints every 10 epochs.
- If Colab disconnects, you can resume training from `last.pt` saved in your Drive.
- Check paths with `!ls /content/drive/MyDrive/runs/detect/` to find your checkpoint folders.
