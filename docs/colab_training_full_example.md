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

## 6. Train YOLOv8 Model

```python
from ultralytics import YOLO

DATA_YAML = '/content/dataset/data.yaml'

model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=0  # Use GPU
)
```

---

## 7. Save Results Back to Google Drive

```python
!cp -r /content/runs /content/drive/MyDrive/
```

---

## 8. Download Your Model

- After training, download `best.pt` and results from `/content/drive/MyDrive/runs/detect/train/weights/`.

---

**Tip:**  
- If you get a path error, check the output of `print(os.listdir('/content'))` and `print(os.listdir('/content/dataset'))` to confirm where your files are.
