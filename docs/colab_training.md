# Training CoastVision YOLOv8 Model on Google Colab

## Steps

1. **Upload Dataset**
   - Zip your `dataset` folder and upload it directly to the root of your Google Drive (not inside any folder).

2. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

3. **Colab Notebook Template**

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Unzip your dataset (since it's in the root of Drive)
!unzip -q "/content/drive/MyDrive/dataset.zip" -d /content/

# 3. Install Ultralytics (YOLOv8)
!pip install ultralytics

# 4. (Optional) Check GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 5. Train YOLOv8
from ultralytics import YOLO

DATA_YAML = '/content/dataset/data.yaml'  # Update if needed

model = YOLO('yolov8n.pt')
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device=0  # Use GPU
)

# 6. Save results back to Google Drive
!cp -r runs /content/drive/MyDrive/
```

4. **After Training**
   - Download your `best.pt` model and results from Google Drive.

---

**Tip:**  
- Always save your results to Google Drive to avoid data loss if the Colab session disconnects.
