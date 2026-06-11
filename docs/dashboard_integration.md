# Integrating Trained Model with Dashboard

## Overview
After training your custom drowning detection model, integrate it with your Flask dashboard to detect drowning, swimming, and persons out of water across 6 beach zones.

---

## Step 1: Complete Model Training

Train your model using:
- Local training: `python scripts/train_yolov8.py`
- Or Colab training: Follow `/docs/colab_training_full_example.md`

After training, you'll have `best.pt` in `runs/detect/trainX/weights/`

---

## Step 2: Copy Trained Model

```bash
# Copy best.pt to models folder
cp runs/detect/train2/weights/best.pt models/best.pt
```

---

## Step 3: Update Dashboard Model Path

Open `frontend/TE PROJ/main.py` and find the YOLO model initialization:

**Change from:**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # General pre-trained model
```

**Change to:**
```python
from ultralytics import YOLO
import os

# Path to your trained model (relative from main.py location)
MODEL_PATH = os.path.join('..', '..', 'models', 'best.pt')
model = YOLO(MODEL_PATH)
```

---

## Step 4: Update Class Names (if needed)

Your model detects 3 classes:
- 0: Drowning
- 1: Person out of water
- 2: Swimming

Make sure your dashboard displays these class names correctly. Update any detection display code:

```python
class_names = {
    0: 'Drowning',
    1: 'Person out of water',
    2: 'Swimming'
}
```

---

## Step 5: Test Integration

1. Start the dashboard:
   ```bash
   cd frontend/"TE PROJ"
   python main.py
   ```

2. Open browser: `http://localhost:5000`

3. Test with sample videos from each zone

4. Verify detections show correct classes (Drowning, Person out of water, Swimming)

---

## Step 6: Add Alert System (Optional)

For drowning alerts, add logic in your detection callback:

```python
def process_detection(results):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Drowning detected
                # Trigger alert
                send_alert_to_lifeguard(zone_id, timestamp)
```

---

## Troubleshooting

**Model not found error:**
- Check `MODEL_PATH` is correct relative to `main.py`
- Verify `best.pt` exists in `models/` folder

**Wrong classes detected:**
- Ensure you're using your trained `best.pt`, not the pre-trained model
- Check `data.yaml` class names match your expectations

**Performance issues:**
- Use `yolov8n.pt` (nano) for faster inference
- Reduce video resolution if needed
- Process fewer frames per second

---

## Next Steps

- Add real-time alerting system
- Implement lifeguard notification interface
- Add logging and analytics
- Create heatmaps for high-risk zones
