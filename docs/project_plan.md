# Project Plan: CoastVision

## 1. Data Collection & Annotation
- Identify and gather relevant datasets (public sources, Roboflow, custom collection).
- Organize raw videos/images in `/data/raw_videos/`.
- Extract frames from videos using scripts (see `/scripts/extract_frames.py`).
- Annotate images for drowning/person detection using tools like LabelImg or Roboflow.
- Store annotations in YOLO format in `/dataset/`.

## 2. Data Preprocessing
- Clean and verify annotations.
- Split data into training, validation, and test sets.
- Augment data if needed (flipping, rotation, etc.).
- Document preprocessing steps in `/docs`.

## 3. Model Development
- Train YOLOv8 model for drowning/person detection using annotated dataset.
- Experiment with different model sizes (n, s, m, l) based on hardware.
- Tune hyperparameters (batch size, epochs, learning rate).
- Save training scripts in `/scripts/train_yolov8.py`.

## 4. Model Evaluation
- Evaluate model performance on validation and test sets.
- Analyze metrics: mAP, precision, recall, confusion matrix.
- Visualize results and sample detections.
- Document findings and sample outputs in `/docs`.

## 5. Inference & Demo
- Run inference on new images/videos using `/scripts/inference_yolov8.py`.
- Test on zone-wise beach images and real-world scenarios.
- Save and document sample outputs.

## 6. Deployment & Integration
- Integrate trained model into backend (Flask/FastAPI) for real-time video stream processing.
- Develop alerting mechanism for lifeguards (e.g., Android app, notifications).
- Plan for dashboard/analytics UI (Streamlit/PyQt).

## 7. Future Enhancements
- Add pose estimation (MediaPipe/OpenPose) for advanced swimmer tracking.
- Expand to rip current detection and crowd management modules.
- Incorporate citizen-science data uploads for continuous improvement.
- Add heatmaps and analytics for risk assessment.

## Useful Links

- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51 Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Roboflow](https://roboflow.com/)
- [LabelImg](https://github.com/tzutalin/labelImg)

---

**Next Steps:**
1. Complete data collection and annotation.
2. Train and validate the drowning detection model.
3. Integrate and test in real-world scenarios before expanding features.
