# CoastVision: AI-Powered Beach Surveillance System 

## Project Overview
CoastVision is an AI-based surveillance system designed to detect humans in water, identify potential drowning behavior, and alert lifeguards in real-time using computer vision and machine learning. later the surveillance system shall be connected to an Android Application which will be for lifeguards where then can recieve alerts and images . 

## Objectives
- Detect humans entering water zones using AI models and camera feeds.
- Analyze human behavior to detect distress (e.g., erratic motion, struggle, waving).
- Alert lifeguards in real-time with camera ID and approximate distance.
- Provide heatmaps for analyzing peak activity times or risky zones.

## Tech Stack
- **Python** (primary language)
- **OpenCV, YOLOv5/v8, MediaPipe/OpenPose** (Computer Vision)
- **TensorFlow / Keras / PyTorch** (Behavior Analysis, LSTM models)
- **Flask / FastAPI / Socket.IO** (Streaming & Alerts)
- **Roboflow / LabelImg** (Data annotation)
- **Streamlit / PyQt** (Dashboard UI)
- **Google Colab / Kaggle Kernels** (for training with GPU)

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/coastvision.git
    ```

2. Install dependencies (to be updated as the project grows).

3. Follow the documentation in `/docs` for setup and usage.

## Folder Structure

```
/backend      # Backend API and streaming code
/frontend     # Lifeguard app and dashboard UI
/models       # Model training and inference scripts
/data         # Datasets and annotations (not tracked in git)
/docs         # Documentation and research notes
```

## License
MIT License (add LICENSE file as needed)
