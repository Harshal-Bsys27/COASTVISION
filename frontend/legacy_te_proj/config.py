# config.py
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent

# Put your zone videos in a "videos" subfolder (change names if you used different names)
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_PATHS = [
    str(VIDEO_DIR / f"zone{i}.mp4") for i in range(1, 7)
]

# Friendly names for zones
ZONE_NAMES = [f"Zone {i}" for i in range(1, 7)]

# UI constants
DISPLAY_WIDTH = 480  # preview width per zone

ALERT_CSV = str(BASE_DIR / "alerts.csv")

def get_device_string():
    if torch.cuda.is_available():
        try:
            return f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU: (cuda available)"
    return "CPU Mode"
