from ultralytics import YOLO
import sys

if __name__ == "__main__":
    # Path to your trained model
    MODEL_PATH = 'runs/detect/train5/weights/best.pt'  # Update as needed
    # Path to image or video for inference
    SOURCE = sys.argv[1] if len(sys.argv) > 1 else 'sample.jpg'  # Default to 'sample.jpg'

    model = YOLO(MODEL_PATH)
    results = model.predict(SOURCE, save=True, conf=0.25)  # Adjust conf threshold as needed

    print(f"Inference complete. Results saved in 'runs/detect/predict'.")
