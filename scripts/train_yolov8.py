from ultralytics import YOLO

if __name__ == "__main__":
    # Update this path to match your dataset location
    DATA_YAML = 'dataset/data.yaml'

    # Train YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0  # Use GPU 0
    )

    # Save results and model weights in 'runs/' directory
