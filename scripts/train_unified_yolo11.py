"""
Train Unified YOLOv11 Model for CoastVision
Combines all detection classes into a single model with class weighting
for imbalanced data handling.

Current dataset distribution:
- Drowning:           10,867 (52%)
- Swimming:            7,269 (35%)
- Person out of water:   616 (3%)  ← MINORITY (apply higher weight)

Strategy:
1. Apply 3x weight to minority class (Person_Out)
2. Apply 2x weight to critical class (Drowning)
3. Use augmentation for synthetic diversity
4. Monitor validation metrics per class
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch


def train_unified_model(
    data_yaml="dataset/data.yaml",
    model_size="n",  # nano, small, medium, large, x
    epochs=100,
    imgsz=640,
    batch_size=16,
    device=0,
    patience=20,
    save_best=True,
):
    """
    Train unified YOLOv11 model with class weighting
    
    Args:
        data_yaml: Path to data.yaml with all 3 classes
        model_size: Model size (n=nano, s=small, etc.)
        epochs: Training epochs
        imgsz: Image size
        batch_size: Batch size (adjust for GPU memory)
        device: Device ID (0 for first GPU, 'cpu' for CPU)
        patience: Early stopping patience
        save_best: Save best model only
    """
    
    print("=" * 80)
    print("🚀 CoastVision: Unified YOLOv11 Training")
    print("=" * 80)
    
    # Check device
    if isinstance(device, int) and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  Training on CPU (slow)")
    
    # Load model
    model_name = f"yolo11{model_size}.pt"
    print(f"\n📦 Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    print("\n📊 Training Configuration:")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}×{imgsz}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Early stopping patience: {patience}")
    print("\n⚖️  Class Weighting (to handle imbalance):")
    print("  - Drowning (10,867 samples):        2.0x weight")
    print("  - Swimming (7,269 samples):         1.0x weight (baseline)")
    print("  - Person_Out (616 samples):         4.0x weight ⭐ MINORITY CLASS")
    print("\n🔧 Augmentation Strategy:")
    print("  - Mosaic: 1.0 (4-image tiles)")
    print("  - Mixup: 0.1 (blend images)")
    print("  - HSV: Random hue/saturation/value shifts")
    print("  - Rotation: ±10° for perspective diversity")
    print("  - Flip: Horizontal & vertical augmentation")
    print("  - Scale: 0.5-2.0x size variations")
    
    # Train with class weights and aggressive augmentation
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=patience,
        save=save_best,
        save_best=save_best,
        
        # Class weighting (higher weight = model pays more attention)
        class_weights=[2.0, 1.0, 4.0],  # [Drowning, Swimming, Person_Out]
        
        # Augmentation
        augment=True,
        hsv_h=0.015,      # HSV-Hue
        hsv_s=0.7,        # HSV-Saturation (more variation)
        hsv_v=0.4,        # HSV-Value (brightness)
        degrees=10,       # Rotation
        translate=0.1,    # Translation offset
        scale=0.5,        # Scale variation (0.5-1.5x)
        flipud=0.5,       # Flip upside-down
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation (critical for small objects)
        mixup=0.1,        # Mixup blending
        copy_paste=0.0,   # Disable copy-paste (not ideal for aquatic scenes)
        
        # Optimization
        optimizer="SGD",  # SGD often better than Adam for YOLO
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final LR ratio
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights (if you want to emphasize certain losses)
        box=7.5,          # Box loss (spatial accuracy)
        cls=0.5,          # Classification loss
        dfl=1.5,          # DFL loss
        
        # Validation
        val=True,
        split=0.2,
        
        # Callbacks
        verbose=True,
        
        # Output
        project="runs/train",
        name="unified_yolo11",
        exist_ok=False,
    )
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n📊 Results saved to: {results.save_dir}")
    
    # Print best metrics
    if hasattr(results, 'metrics'):
        print("\n🎯 Best Metrics:")
        print(f"  - mAP50: {results.metrics.get('metrics/mAP50', 'N/A'):.3f}")
        print(f"  - mAP50-95: {results.metrics.get('metrics/mAP50-95', 'N/A'):.3f}")
    
    # Export best model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"\n💾 Best model saved to: {best_model_path}")
        print(f"   → Copy this to models/best.pt to use in production")
        
        # Validate on test set
        print("\n🧪 Validating on test set...")
        validation_results = model.val(
            data=data_yaml,
            device=device,
            imgsz=imgsz,
            batch=batch_size,
            verbose=True,
        )
        
        return results, validation_results
    else:
        print("⚠️  Best model not found (check training logs)")
        return results, None


def analyze_training_results(results):
    """Analyze and visualize training results"""
    print("\n" + "=" * 80)
    print("📈 Training Analysis")
    print("=" * 80)
    
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\nFinal Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train unified YOLOv11 for CoastVision")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="n", help="Model size (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    
    results, val_results = train_unified_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
    )
    
    analyze_training_results(results)
    
    print("\n" + "=" * 80)
    print("🎓 Next Steps:")
    print("=" * 80)
    print("1. Review runs/train/unified_yolo11/results.csv for detailed metrics")
    print("2. Check per-class precision/recall (especially Person_Out)")
    print("3. If accuracy < 85%, expand dataset with more Person_Out images")
    print("4. Copy best.pt to models/best.pt for production deployment")
    print("5. Re-run backend with new model: python backend/server.py")
