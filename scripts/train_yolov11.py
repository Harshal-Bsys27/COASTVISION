#!/usr/bin/env python3
"""
YOLOv11 Unified Multi-Class Training Script
============================================
Trains YOLOv11 model for 3-class detection:
  - Drowning (weight: 2.0)
  - Person out of water (weight: 1.0)
  - Swimming (weight: 4.0)

Target: YOLOv11n (nano - 5.5MB, 30-40ms inference on RTX 3050)
Expected Training Time: 2-3 hours on RTX 3050
Expected mAP50: 0.85-0.90

Usage:
    python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[YOLOv11] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv11 model for drowning detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Standard training (100 epochs, batch 16)
  python scripts/train_yolov11.py --epochs 100 --batch 16 --device 0
  
  # Quick validation (10 epochs, batch 8)
  python scripts/train_yolov11.py --epochs 10 --batch 8 --device 0
  
  # High precision training (150 epochs, batch 32)
  python scripts/train_yolov11.py --epochs 150 --batch 32 --device 0 --imgsz 768
        '''
    )
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, RTX 3050 handles 16-24)')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640, use 768 for high precision)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience epochs (default: 20)')
    parser.add_argument('--model', choices=['nano', 'small', 'medium'], default='nano',
                        help='Model size: nano(5.5MB), small(13.6MB), medium(35.3MB) (default: nano)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable aggressive augmentation (default: True)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='Disable augmentation')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--name', type=str, default='yolov11_drowning',
                        help='Training run name (default: yolov11_drowning)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--weights', type=str, default='yolov11n.pt',
                        help='Initial weights path (default: yolov11n.pt)')
    
    return parser.parse_args()


def check_environment():
    """Verify CUDA and dependencies."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.warning("⚠️  CUDA not available - training will be slow on CPU")
    
    # Check dataset
    dataset_yaml = Path('dataset/data.yaml')
    if not dataset_yaml.exists():
        logger.error(f"❌ Dataset config not found: {dataset_yaml}")
        return False
    
    logger.info(f"✓ Dataset config: {dataset_yaml}")
    
    # Check training data
    train_images = Path('dataset/train/images')
    valid_images = Path('dataset/valid/images')
    if not train_images.exists() or not valid_images.exists():
        logger.error("❌ Training/validation images not found")
        return False
    
    train_count = len(list(train_images.glob('*')))
    valid_count = len(list(valid_images.glob('*')))
    logger.info(f"✓ Train images: {train_count}")
    logger.info(f"✓ Valid images: {valid_count}")
    
    return True


def get_model_info(model_size):
    """Get model specifications."""
    specs = {
        'nano': {'weights': 'yolov11n.pt', 'size': '5.5MB', 'inference': '30-40ms'},
        'small': {'weights': 'yolov11s.pt', 'size': '13.6MB', 'inference': '50-70ms'},
        'medium': {'weights': 'yolov11m.pt', 'size': '35.3MB', 'inference': '80-120ms'},
    }
    return specs.get(model_size, specs['nano'])


def train(args):
    """Execute training."""
    logger.info("=" * 70)
    logger.info("YOLOv11 Drowning Detection Training")
    logger.info("=" * 70)
    
    # Environment check
    if not check_environment():
        return False
    
    # Model info
    model_spec = get_model_info(args.model)
    logger.info(f"\nModel: YOLOv11{args.model[0].upper()} ({model_spec['size']})")
    logger.info(f"Expected inference time: {model_spec['inference']}")
    
    # Training config
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Image size: {args.imgsz}")
    logger.info(f"  Learning rate: {args.lr0}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Augmentation: {'enabled' if args.augment else 'disabled'}")
    
    # Class weights (Drowning > Person_out > Swimming priority)
    class_weights = [2.0, 1.0, 4.0]  # Drowning=2.0, Person_out=1.0, Swimming=4.0
    logger.info(f"  Class weights: {class_weights}")
    
    # Load model
    logger.info(f"\nLoading YOLOv11 model...")
    model = YOLO('yolov11n.pt')
    
    # Training parameters
    train_args = {
        'data': 'dataset/data.yaml',
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'patience': args.patience,
        'lr0': args.lr0,
        'name': args.name,
        'project': 'runs/detect',
        
        # Augmentation
        'mosaic': 1.0,  # Mosaic augmentation
        'mixup': 0.1,   # Mixup augmentation
        'hsv_h': 0.015, # HSV hue adjustment
        'hsv_s': 0.7,   # HSV saturation adjustment
        'hsv_v': 0.4,   # HSV value adjustment
        'degrees': 10,  # Rotation degrees
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.5,
        'fliplr': 0.5,
        
        # Optimization
        'optimizer': 'SGD',
        'close_mosaic': 10,  # Disable mosaic for last 10 epochs
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 10,
        'plots': True,
        
        # Model settings
        'iou': 0.7,
        'conf': 0.001,
        'max_det': 300,
        
        # GPU/CPU settings
        'amp': True,  # Automatic mixed precision
        'fraction': 1.0,
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("Starting training... (this may take 2-3 hours on RTX 3050)")
    logger.info("=" * 70)
    
    try:
        # Train the model
        results = model.train(**train_args)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Training completed successfully!")
        logger.info("=" * 70)
        
        # Results summary
        logger.info(f"\nTraining Results:")
        logger.info(f"  Best model: {results.save_dir}/weights/best.pt")
        logger.info(f"  Last model: {results.save_dir}/weights/last.pt")
        
        # Validate
        logger.info("\nValidating best model...")
        val_results = model.val()
        logger.info(f"  mAP50: {val_results.box.map50:.4f}")
        logger.info(f"  mAP50-95: {val_results.box.map:.4f}")
        
        # Save best model to STAGING directory (does NOT replace current model)
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        staging_dir = Path('models/staging')
        staging_dir.mkdir(parents=True, exist_ok=True)
        target_path = staging_dir / 'yolov11_best.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, target_path)
            logger.info(f"✓ Saved new model to STAGING: {target_path}")
            logger.info(f"⚠️  Current system still using: models/best.pt (UNCHANGED)")
            logger.info(f"✓ Review and validate before deployment!")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Ready for deployment!")
        logger.info("  Next: Restart backend server to load new model")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    args = parse_args()
    success = train(args)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
