#!/usr/bin/env python3
"""
Safe Model Deployment Tool
==========================
Validates new YOLOv11 model before deploying to production.
Keeps current system running until validation passes.

Usage:
    # 1. After training completes, staging model is at: models/staging/yolov11_best.pt
    # 2. Validate the model:
    python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt

    # 3. If validation passes, deploy:
    python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --deploy
"""

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[Deploy] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate and deploy new YOLOv11 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Step 1: Validate new model (no changes to system)
  python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt

  # Step 2: If validation passes, deploy to production
  python scripts/validate_and_deploy.py --model models/staging/yolov11_best.pt --deploy

  # Step 3: Restart backend to load new model
  # Backend will auto-load from models/best.pt
        '''
    )
    
    parser.add_argument('--model', required=True,
                        help='Path to model to validate (e.g., models/staging/yolov11_best.pt)')
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy model to production after validation (replaces models/best.pt)')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='Backup current model before deployment (default: True)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare new model against current model')
    
    return parser.parse_args()


def validate_model(model_path):
    """Validate model on test dataset."""
    logger.info(f"\n{'='*70}")
    logger.info("STEP 1: Validating new model on test dataset...")
    logger.info(f"{'='*70}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return None
    
    try:
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = YOLO(str(model_path))
        
        # Validate
        logger.info("Running validation...")
        val_results = model.val(data='dataset/data.yaml')
        
        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION RESULTS:")
        logger.info(f"{'='*70}")
        logger.info(f"  mAP50:     {val_results.box.map50:.4f}")
        logger.info(f"  mAP50-95:  {val_results.box.map:.4f}")
        logger.info(f"  Precision: {val_results.box.mp:.4f}")
        logger.info(f"  Recall:    {val_results.box.mr:.4f}")
        
        return val_results
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(new_model_path):
    """Compare new model with current production model."""
    logger.info(f"\n{'='*70}")
    logger.info("STEP 2: Comparing new vs current model...")
    logger.info(f"{'='*70}")
    
    current_model_path = Path('models/best.pt')
    new_model_path = Path(new_model_path)
    
    if not current_model_path.exists():
        logger.warning(f"⚠️  Current model not found: {current_model_path}")
        return None
    
    try:
        logger.info("Validating current production model...")
        current_model = YOLO(str(current_model_path))
        current_results = current_model.val(data='dataset/data.yaml')
        
        logger.info("Validating new model...")
        new_model = YOLO(str(new_model_path))
        new_results = new_model.val(data='dataset/data.yaml')
        
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON:")
        logger.info(f"{'='*70}")
        
        # mAP50 comparison
        current_map50 = current_results.box.map50
        new_map50 = new_results.box.map50
        improvement = ((new_map50 - current_map50) / current_map50 * 100)
        
        logger.info(f"\n  mAP50:")
        logger.info(f"    Current: {current_map50:.4f}")
        logger.info(f"    New:     {new_map50:.4f}")
        logger.info(f"    Change:  {improvement:+.1f}% {'✓ BETTER' if improvement > 0 else '✗ WORSE'}")
        
        # mAP50-95 comparison
        current_map = current_results.box.map
        new_map = new_results.box.map
        improvement_95 = ((new_map - current_map) / current_map * 100)
        
        logger.info(f"\n  mAP50-95:")
        logger.info(f"    Current: {current_map:.4f}")
        logger.info(f"    New:     {new_map:.4f}")
        logger.info(f"    Change:  {improvement_95:+.1f}% {'✓ BETTER' if improvement_95 > 0 else '✗ WORSE'}")
        
        # Recommendation
        logger.info(f"\n{'='*70}")
        if improvement >= 5.0:  # At least 5% improvement
            logger.info("✓ RECOMMENDATION: Safe to deploy (meets improvement threshold)")
            return True
        elif improvement >= 0:
            logger.info("⚠️  CAUTION: Minor improvement only (+0-5%)")
            logger.info("   Consider collecting more data before deploying")
            return False
        else:
            logger.error("❌ REJECT: New model performs WORSE")
            logger.error("   Do NOT deploy")
            return False
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        return None


def deploy_model(new_model_path, backup=True):
    """Deploy new model to production."""
    logger.info(f"\n{'='*70}")
    logger.info("STEP 3: Deploying new model to production...")
    logger.info(f"{'='*70}")
    
    new_model_path = Path(new_model_path)
    current_model_path = Path('models/best.pt')
    
    if not new_model_path.exists():
        logger.error(f"❌ New model not found: {new_model_path}")
        return False
    
    try:
        # Backup current model
        if backup and current_model_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = Path('models/best.pt.backup') / f"best_{timestamp}.pt"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(current_model_path, backup_path)
            logger.info(f"✓ Backed up current model: {backup_path}")
        
        # Deploy
        shutil.copy(new_model_path, current_model_path)
        logger.info(f"✓ Deployed new model to: {current_model_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ DEPLOYMENT COMPLETE!")
        logger.info(f"{'='*70}")
        logger.info(f"\n⚠️  NEXT STEPS:")
        logger.info(f"  1. Restart backend server:")
        logger.info(f"     python backend/server.py")
        logger.info(f"\n  2. Backend will auto-load new model from: models/best.pt")
        logger.info(f"\n  3. Dashboard will show real-time improvements in detection")
        logger.info(f"\n  If issues occur, restore backup:")
        logger.info(f"     cp {backup_path} {current_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        return False


def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("YOLOv11 Model Deployment Pipeline")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Deploy: {'Yes' if args.deploy else 'No (validation only)'}")
    logger.info(f"Backup: {args.backup}")
    
    # Step 1: Validate
    val_results = validate_model(args.model)
    if not val_results:
        logger.error("\n❌ Validation failed - aborting")
        return False
    
    # Step 2: Compare (optional)
    if args.compare:
        comparison_ok = compare_models(args.model)
        if not comparison_ok:
            logger.error("\n❌ Comparison failed - aborting deployment")
            if args.deploy:
                return False
    
    # Step 3: Deploy (if requested)
    if args.deploy:
        logger.info("\n⚠️  DEPLOYING TO PRODUCTION...")
        success = deploy_model(args.model, backup=args.backup)
        return success
    else:
        logger.info(f"\n{'='*70}")
        logger.info("✓ VALIDATION COMPLETE - Model ready for deployment")
        logger.info(f"{'='*70}")
        logger.info(f"\nWhen ready to deploy, run:")
        logger.info(f"  python scripts/validate_and_deploy.py --model {args.model} --deploy")
        return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
