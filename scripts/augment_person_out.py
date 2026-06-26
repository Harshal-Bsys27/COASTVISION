#!/usr/bin/env python3
"""
Person Out of Water - Data Augmentation Script
==============================================
Creates synthetic augmentations of existing Person_out images
to expand the minority class before YOLOv11 training.

Usage:
    python scripts/augment_person_out.py --count 1000 --output dataset/augmented
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import random
import math

logging.basicConfig(level=logging.INFO, format='[Augment] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Augment Person_out dataset')
    parser.add_argument('--source', default='dataset/train/images',
                        help='Source images directory')
    parser.add_argument('--labels', default='dataset/train/labels',
                        help='Source labels directory')
    parser.add_argument('--output', default='dataset/augmented',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=1000,
                        help='Number of augmented images to generate')
    parser.add_argument('--person-out-id', type=int, default=1,
                        help='Class ID for Person_out (default: 1)')
    return parser.parse_args()


def get_person_out_images(images_dir, labels_dir, person_out_id=1):
    """Get list of images containing Person_out detections."""
    person_out_images = []
    
    labels_path = Path(labels_dir)
    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            has_person_out = False
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == person_out_id:
                    has_person_out = True
                    break
            
            if has_person_out:
                img_name = label_file.stem
                img_path = Path(images_dir) / f"{img_name}.jpg"
                if not img_path.exists():
                    img_path = Path(images_dir) / f"{img_name}.png"
                
                if img_path.exists():
                    person_out_images.append(img_path)
    
    return person_out_images


def apply_augmentations(image, label_file):
    """Apply random augmentations to image and labels."""
    augmentations = [
        ('brightness', random.uniform(0.7, 1.3)),
        ('contrast', random.uniform(0.7, 1.3)),
        ('saturation', random.uniform(0.5, 1.5)),
        ('blur', random.randint(1, 5)),
        ('noise', random.uniform(0, 20)),
        ('rotation', random.uniform(-15, 15)),
        ('flip_h', random.choice([True, False])),
        ('crop', random.uniform(0.8, 1.0)),
    ]
    
    aug_type = random.choice(list(range(len(augmentations))))
    name, value = augmentations[aug_type]
    
    h, w = image.shape[:2]
    
    if name == 'brightness':
        image = cv2.convertScaleAbs(image, alpha=value, beta=0)
    
    elif name == 'contrast':
        image = cv2.convertScaleAbs(image, alpha=value, beta=0)
    
    elif name == 'saturation':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif name == 'blur':
        kernel_size = int(value) * 2 + 1
        image = cv2.blur(image, (kernel_size, kernel_size))
    
    elif name == 'noise':
        noise = np.random.normal(0, value, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    elif name == 'rotation':
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, value, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h))
        
        # Update bounding boxes
        if label_file:
            labels = []
            with open(label_file, 'r') as f:
                labels = f.readlines()
            # Note: Rotation updates labels - simplified here
            # For production, use proper affine transformation of coordinates
    
    elif name == 'flip_h':
        image = cv2.flip(image, 1)
        # Update x-coordinates in labels
        if label_file:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            with open(label_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w_norm, h_norm = parts
                        x = str(1.0 - float(x))  # Flip x coordinate
                        f.write(f"{class_id} {x} {y} {w_norm} {h_norm}\n")
    
    elif name == 'crop':
        crop_h = int(h * value)
        crop_w = int(w * value)
        y_start = random.randint(0, h - crop_h)
        x_start = random.randint(0, w - crop_w)
        image = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
        image = cv2.resize(image, (w, h))
    
    return image


def augment_dataset(source_dir, labels_dir, output_dir, count, person_out_id):
    """Generate augmented images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Finding Person_out images...")
    person_out_images = get_person_out_images(source_dir, labels_dir, person_out_id)
    
    if not person_out_images:
        logger.error(f"No images with Person_out class (ID: {person_out_id}) found!")
        return
    
    logger.info(f"Found {len(person_out_images)} images with Person_out detections")
    
    logger.info(f"Generating {count} augmented images...")
    for i in range(count):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{count}")
        
        # Random image
        img_path = random.choice(person_out_images)
        image = cv2.imread(str(img_path))
        
        if image is None:
            continue
        
        # Apply augmentation
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        augmented = apply_augmentations(image, label_path)
        
        # Save augmented image
        output_name = f"aug_{img_path.stem}_{i:05d}.jpg"
        output_image_path = output_path / output_name
        cv2.imwrite(str(output_image_path), augmented)
        
        # Copy label
        output_label_path = output_path / f"{output_name.replace('.jpg', '.txt')}"
        if label_path.exists():
            with open(label_path, 'r') as src:
                with open(output_label_path, 'w') as dst:
                    dst.write(src.read())
    
    logger.info(f"✓ Augmentation complete!")
    logger.info(f"  Generated images: {output_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Move images to: dataset/train/images/")
    logger.info(f"  2. Move labels to: dataset/train/labels/")
    logger.info(f"  3. Run: python scripts/train_yolov11.py --epochs 100")


def main():
    args = parse_args()
    augment_dataset(args.source, args.labels, args.output, args.count, args.person_out_id)


if __name__ == '__main__':
    main()
