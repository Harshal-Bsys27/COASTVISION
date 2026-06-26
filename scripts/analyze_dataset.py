#!/usr/bin/env python3
"""
Dataset Analysis & Class Distribution Checker
==============================================
Analyzes current dataset and identifies class imbalances.
"""

import os
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='[Dataset] %(message)s')
logger = logging.getLogger(__name__)


def analyze_labels(split='train'):
    """Analyze class distribution in labels."""
    labels_dir = Path(f'dataset/{split}/labels')
    
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return None
    
    class_counts = defaultdict(int)
    image_counts = 0
    annotation_counts = 0
    
    for label_file in labels_dir.glob('*.txt'):
        image_counts += 1
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    annotation_counts += 1
    
    return {
        'images': image_counts,
        'annotations': annotation_counts,
        'classes': dict(class_counts)
    }


def main():
    """Main analysis."""
    logger.info("=" * 60)
    logger.info("DATASET ANALYSIS - Class Distribution")
    logger.info("=" * 60)
    
    class_names = {0: 'Drowning', 1: 'Person out of water', 2: 'Swimming'}
    
    total_stats = {'Drowning': 0, 'Person out of water': 0, 'Swimming': 0}
    
    for split in ['train', 'valid', 'test']:
        logger.info(f"\n{split.upper()} Split:")
        stats = analyze_labels(split)
        
        if stats:
            logger.info(f"  Images: {stats['images']}")
            logger.info(f"  Total annotations: {stats['annotations']}")
            logger.info(f"  Average per image: {stats['annotations'] / stats['images']:.2f}")
            
            logger.info("  Class distribution:")
            for class_id in range(3):
                count = stats['classes'].get(class_id, 0)
                class_name = class_names[class_id]
                percent = (count / stats['annotations'] * 100) if stats['annotations'] > 0 else 0
                logger.info(f"    {class_name:20s}: {count:6d} ({percent:5.1f}%)")
                total_stats[class_name] += count
    
    # Total summary
    logger.info("\n" + "=" * 60)
    logger.info("TOTAL DISTRIBUTION:")
    total_annotations = sum(total_stats.values())
    for class_name, count in total_stats.items():
        percent = (count / total_annotations * 100) if total_annotations > 0 else 0
        logger.info(f"  {class_name:20s}: {count:6d} ({percent:5.1f}%)")
    
    # Imbalance analysis
    logger.info("\n" + "=" * 60)
    logger.info("IMBALANCE ANALYSIS:")
    person_out_count = total_stats['Person out of water']
    drowning_count = total_stats['Drowning']
    swimming_count = total_stats['Swimming']
    
    drowning_ratio = drowning_count / person_out_count if person_out_count > 0 else 0
    swimming_ratio = swimming_count / person_out_count if person_out_count > 0 else 0
    
    logger.info(f"  Drowning   vs Person_out: {drowning_ratio:.2f}x")
    logger.info(f"  Swimming   vs Person_out: {swimming_ratio:.2f}x")
    
    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("EXPANSION RECOMMENDATIONS:")
    
    target_person_out = 3000
    if person_out_count < target_person_out:
        needed = target_person_out - person_out_count
        logger.info(f"  ⚠️  Person_out class needs expansion:")
        logger.info(f"      Current: {person_out_count} annotations")
        logger.info(f"      Target: {target_person_out} annotations")
        logger.info(f"      Need to add: ~{needed} annotations")
        logger.info(f"      Estimated images to collect: {needed // 1.5:.0f} (@ 1.5 boxes/image)")
    
    logger.info("\n" + "=" * 60)


if __name__ == '__main__':
    main()
