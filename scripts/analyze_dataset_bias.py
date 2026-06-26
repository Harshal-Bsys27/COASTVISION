#!/usr/bin/env python3
"""
Dataset Bias Analysis & Collection Guide
=========================================
Analyzes current class distribution and provides detailed expansion roadmap.

Usage:
    python scripts/analyze_dataset_bias.py --detailed
"""

import logging
from pathlib import Path
from collections import defaultdict
import argparse

logging.basicConfig(level=logging.INFO, format='[Analysis] %(message)s')
logger = logging.getLogger(__name__)


def analyze_class_distribution():
    """Detailed class distribution analysis."""
    logger.info("=" * 80)
    logger.info("DATASET BIAS ANALYSIS - Current State")
    logger.info("=" * 80)
    
    splits = ['train', 'valid', 'test']
    all_stats = {}
    
    for split in splits:
        labels_dir = Path(f'dataset/{split}/labels')
        if not labels_dir.exists():
            continue
        
        class_counts = defaultdict(int)
        image_count = 0
        
        for label_file in labels_dir.glob('*.txt'):
            image_count += 1
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        
        all_stats[split] = {
            'images': image_count,
            'classes': dict(class_counts)
        }
    
    # Display results
    class_names = {0: 'Drowning', 1: 'Person_out', 2: 'Swimming'}
    
    logger.info("\n" + "=" * 80)
    logger.info("CURRENT DISTRIBUTION")
    logger.info("=" * 80)
    
    total_by_class = defaultdict(int)
    
    for split in splits:
        if split in all_stats:
            logger.info(f"\n{split.upper()}:")
            stats = all_stats[split]
            total_annotations = sum(stats['classes'].values())
            
            logger.info(f"  Images: {stats['images']}")
            logger.info(f"  Total annotations: {total_annotations}")
            logger.info(f"  Avg boxes/image: {total_annotations / stats['images']:.2f}")
            logger.info(f"\n  Class Breakdown:")
            
            for class_id in range(3):
                count = stats['classes'].get(class_id, 0)
                class_name = class_names[class_id]
                pct = (count / total_annotations * 100) if total_annotations > 0 else 0
                logger.info(f"    {class_name:20s}: {count:6d} ({pct:5.1f}%)")
                total_by_class[class_id] += count
    
    # Overall balance
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL IMBALANCE METRICS")
    logger.info("=" * 80)
    
    total = sum(total_by_class.values())
    
    logger.info(f"\nTotal annotations: {total}")
    for class_id in range(3):
        count = total_by_class[class_id]
        class_name = class_names[class_id]
        pct = (count / total * 100) if total > 0 else 0
        logger.info(f"  {class_name:20s}: {count:6d} ({pct:5.1f}%)")
    
    # Calculate imbalance ratios
    drowning_count = total_by_class[0]
    person_out_count = total_by_class[1]
    swimming_count = total_by_class[2]
    
    logger.info("\nImbalance Ratios:")
    if person_out_count > 0:
        logger.info(f"  Drowning vs Person_out: {drowning_count/person_out_count:.2f}:1")
        logger.info(f"  Swimming vs Person_out: {swimming_count/person_out_count:.2f}:1")
    
    # Ideal balance calculation
    logger.info("\n" + "=" * 80)
    logger.info("TARGET BALANCED DISTRIBUTION (Recommended)")
    logger.info("=" * 80)
    
    target_total = 40000
    target_drowning = int(target_total * 0.30)  # 30% - most critical
    target_person_out = int(target_total * 0.35)  # 35% - important context
    target_swimming = int(target_total * 0.35)  # 35% - normal activity
    
    logger.info(f"\nTarget total annotations: {target_total}")
    logger.info(f"  Drowning (critical):      {target_drowning:6d} (30%) - Current: {drowning_count:6d} | Need: {max(0, target_drowning - drowning_count):+6d}")
    logger.info(f"  Person_out (context):     {target_person_out:6d} (35%) - Current: {person_out_count:6d} | Need: {max(0, target_person_out - person_out_count):+6d}")
    logger.info(f"  Swimming (normal):        {target_swimming:6d} (35%) - Current: {swimming_count:6d} | Need: {max(0, target_swimming - swimming_count):+6d}")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("EXPANSION RECOMMENDATIONS")
    logger.info("=" * 80)
    
    needs = {
        'Drowning': max(0, target_drowning - drowning_count),
        'Person_out': max(0, target_person_out - person_out_count),
        'Swimming': max(0, target_swimming - swimming_count),
    }
    
    logger.info("\nPriority: HIGH → LOW")
    sorted_needs = sorted(needs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (class_name, needed) in enumerate(sorted_needs, 1):
        if needed > 0:
            estimated_images = int(needed / 1.5)  # ~1.5 boxes per image average
            logger.info(f"\n{i}. {class_name}")
            logger.info(f"   Need: {needed} annotations")
            logger.info(f"   ≈ {estimated_images} images to collect")
            
            if class_name == 'Drowning':
                logger.info(f"   Priority: CRITICAL (life-saving)")
                logger.info(f"   Sources: Rescue training videos, water safety footage")
            elif class_name == 'Person_out':
                logger.info(f"   Priority: HIGH (context detection)")
                logger.info(f"   Sources: Beach scenes, people standing in water")
            else:
                logger.info(f"   Priority: MEDIUM (baseline activity)")
                logger.info(f"   Sources: Swimming videos, recreational water content")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset bias')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    args = parser.parse_args()
    
    analyze_class_distribution()


if __name__ == '__main__':
    main()
