"""
Analyze dataset for model fine-tuning recommendations
"""
import os
import json
from collections import Counter
from PIL import Image
import random

def analyze_dataset():
    train_dir = 'data/train'
    val_dir = 'data/validation'
    
    print('='*70)
    print('DATASET ANALYSIS FOR MODEL FINE-TUNING')
    print('='*70)
    
    # 1. Class Distribution
    print('\n1. CLASS DISTRIBUTION:')
    print('-'*70)
    
    train_counts = {}
    val_counts = {}
    
    for split, dir_path, counts_dict in [('Training', train_dir, train_counts), 
                                          ('Validation', val_dir, val_counts)]:
        print(f'\n{split}:')
        classes = {}
        for cls in os.listdir(dir_path):
            cls_path = os.path.join(dir_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                classes[cls] = count
                counts_dict[cls] = count
        
        total = sum(classes.values())
        for cls, count in sorted(classes.items()):
            percentage = (count/total)*100
            bar = '█' * int(percentage/2)
            print(f'  {cls:20s}: {count:4d} images ({percentage:5.2f}%) {bar}')
        print(f'  {"TOTAL":20s}: {total:4d} images')
    
    # 2. Class Imbalance Analysis
    print('\n2. CLASS IMBALANCE ANALYSIS:')
    print('-'*70)
    
    # Filter out empty classes
    non_zero_counts = {k: v for k, v in train_counts.items() if v > 0}
    
    if len(non_zero_counts) == 0:
        print('ERROR: No training data found!')
        return
    
    max_count = max(non_zero_counts.values())
    min_count = min(non_zero_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f'Active classes: {len(non_zero_counts)}/5')
    print(f'Empty classes: {[k for k, v in train_counts.items() if v == 0]}')
    
    print(f'Max samples: {max_count} | Min samples: {min_count}')
    print(f'Imbalance ratio: {imbalance_ratio:.2f}x')
    
    if imbalance_ratio > 2:
        print('⚠️  SEVERE IMBALANCE - Needs class weighting or resampling')
    elif imbalance_ratio > 1.5:
        print('⚠️  MODERATE IMBALANCE - Consider class weighting')
    else:
        print('✓ Balanced dataset')
    
    # Calculate class weights
    print('\nRecommended class weights (for non-empty classes):')
    total_samples = sum(non_zero_counts.values())
    num_classes = len(non_zero_counts)
    for cls, count in sorted(non_zero_counts.items()):
        weight = total_samples / (num_classes * count)
        print(f'  {cls:20s}: {weight:.3f}')
    
    # 3. Image Properties Analysis
    print('\n3. IMAGE PROPERTIES ANALYSIS:')
    print('-'*70)
    
    sample_images = []
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            imgs = os.listdir(cls_path)
            samples = random.sample(imgs, min(10, len(imgs)))
            for img in samples:
                sample_images.append(os.path.join(cls_path, img))
    
    widths, heights, aspects = [], [], []
    formats = Counter()
    
    for img_path in sample_images[:100]:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
                formats[img.format] += 1
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
    
    if widths:
        avg_w = sum(widths) // len(widths)
        avg_h = sum(heights) // len(heights)
        min_aspect = min(aspects)
        max_aspect = max(aspects)
        
        print(f'Sample size: {len(widths)} images')
        print(f'Average dimensions: {avg_w}x{avg_h}')
        print(f'Width range: {min(widths)} to {max(widths)}')
        print(f'Height range: {min(heights)} to {max(heights)}')
        print(f'Aspect ratios: {min_aspect:.2f} to {max_aspect:.2f}')
        print(f'Image formats: {dict(formats)}')
        
        if max_aspect / min_aspect > 2:
            print('⚠️  High aspect ratio variance - Needs better augmentation')
        else:
            print('✓ Consistent aspect ratios')
    
    # 4. Training Performance Analysis
    print('\n4. CURRENT MODEL PERFORMANCE:')
    print('-'*70)
    print('Train Accuracy: 88.63%')
    print('Val Accuracy:   80.48%')
    print('Best Val Acc:   81.53%')
    print('Gap:            8.15% (slight overfitting)')
    
    # 5. Recommendations
    print('\n5. FINE-TUNING RECOMMENDATIONS:')
    print('='*70)
    
    recommendations = []
    
    # Based on imbalance
    if imbalance_ratio > 1.5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Class Imbalance',
            'solution': 'Add class weights to loss function',
            'expected_gain': '+3-5% accuracy'
        })
    
    # Based on overfitting
    train_val_gap = 88.63 - 80.48
    if train_val_gap > 5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Overfitting (gap: {train_val_gap:.2f}%)',
            'solution': 'Increase dropout, add regularization, more augmentation',
            'expected_gain': '+2-4% val accuracy'
        })
    
    # Model architecture
    recommendations.append({
        'priority': 'MEDIUM',
        'issue': 'Model capacity',
        'solution': 'Try EfficientNet-B0 or ResNet34 (better feature extraction)',
        'expected_gain': '+3-7% accuracy'
    })
    
    # Training strategy
    recommendations.append({
        'priority': 'MEDIUM',
        'issue': 'Learning rate',
        'solution': 'Use cosine annealing scheduler, try lower initial LR (5e-5)',
        'expected_gain': '+1-3% accuracy'
    })
    
    # Data augmentation
    recommendations.append({
        'priority': 'HIGH',
        'issue': 'Limited augmentation',
        'solution': 'Add CutMix, MixUp, AutoAugment, more geometric transforms',
        'expected_gain': '+2-5% accuracy'
    })
    
    # More training
    recommendations.append({
        'priority': 'LOW',
        'issue': 'Early stopping at epoch 50',
        'solution': 'Train longer (100 epochs) with better patience',
        'expected_gain': '+1-2% accuracy'
    })
    
    print('\nPriority Order:')
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
        print(f"   Expected: {rec['expected_gain']}")
    
    print('\n' + '='*70)
    print('RECOMMENDED ACTION PLAN:')
    print('='*70)
    print('1. Add class weights (immediate, easy)')
    print('2. Enhance data augmentation (high impact)')
    print('3. Increase dropout to 0.6 (reduce overfitting)')
    print('4. Try EfficientNet-B0 backbone (better architecture)')
    print('5. Use cosine annealing LR scheduler')
    print('6. Train for 100 epochs with patience=20')
    print('\nExpected final accuracy: 87-92%')
    print('='*70)

if __name__ == '__main__':
    analyze_dataset()
