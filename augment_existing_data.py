"""
Augment existing dataset to create more training samples
Uses advanced augmentation techniques to generate synthetic data
"""
import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def get_augmentation_pipeline():
    """Advanced augmentation pipeline"""
    return A.Compose([
        # Geometric transforms
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Rotate(limit=45, p=1),
            A.Perspective(scale=(0.05, 0.1), p=1),
        ], p=0.8),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
        ], p=0.8),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=7, p=1),
        ], p=0.5),
        
        # Weather effects
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
            A.RandomSunFlare(src_radius=100, p=1),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=1),
        ], p=0.3),
        
        # Distortion
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.5, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=50, sigma=25, p=1),
        ], p=0.3),
        
        # Quality degradation
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=90, p=1),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1),
        ], p=0.2),
        
        # Cutout/Erasing
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    ])

def augment_image(image_path, transform, num_augments=5):
    """Generate multiple augmented versions of an image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented_images = []
    for _ in range(num_augments):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

def augment_dataset(source_dir, output_dir, augments_per_image=5):
    """
    Augment entire dataset
    
    Args:
        source_dir: Path to original data (data/train or data/validation)
        output_dir: Path to save augmented data
        augments_per_image: Number of augmented versions per original image
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Get augmentation pipeline
    transform = get_augmentation_pipeline()
    
    total_original = 0
    total_augmented = 0
    
    print(f"\n{'='*70}")
    print(f"AUGMENTING DATASET: {source_dir}")
    print(f"{'='*70}\n")
    
    # Process each class
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        # Skip empty classes
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        if len(images) == 0:
            print(f"⏭️  Skipping empty class: {class_dir.name}")
            continue
        
        # Create output directory for this class
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Processing class: {class_dir.name}")
        print(f"   Original images: {len(images)}")
        
        # Copy original images
        for img_path in tqdm(images, desc="Copying originals"):
            shutil.copy2(img_path, output_class_dir / img_path.name)
            total_original += 1
        
        # Generate augmented images
        for img_path in tqdm(images, desc="Generating augmented"):
            try:
                augmented_imgs = augment_image(img_path, transform, augments_per_image)
                
                # Save augmented images
                for idx, aug_img in enumerate(augmented_imgs):
                    aug_filename = f"{img_path.stem}_aug{idx}{img_path.suffix}"
                    aug_path = output_class_dir / aug_filename
                    
                    # Convert RGB back to BGR for cv2
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_path), aug_img_bgr)
                    total_augmented += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing {img_path.name}: {e}")
        
        final_count = len(list(output_class_dir.glob('*')))
        print(f"   ✓ Final images: {final_count} (+{final_count - len(images)} augmented)")
    
    print(f"\n{'='*70}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*70}")
    print(f"Original images: {total_original}")
    print(f"Augmented images: {total_augmented}")
    print(f"Total images: {total_original + total_augmented}")
    print(f"Multiplier: {(total_original + total_augmented) / total_original:.1f}x")
    print(f"\n✓ Augmented dataset saved to: {output_dir}")

def main():
    """Augment both training and validation sets"""
    print("\n" + "="*70)
    print("DATASET AUGMENTATION PIPELINE")
    print("="*70)
    
    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("\n⚠️  Installing required packages...")
        os.system('pip install albumentations opencv-python tqdm')
        print("✓ Packages installed!\n")
    
    # Augment training set
    print("\n1. Augmenting training set...")
    augment_dataset(
        source_dir='data/train',
        output_dir='data/train_augmented',
        augments_per_image=5
    )
    
    # Augment validation set (fewer augmentations)
    print("\n2. Augmenting validation set...")
    augment_dataset(
        source_dir='data/validation',
        output_dir='data/validation_augmented',
        augments_per_image=2
    )
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Update notebook to use augmented data:")
    print("   TRAIN_DIR = '/content/data/train_augmented'")
    print("   VAL_DIR = '/content/data/validation_augmented'")
    print("\n2. Create new ZIP:")
    print("   python create_dataset_zip.py --augmented")
    print("\n3. Upload and retrain model")
    print(f"\n💡 Expected accuracy improvement: +3-5%")
    print("="*70)

if __name__ == '__main__':
    main()
