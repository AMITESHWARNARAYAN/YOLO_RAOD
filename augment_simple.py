"""
Simple data augmentation using only PIL - no complex dependencies
Generates 5x more training data from existing images
"""
import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import shutil
from tqdm import tqdm

def simple_augment(image, aug_type):
    """Apply simple PIL-based augmentation"""
    if aug_type == 0:
        # Horizontal flip
        return ImageOps.mirror(image)
    
    elif aug_type == 1:
        # Rotate 15-30 degrees
        angle = random.choice([15, -15, 20, -20, 25, -25, 30, -30])
        return image.rotate(angle, expand=False, fillcolor='black')
    
    elif aug_type == 2:
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    elif aug_type == 3:
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.8, 1.4)
        return enhancer.enhance(factor)
    
    elif aug_type == 4:
        # Color saturation
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    elif aug_type == 5:
        # Gaussian blur
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    
    elif aug_type == 6:
        # Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        factor = random.uniform(0.5, 2.0)
        return enhancer.enhance(factor)
    
    elif aug_type == 7:
        # Vertical flip
        return ImageOps.flip(image)
    
    elif aug_type == 8:
        # Combined: brightness + rotation
        img = image.rotate(random.randint(-20, 20), expand=False)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(0.8, 1.2))
    
    else:
        # Combined: contrast + blur
        enhancer = ImageEnhance.Contrast(image)
        img = enhancer.enhance(random.uniform(0.9, 1.3))
        return img.filter(ImageFilter.GaussianBlur(radius=0.8))

def augment_dataset_simple(source_dir, output_dir, augments_per_image=5):
    """
    Augment dataset using simple PIL transforms
    
    Args:
        source_dir: Original data directory
        output_dir: Output directory for augmented data
        augments_per_image: Number of augmented versions per image
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_original = 0
    total_augmented = 0
    
    print(f"\n{'='*70}")
    print(f"AUGMENTING: {source_dir}")
    print(f"{'='*70}\n")
    
    # Process each class
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        # Get images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        if len(images) == 0:
            print(f"⏭️  Skipping empty class: {class_dir.name}")
            continue
        
        # Create output directory
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Class: {class_dir.name}")
        print(f"   Original: {len(images)} images")
        
        # Copy originals
        for img_path in images:
            shutil.copy2(img_path, output_class_dir / img_path.name)
            total_original += 1
        
        # Generate augmented versions
        print(f"   Generating {augments_per_image} augmented versions per image...")
        for img_path in tqdm(images, desc="   Augmenting"):
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Generate multiple augmented versions
                for i in range(augments_per_image):
                    # Apply random augmentation
                    aug_type = random.randint(0, 9)
                    aug_image = simple_augment(image, aug_type)
                    
                    # Save augmented image
                    aug_filename = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                    aug_path = output_class_dir / aug_filename
                    aug_image.save(aug_path, quality=95)
                    total_augmented += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing {img_path.name}: {e}")
        
        final_count = len(list(output_class_dir.glob('*')))
        increase = ((final_count / len(images)) - 1) * 100
        print(f"   ✓ Total: {final_count} images (+{increase:.0f}%)")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Original images:   {total_original:,}")
    print(f"Augmented images:  {total_augmented:,}")
    print(f"Total images:      {total_original + total_augmented:,}")
    print(f"Dataset size:      {(total_original + total_augmented) / total_original:.1f}x larger")

def main():
    print("\n" + "="*70)
    print("SIMPLE DATASET AUGMENTATION")
    print("="*70)
    print("\nUsing PIL-based augmentations:")
    print("  ✓ Flips (horizontal/vertical)")
    print("  ✓ Rotations (15-30 degrees)")
    print("  ✓ Brightness adjustment")
    print("  ✓ Contrast adjustment")
    print("  ✓ Color saturation")
    print("  ✓ Blur and sharpness")
    print("  ✓ Combined transforms")
    
    # Augment training set
    print("\n" + "="*70)
    print("1️⃣  AUGMENTING TRAINING SET")
    print("="*70)
    augment_dataset_simple(
        source_dir='data/train',
        output_dir='data/train_augmented',
        augments_per_image=5
    )
    
    # Augment validation set (fewer augmentations)
    print("\n" + "="*70)
    print("2️⃣  AUGMENTING VALIDATION SET")
    print("="*70)
    augment_dataset_simple(
        source_dir='data/validation',
        output_dir='data/validation_augmented',
        augments_per_image=2
    )
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*70)
    print("\nAugmented data saved to:")
    print("  📁 data/train_augmented/")
    print("  📁 data/validation_augmented/")
    
    print("\n📋 Next steps:")
    print("  1. Create new ZIP with augmented data:")
    print("     python -c \"import shutil; shutil.make_archive('road_dataset_augmented', 'zip', 'data')\"")
    print("\n  2. Upload to Google Drive")
    print("\n  3. Update notebook paths:")
    print("     TRAIN_DIR = '/content/data/train_augmented'")
    print("     VAL_DIR = '/content/data/validation_augmented'")
    print("\n  4. Retrain model")
    
    print("\n💡 Expected improvements:")
    print("  • +3-5% accuracy from 6x more training data")
    print("  • Better generalization from diverse augmentations")
    print("  • Reduced overfitting")
    print("="*70)

if __name__ == '__main__':
    main()
