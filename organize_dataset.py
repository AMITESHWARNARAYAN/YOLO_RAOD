"""
Dataset Organization Script for Road Condition Detection
Processes JSON annotations and organizes images into train/validation folders
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Directories
IMG_DIR = "data/img"
ANN_DIR = "data/ann"
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"

# Class mapping: JSON label -> Our model class
CLASS_MAPPING = {
    "pothole": "Pothole",
    "alligator crack": "Severe_Damage",  # Alligator cracks are severe
    "longitudinal crack": "Crack",
    "lateral crack": "Crack",
    # Images with no defects will be labeled as "Good"
}

# Our 5 classes for the model
MODEL_CLASSES = ["Good", "Minor_Damage", "Pothole", "Crack", "Severe_Damage"]

# Train/validation split ratio
TRAIN_RATIO = 0.8


def load_annotation(ann_path):
    """Load and parse JSON annotation file"""
    with open(ann_path, 'r') as f:
        return json.load(f)


def get_image_class(ann_data):
    """
    Determine the class of an image based on its annotations.
    Priority: Pothole > Severe_Damage > Crack > Minor_Damage > Good
    """
    if not ann_data.get("objects"):
        return "Good"
    
    detected_classes = set()
    for obj in ann_data["objects"]:
        class_title = obj.get("classTitle", "")
        if class_title in CLASS_MAPPING:
            detected_classes.add(CLASS_MAPPING[class_title])
    
    # Priority-based class selection
    if "Pothole" in detected_classes:
        return "Pothole"
    elif "Severe_Damage" in detected_classes:
        return "Severe_Damage"
    elif "Crack" in detected_classes:
        return "Crack"
    elif "Minor_Damage" in detected_classes:
        return "Minor_Damage"
    else:
        return "Good"


def organize_dataset():
    """Main function to organize dataset"""
    print("\n" + "="*60)
    print("Road Condition Dataset Organization")
    print("="*60 + "\n")
    
    # Create train/validation directories
    for split in [TRAIN_DIR, VAL_DIR]:
        for class_name in MODEL_CLASSES:
            os.makedirs(os.path.join(split, class_name), exist_ok=True)
    
    # Collect all image-class pairs
    image_class_map = {}
    no_annotation = []
    
    print("📋 Processing annotations...")
    for img_file in Path(IMG_DIR).glob("*.jpeg"):
        img_name = img_file.name
        ann_file = Path(ANN_DIR) / f"{img_name}.json"
        
        if ann_file.exists():
            try:
                ann_data = load_annotation(ann_file)
                img_class = get_image_class(ann_data)
                image_class_map[img_name] = img_class
            except Exception as e:
                print(f"⚠️  Error processing {img_name}: {e}")
        else:
            no_annotation.append(img_name)
    
    if no_annotation:
        print(f"\n⚠️  Warning: {len(no_annotation)} images have no annotations")
        print("   These will be skipped.")
    
    # Group images by class
    class_images = defaultdict(list)
    for img_name, img_class in image_class_map.items():
        class_images[img_class].append(img_name)
    
    # Display class distribution
    print("\n📊 Class Distribution:")
    print("-" * 40)
    for class_name in MODEL_CLASSES:
        count = len(class_images[class_name])
        print(f"  {class_name:20s}: {count:4d} images")
    print("-" * 40)
    print(f"  {'Total':20s}: {len(image_class_map):4d} images\n")
    
    # Split and copy images
    train_count = defaultdict(int)
    val_count = defaultdict(int)
    
    print("📦 Organizing images into train/validation splits...")
    for class_name in MODEL_CLASSES:
        images = class_images[class_name]
        if not images:
            print(f"  ⚠️  No images found for class: {class_name}")
            continue
        
        # Shuffle for random split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train directory
        for img_name in train_images:
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(TRAIN_DIR, class_name, img_name)
            shutil.copy2(src, dst)
            train_count[class_name] += 1
        
        # Copy to validation directory
        for img_name in val_images:
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(VAL_DIR, class_name, img_name)
            shutil.copy2(src, dst)
            val_count[class_name] += 1
        
        print(f"  ✓ {class_name:20s}: {len(train_images):4d} train, {len(val_images):4d} val")
    
    # Summary
    total_train = sum(train_count.values())
    total_val = sum(val_count.values())
    
    print("\n" + "="*60)
    print("✅ Dataset Organization Complete!")
    print("="*60)
    print(f"\n📁 Training images:   {total_train:4d}")
    print(f"📁 Validation images: {total_val:4d}")
    print(f"📁 Total organized:   {total_train + total_val:4d}\n")
    
    print("📂 Directory structure created:")
    print(f"   {TRAIN_DIR}/")
    for class_name in MODEL_CLASSES:
        print(f"      {class_name}/ ({train_count[class_name]} images)")
    print(f"\n   {VAL_DIR}/")
    for class_name in MODEL_CLASSES:
        print(f"      {class_name}/ ({val_count[class_name]} images)")
    
    print("\n💡 Next step: Run 'python create_dataset_zip.py' to create ZIP for Colab")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if directories exist
    if not os.path.exists(IMG_DIR):
        print(f"❌ Error: Image directory '{IMG_DIR}' not found!")
        exit(1)
    
    if not os.path.exists(ANN_DIR):
        print(f"❌ Error: Annotation directory '{ANN_DIR}' not found!")
        exit(1)
    
    organize_dataset()
