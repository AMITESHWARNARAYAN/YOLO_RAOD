"""
Download and organize dataset from GitHub repository
"""

import os
import requests
from pathlib import Path
import shutil
from urllib.parse import urljoin
import time


def download_file(url, dest_path):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return False


def download_github_dataset():
    """Download dataset from GitHub repository"""
    
    print("\n" + "="*60)
    print("Downloading Road Condition Dataset from GitHub")
    print("="*60)
    print("Source: github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset")
    print()
    
    # GitHub raw content base URL
    base_url = "https://raw.githubusercontent.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset/master/Dataset/"
    
    # Dataset structure from the repository
    dataset_files = {
        "Crack": [
            "01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg",
            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg"
        ],
        "Pothole": [
            "01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg",
            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg"
        ],
        "Normal": [  # This will be mapped to "Good"
            "01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg", "07.jpg", "08.jpg", "09.jpg", "10.jpg",
            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg"
        ]
    }
    
    # Create data directories
    base_dir = Path("data")
    
    total_downloaded = 0
    total_failed = 0
    
    for category, files in dataset_files.items():
        # Map category names
        if category == "Normal":
            class_name = "Good"
        elif category == "Crack":
            class_name = "Crack"
        elif category == "Pothole":
            class_name = "Pothole"
        
        print(f"\n{'='*60}")
        print(f"Downloading: {category} ({class_name})")
        print(f"{'='*60}")
        
        # Calculate split (80% train, 20% validation)
        total = len(files)
        train_count = int(total * 0.8)
        
        train_files = files[:train_count]
        val_files = files[train_count:]
        
        # Download training files
        train_dir = base_dir / "train" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining set ({len(train_files)} images):")
        for filename in train_files:
            url = urljoin(base_url + category + "/", filename)
            dest_path = train_dir / f"{category}_{filename}"
            
            print(f"  Downloading: {filename}...", end=" ")
            if download_file(url, dest_path):
                print("✓")
                total_downloaded += 1
            else:
                print("✗")
                total_failed += 1
            
            time.sleep(0.2)  # Be nice to GitHub
        
        # Download validation files
        val_dir = base_dir / "validation" / class_name
        val_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nValidation set ({len(val_files)} images):")
        for filename in val_files:
            url = urljoin(base_url + category + "/", filename)
            dest_path = val_dir / f"{category}_{filename}"
            
            print(f"  Downloading: {filename}...", end=" ")
            if download_file(url, dest_path):
                print("✓")
                total_downloaded += 1
            else:
                print("✗")
                total_failed += 1
            
            time.sleep(0.2)  # Be nice to GitHub
    
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {total_downloaded} images")
    print(f"Failed: {total_failed} images")
    print()
    
    # Show statistics
    print("Dataset Statistics:")
    for split in ["train", "validation"]:
        for class_name in ["Good", "Crack", "Pothole"]:
            class_dir = base_dir / split / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.jpg")))
                print(f"  {split}/{class_name}: {count} images")
    
    print("\n" + "="*60)
    print("⚠ Note: This dataset only contains 3 classes:")
    print("  - Good (Normal roads)")
    print("  - Crack")
    print("  - Pothole")
    print()
    print("You still need to add images for:")
    print("  - Minor_Damage")
    print("  - Severe_Damage")
    print("="*60)
    print("\nNext steps:")
    print("1. Add images for missing classes (Minor_Damage, Severe_Damage)")
    print("2. Run: python create_dataset_zip.py")
    print("3. Upload to Google Colab for training")
    print("="*60)


if __name__ == "__main__":
    download_github_dataset()
