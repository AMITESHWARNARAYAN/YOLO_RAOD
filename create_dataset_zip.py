"""
Create ZIP file of dataset for Google Colab upload
Works with organized train/validation folders
"""

import zipfile
from pathlib import Path
from datetime import datetime
import os


def check_organized_folders():
    """Check if train/validation folders are properly organized"""
    
    data_dir = Path("data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    
    if not train_dir.exists() or not val_dir.exists():
        print("✗ train/validation folders not found!")
        print("\n💡 Run 'python organize_dataset.py' first to organize your dataset")
        return False
    
    # Count images per class
    classes = ["Good", "Minor_Damage", "Pothole", "Crack", "Severe_Damage"]
    train_count = {}
    val_count = {}
    
    for class_name in classes:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        if train_class_dir.exists():
            train_count[class_name] = len(list(train_class_dir.glob("*.jpeg")))
        else:
            train_count[class_name] = 0
        
        if val_class_dir.exists():
            val_count[class_name] = len(list(val_class_dir.glob("*.jpeg")))
        else:
            val_count[class_name] = 0
    
    total_train = sum(train_count.values())
    total_val = sum(val_count.values())
    
    if total_train == 0 and total_val == 0:
        print("✗ No images found!")
        print("\n💡 Run 'python organize_dataset.py' first to organize your dataset")
        return False
    
    print("\n📊 Dataset Summary:")
    print("-" * 50)
    print(f"{'Class':<20} {'Train':<10} {'Validation':<10}")
    print("-" * 50)
    for class_name in classes:
        print(f"{class_name:<20} {train_count[class_name]:<10} {val_count[class_name]:<10}")
    print("-" * 50)
    print(f"{'Total':<20} {total_train:<10} {total_val:<10}")
    print("-" * 50)
    
    return True


def create_dataset_zip():
    """Create ZIP file of data folder for Colab upload"""
    
    print("\n" + "="*60)
    print("Create Dataset ZIP for Google Colab")
    print("="*60 + "\n")
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("✗ Data folder not found!")
        return None
    
    # Check if folders are organized
    if not check_organized_folders():
        return None
    
    # Count total images
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    
    train_count = sum(1 for _ in train_dir.rglob("*.jpeg") if _.is_file())
    val_count = sum(1 for _ in val_dir.rglob("*.jpeg") if _.is_file())
    total_images = train_count + val_count
    
    print(f"\n📦 Creating ZIP file...")
    print(f"   Training images:   {train_count}")
    print(f"   Validation images: {val_count}")
    print(f"   Total images:      {total_images}\n")
    
    # Create ZIP filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"road_dataset_{timestamp}.zip"
    
    print(f"📝 Filename: {zip_filename}")
    print("⏳ Please wait...")
    
    # Create ZIP file
    files_added = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from train and validation
        for split in ["train", "validation"]:
            split_dir = data_dir / split
            if split_dir.exists():
                for file_path in split_dir.rglob("*.jpeg"):
                    if file_path.is_file():
                        # Store with relative path
                        arcname = file_path.relative_to(data_dir.parent)
                        zipf.write(file_path, arcname)
                        files_added += 1
                        if files_added % 100 == 0:
                            print(f"   Added {files_added} files...")
    
    # Get file size
    zip_path = Path(zip_filename)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("✅ ZIP File Created Successfully!")
    print(f"{'='*60}")
    print(f"📄 Filename:  {zip_filename}")
    print(f"📦 Size:      {size_mb:.2f} MB")
    print(f"📁 Location:  {zip_path.absolute()}")
    print(f"📊 Files:     {files_added} images")
    print(f"\n{'='*60}")
    print("📝 Next Steps:")
    print(f"{'='*60}")
    print("1. Open train_road_model.ipynb in VS Code")
    print("2. Upload to Google Colab (File → Upload)")
    print("3. In Colab, run Step 2 - Option A (Upload ZIP)")
    print(f"4. Upload: {zip_filename}")
    print("5. Follow notebook instructions to train model")
    print(f"{'='*60}\n")
    
    return zip_filename


if __name__ == "__main__":
    create_dataset_zip()
