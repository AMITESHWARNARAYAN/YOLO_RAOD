"""
Prepare dataset for Google Colab training
Creates a zip file of yolo_data folder
"""

import os
import zipfile
from pathlib import Path

def zip_dataset(source_dir='yolo_data', output_file='yolo_data.zip'):
    """
    Create a zip file of the YOLO dataset for upload to Colab
    """
    if not os.path.exists(source_dir):
        print(f"❌ Error: {source_dir} folder not found!")
        print("Please run convert_to_yolo_format.py first.")
        return
    
    print(f"Creating {output_file}...")
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        source_path = Path(source_dir)
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Add file to zip with relative path
                arcname = file_path.relative_to(source_path.parent)
                zipf.write(file_path, arcname)
                
                # Progress indicator
                if file_path.suffix in ['.jpg', '.jpeg', '.png']:
                    print('.', end='', flush=True)
    
    print(f"\n\n✓ Created {output_file}")
    print(f"✓ Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"\nNext steps:")
    print(f"1. Upload {output_file} to Google Colab")
    print(f"2. Open train_yolov8_colab.ipynb in Colab")
    print(f"3. Run all cells to train the model")
    print(f"4. Download the trained model when complete")
    print(f"\nColab link: https://colab.research.google.com/")

if __name__ == '__main__':
    zip_dataset()
