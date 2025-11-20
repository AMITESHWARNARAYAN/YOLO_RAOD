"""
Fetch additional road condition datasets to improve training
Combines multiple sources for better accuracy
"""
import os
import requests
from pathlib import Path
import zipfile
import json

def download_file(url, dest_path):
    """Download file with progress"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = (downloaded / total_size) * 100
                    print(f'\rDownloading: {progress:.1f}%', end='')
    print('\n✓ Download complete!')

def fetch_additional_datasets():
    """
    Fetch additional road condition datasets from public sources
    """
    print("="*70)
    print("FETCHING ADDITIONAL TRAINING DATA")
    print("="*70)
    
    datasets = {
        "1. RDD2020 (Road Damage Dataset)": {
            "description": "14,000+ images from India, Japan, Czech Republic",
            "classes": ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"],
            "size": "~2.5 GB",
            "source": "Kaggle/IEEE BigData",
            "url": "https://github.com/sekilab/RoadDamageDetector/",
            "instructions": [
                "1. Go to: https://www.kaggle.com/datasets/saumyapatel/road-damage-detection",
                "2. Sign in to Kaggle",
                "3. Click 'Download' button",
                "4. Extract to: data/rdd2020/",
                "5. Run: python merge_datasets.py --source rdd2020"
            ]
        },
        
        "2. RDDC (Road Damage Detection Challenge)": {
            "description": "5,000+ annotated road images",
            "classes": ["D00 (Crack)", "D10 (Line Crack)", "D20 (Alligator)", "D40 (Pothole)"],
            "size": "~1 GB",
            "source": "IEEE BigData Cup",
            "url": "https://github.com/sekilab/RoadDamageDetector",
            "instructions": [
                "1. Visit: https://github.com/sekilab/RoadDamageDetector",
                "2. Download 'RoadDamageDataset' folder",
                "3. Extract to: data/rddc/",
                "4. Run: python merge_datasets.py --source rddc"
            ]
        },
        
        "3. Crack500 Dataset": {
            "description": "500 road crack images with pixel-level annotations",
            "classes": ["Crack"],
            "size": "~300 MB",
            "source": "Academic Research",
            "url": "https://github.com/fyangneil/pavement-crack-detection",
            "instructions": [
                "1. Visit: https://github.com/fyangneil/pavement-crack-detection",
                "2. Download dataset",
                "3. Extract to: data/crack500/",
                "4. Run: python merge_datasets.py --source crack500"
            ]
        },
        
        "4. CFD (Crack Forest Dataset)": {
            "description": "118 road crack images, high quality",
            "classes": ["Crack"],
            "size": "~50 MB",
            "source": "Academic Research",
            "url": "https://github.com/cuilimeng/CrackForest-dataset",
            "instructions": [
                "1. Visit: https://github.com/cuilimeng/CrackForest-dataset",
                "2. Clone or download ZIP",
                "3. Extract to: data/cfd/",
                "4. Run: python merge_datasets.py --source cfd"
            ]
        },
        
        "5. Augmentation (Synthetic Data)": {
            "description": "Generate 5,000+ synthetic images from existing data",
            "classes": ["All existing classes"],
            "size": "~1 GB",
            "source": "Augmentation pipeline",
            "url": "Local generation",
            "instructions": [
                "Run: python augment_existing_data.py",
                "This will create augmented versions of your current data"
            ]
        }
    }
    
    print("\n📦 AVAILABLE DATASETS:\n")
    for name, info in datasets.items():
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Classes: {', '.join(info['classes'])}")
        print(f"  Size: {info['size']}")
        print(f"  Source: {info['source']}")
        print(f"\n  Instructions:")
        for instruction in info['instructions']:
            print(f"    {instruction}")
    
    print("\n" + "="*70)
    print("QUICK START RECOMMENDATIONS:")
    print("="*70)
    print("\n🎯 Best strategy for quick improvement:")
    print("\n1. EASIEST: Run augmentation (generates synthetic data immediately)")
    print("   Command: python augment_existing_data.py")
    print("   Time: 10-15 minutes")
    print("   Gain: +2-4% accuracy (5,000 new images)")
    print("\n2. MOST EFFECTIVE: Download RDD2020 (largest dataset)")
    print("   Download from Kaggle")
    print("   Time: 30-45 minutes")
    print("   Gain: +5-8% accuracy (14,000 new images)")
    print("\n3. BALANCED: Use both augmentation + RDD2020")
    print("   Expected final accuracy: 90-95%")
    print("="*70)
    
    # Create data directories
    os.makedirs('data/external', exist_ok=True)
    print("\n✓ Created data/external/ directory for new datasets")
    
    return datasets

if __name__ == '__main__':
    datasets = fetch_additional_datasets()
    
    print("\n💡 TIP: Start with augmentation while downloading larger datasets!")
    print("    Run: python augment_existing_data.py")
