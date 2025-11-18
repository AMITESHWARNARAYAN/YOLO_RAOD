"""
Model Deployment Script - After training on Google Colab
"""

import os
import shutil
from pathlib import Path


def check_model_downloaded():
    """Check if model is in Downloads folder"""
    downloads = Path.home() / "Downloads"
    model_files = list(downloads.glob("road_condition_model*.pth"))
    
    if model_files:
        print(f"✓ Found model: {model_files[0]}")
        return model_files[0]
    else:
        print("✗ Model not found in Downloads folder")
        print("\nPlease download the trained model from Colab:")
        print("1. In Colab, run the download cell")
        print("2. Save 'road_condition_model_best.pth' to Downloads")
        return None


def copy_model_to_project(source_path):
    """Copy model to project models directory"""
    project_dir = Path(__file__).parent
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    dest_path = models_dir / "road_condition_model.pth"
    
    shutil.copy2(source_path, dest_path)
    print(f"✓ Model copied to: {dest_path}")
    
    return dest_path


def verify_pytorch_installed():
    """Verify PyTorch is installed"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("\nInstall with: pip install -r requirements.txt")
        return False


def verify_dependencies():
    """Check all dependencies"""
    print("\n=== Checking Dependencies ===")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'googlemaps': 'Google Maps API',
        'numpy': 'NumPy'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def run_model_verification():
    """Test model loading"""
    print("\n=== Testing Model ===")
    
    try:
        from road_condition_model import RoadConditionModel
        
        model = RoadConditionModel()
        model.load_model()
        
        print("✓ Model loads successfully!")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("Road Condition Model Deployment")
    print("="*60)
    
    # Step 1: Check dependencies
    if not verify_dependencies():
        return
    
    # Step 2: Find downloaded model
    print("\n=== Locating Model ===")
    model_path = check_model_downloaded()
    
    if not model_path:
        return
    
    # Step 3: Copy to project
    print("\n=== Deploying Model ===")
    dest_path = copy_model_to_project(model_path)
    
    # Step 4: Verify installation
    if run_model_verification():
        print("\n" + "="*60)
        print("✓ DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print("\nNext steps:")
        print("1. Configure Google Maps API key in config.py")
        print("2. Test detection: python main.py --image test.jpg")
        print("3. Start live monitoring: python main.py")
        print("="*60)
    else:
        print("\n✗ Deployment incomplete - model verification failed")


if __name__ == "__main__":
    main()
