"""
Model Verification Tool - Test trained model
"""

import os
import torch
from pathlib import Path
import argparse
from road_condition_model import RoadConditionModel
from config import MODEL_PATH


def verify_model(model_path=MODEL_PATH):
    """Verify model file exists and loads correctly"""
    print("\n=== Model Verification ===")
    
    # Check file exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("\nTrain your model first:")
        print("1. Open train_road_model.ipynb in VS Code")
        print("2. Connect to Google Colab")
        print("3. Upload dataset and train")
        print("4. Download trained model")
        print("5. Run: python deploy_model.py")
        return False
    
    print(f"✓ Model file exists: {model_path}")
    
    # Check file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    # Try loading model
    try:
        model = RoadConditionModel()
        model.load_model(model_path)
        print("✓ Model loads successfully")
        
        # Check device
        device = model.device
        print(f"  Device: {device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_model_prediction(model_path=MODEL_PATH, image_path=None):
    """Test model prediction on an image"""
    print("\n=== Testing Prediction ===")
    
    if image_path and not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    if image_path:
        print(f"Testing on: {image_path}")
    else:
        print("No test image provided, skipping prediction test")
        return True
    
    try:
        import cv2
        from datetime import datetime
        
        # Load model
        model = RoadConditionModel()
        model.load_model(model_path)
        
        # Load and process image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Failed to load image")
            return False
        
        print(f"  Image shape: {frame.shape}")
        
        # Predict
        start_time = datetime.now()
        result = model.predict(frame)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Prediction successful!")
        print(f"  Class: {result['class_name']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Inference time: {elapsed*1000:.1f}ms")
        
        print(f"\n  All probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob*100:.1f}%")
        
        return True
    
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False


def compare_model_sizes():
    """Compare model file with expected size"""
    if not os.path.exists(MODEL_PATH):
        return
    
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    print("\n=== Model Size Analysis ===")
    print(f"Current model: {size_mb:.2f} MB")
    
    # Expected size range for MobileNetV2 + custom layers
    expected_min = 10  # MB
    expected_max = 50  # MB
    
    if expected_min <= size_mb <= expected_max:
        print(f"✓ Size is within expected range ({expected_min}-{expected_max} MB)")
    else:
        print(f"⚠ Size outside expected range ({expected_min}-{expected_max} MB)")
        if size_mb < expected_min:
            print("  Model may be incomplete or corrupted")
        else:
            print("  Model may contain extra data")


def main():
    parser = argparse.ArgumentParser(description="Verify trained model")
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--image', type=str, help='Test image path')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Road Condition Model Verification")
    print("="*60)
    
    # Verify model
    if not verify_model(args.model):
        return
    
    # Compare sizes
    compare_model_sizes()
    
    # Test prediction
    if args.image:
        test_model_prediction(args.model, args.image)
    
    print("\n" + "="*60)
    print("✓ Verification Complete")
    print("="*60)


if __name__ == "__main__":
    main()
