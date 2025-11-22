"""
Train YOLOv8 for road damage detection
Much faster and better for production than classification
"""

from ultralytics import YOLO
import torch
import os

def train_yolov8(
    model_size='n',  # n (nano), s (small), m (medium), l (large), x (xlarge)
    epochs=100,
    img_size=640,
    batch_size=16,
    device=0
):
    """
    Train YOLOv8 model for road damage detection
    
    Args:
        model_size: Model size (n=fastest, x=most accurate)
        epochs: Training epochs
        img_size: Image size (640 recommended)
        batch_size: Batch size
        device: GPU device (0) or 'cpu'
    """
    
    print("="*70)
    print("YOLOv8 Road Damage Detection Training")
    print("="*70)
    
    # Check if YOLO dataset exists
    if not os.path.exists('yolo_data/data.yaml'):
        print("\n❌ Error: YOLO dataset not found!")
        print("Run 'python convert_to_yolo_format.py' first")
        return
    
    # Load pretrained YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n📦 Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Check device
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == 0 else 'CPU'
    print(f"🔧 Device: {device_name}")
    
    # Training configuration
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {img_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Model: YOLOv8{model_size}")
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Train the model
    results = model.train(
        data='yolo_data/data.yaml',
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,          # Initial learning rate
        lrf=0.01,           # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Data augmentation
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.7,          # HSV-Saturation
        hsv_v=0.4,          # HSV-Value
        degrees=10,         # Rotation
        translate=0.1,      # Translation
        scale=0.5,          # Scale
        shear=0.0,          # Shear
        perspective=0.0,    # Perspective
        flipud=0.0,         # Flip up-down
        fliplr=0.5,         # Flip left-right
        mosaic=1.0,         # Mosaic augmentation
        mixup=0.1,          # Mixup augmentation
        
        # Training settings
        patience=50,        # Early stopping patience
        save=True,          # Save checkpoints
        save_period=10,     # Save every N epochs
        cache=False,        # Cache images (set True if enough RAM)
        workers=8,          # Dataloader workers
        project='runs/detect',
        name='road_damage',
        exist_ok=True,
        
        # Validation
        val=True,
        plots=True,         # Save training plots
        verbose=True
    )
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED!")
    print("="*70)
    
    # Print results
    print(f"\n📊 Final Results:")
    print(f"  Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    # Export model for production
    print("\n📦 Exporting model for production...")
    
    # Load best model
    best_model = YOLO('runs/detect/road_damage/weights/best.pt')
    
    # Export to ONNX (for production deployment)
    print("  Exporting to ONNX format...")
    best_model.export(format='onnx', dynamic=True, simplify=True)
    
    # Export to TorchScript (for Python deployment)
    print("  Exporting to TorchScript format...")
    best_model.export(format='torchscript')
    
    print("\n✓ Model exported successfully!")
    print(f"  Best weights: runs/detect/road_damage/weights/best.pt")
    print(f"  ONNX model: runs/detect/road_damage/weights/best.onnx")
    print(f"  TorchScript: runs/detect/road_damage/weights/best.torchscript")
    
    print("\n🚀 Next steps:")
    print("  1. Test model: python test_yolov8.py")
    print("  2. Deploy API: python api_server.py")
    print("  3. Run detection: python detect_realtime.py")
    
    return best_model

def validate_model(model_path='runs/detect/road_damage/weights/best.pt'):
    """Validate trained model"""
    model = YOLO(model_path)
    results = model.val(data='yolo_data/data.yaml')
    
    print("\n📊 Validation Results:")
    print(f"  mAP@0.5: {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for road damage detection')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'], 
                        help='Model size (n=fastest, x=most accurate)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='0', help='Device (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    # Convert device to int if it's a number
    device = int(args.device) if args.device.isdigit() else args.device
    
    # Train model
    train_yolov8(
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=device
    )
