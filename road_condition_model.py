"""
Road Condition Detection Model - PyTorch with YOLO-based Transfer Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models as torchvision_models
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
from PIL import Image
import os
from config import MODEL_PATH, INPUT_SIZE, ROAD_CONDITIONS


class YOLORoadConditionModel(nn.Module):
    """YOLO-style model with MobileNetV2 backbone"""
    
    def __init__(self, num_classes=5):
        super(YOLORoadConditionModel, self).__init__()
        
        # Load pre-trained MobileNetV2 (YOLO-style feature extractor)
        self.backbone = torchvision_models.mobilenet_v2(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        
        # Remove default classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layers(x)
        return x


class RoadConditionModel:
    """Model wrapper for training and inference"""
    
    def __init__(self, num_classes=5, input_size=INPUT_SIZE):
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = list(ROAD_CONDITIONS.values())
    
    def build_model_with_yolo_features(self):
        """Build model with YOLO-style transfer learning"""
        print("\n=== Building Model with YOLO Features ===")
        
        self.model = YOLORoadConditionModel(self.num_classes).to(self.device)
        
        # Freeze early layers (transfer learning)
        for param in list(self.model.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model built successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.device}")
        
        return self.model
    
    def load_model(self, model_path=MODEL_PATH):
        """Load trained model from file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n=== Loading Model ===")
        print(f"Loading from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model is None:
            self.build_model_with_yolo_features()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Classes: {checkpoint.get('num_classes', self.num_classes)}")
        print(f"  Input size: {checkpoint.get('input_size', self.input_size)}")
        
        return self.model
    
    def save_model(self, model_path=MODEL_PATH):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'class_names': self.class_names
        }, model_path)
        
        print(f"✓ Model saved: {model_path}")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image):
        """Predict road condition"""
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        with torch.no_grad():
            img_tensor = self.preprocess_image(image)
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return {
            'class_id': predicted_class.item(),
            'class_name': self.class_names[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(self.num_classes)
            }
        }
    
    def predict_with_threshold(self, image, threshold=0.7):
        """Predict with confidence threshold"""
        result = self.predict(image)
        
        if result['confidence'] < threshold:
            result['class_name'] = "Uncertain"
            result['warning'] = f"Low confidence ({result['confidence']*100:.1f}%)"
        
        return result


if __name__ == "__main__":
    # Test model creation
    model = RoadConditionModel()
    model.build_model_with_yolo_features()
    print("\n✓ Model test successful!")
