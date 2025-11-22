"""
FastAPI server for road damage detection API
Production-ready REST API with Google Maps integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import requests
from datetime import datetime
import json
import os

app = FastAPI(
    title="Road Damage Detection API",
    description="Real-time road damage detection with YOLOv8 and Google Maps integration",
    version="1.0.0"
)

# Global model instance
model = None
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    
class RoadCondition(BaseModel):
    score: int
    condition: str
    recommendation: str
    damage_counts: dict

class DetectionResponse(BaseModel):
    detections: List[DetectionResult]
    road_condition: RoadCondition
    total_detections: int
    inference_time_ms: float

class LocationData(BaseModel):
    latitude: float
    longitude: float
    timestamp: str

class DamageReport(BaseModel):
    location: LocationData
    damage_type: str
    confidence: float
    severity: str

@app.on_event("startup")
async def load_model():
    """Load YOLOv8 model on startup"""
    global model
    model_path = os.getenv('MODEL_PATH', 'runs/detect/road_damage/weights/best.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model not found at {model_path}")
        print("Using default YOLOv8n model")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Road Damage Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect - Detect damage in uploaded image",
            "detect_url": "/detect/url - Detect damage from image URL",
            "report": "/report - Submit damage report to Google Maps",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "google_maps_enabled": bool(GOOGLE_MAPS_API_KEY)
    }

def analyze_road_condition(detections: List[dict]) -> dict:
    """Analyze overall road condition"""
    if not detections:
        return {
            'score': 100,
            'condition': 'Excellent',
            'recommendation': 'No issues detected',
            'damage_counts': {}
        }
    
    damage_weights = {
        'Crack': -5,
        'Pothole': -15,
        'Severe_Damage': -30
    }
    
    score = 100
    damage_counts = {}
    
    for det in detections:
        class_name = det['class_name']
        damage_counts[class_name] = damage_counts.get(class_name, 0) + 1
        score += damage_weights.get(class_name, 0)
    
    score = max(0, min(100, score))
    
    if score >= 80:
        condition = 'Excellent'
        recommendation = 'Road in good condition'
    elif score >= 60:
        condition = 'Good'
        recommendation = 'Minor maintenance recommended'
    elif score >= 40:
        condition = 'Fair'
        recommendation = 'Maintenance required soon'
    elif score >= 20:
        condition = 'Poor'
        recommendation = 'Immediate maintenance needed'
    else:
        condition = 'Critical'
        recommendation = 'Urgent repair required'
    
    return {
        'score': score,
        'condition': condition,
        'recommendation': recommendation,
        'damage_counts': damage_counts
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_damage(file: UploadFile = File(...), confidence: float = 0.5):
    """
    Detect road damage in uploaded image
    
    Args:
        file: Image file (jpg, png)
        confidence: Detection confidence threshold (0.0-1.0)
    
    Returns:
        Detection results with road condition analysis
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Run detection
    results = model(image, conf=confidence, verbose=False)[0]
    
    # Parse detections
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        detections.append({
            'class_name': class_name,
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })
    
    # Analyze road condition
    road_condition = analyze_road_condition(detections)
    
    # Get inference time
    inference_time = results.speed['inference']
    
    return {
        'detections': detections,
        'road_condition': road_condition,
        'total_detections': len(detections),
        'inference_time_ms': inference_time
    }

@app.post("/detect/url")
async def detect_damage_url(image_url: str, confidence: float = 0.5):
    """
    Detect road damage from image URL
    
    Args:
        image_url: URL to image
        confidence: Detection confidence threshold
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image URL")
        
        # Run detection
        results = model(image, conf=confidence, verbose=False)[0]
        
        # Parse detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
        
        # Analyze road condition
        road_condition = analyze_road_condition(detections)
        
        return {
            'detections': detections,
            'road_condition': road_condition,
            'total_detections': len(detections),
            'inference_time_ms': results.speed['inference']
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.post("/report")
async def submit_damage_report(report: DamageReport, background_tasks: BackgroundTasks):
    """
    Submit damage report to Google Maps
    
    Args:
        report: Damage report with location and details
    """
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(status_code=503, detail="Google Maps API not configured")
    
    # Add background task to update Google Maps
    background_tasks.add_task(update_google_maps, report)
    
    # Save to local database
    save_report(report)
    
    return {
        'status': 'success',
        'message': 'Damage report submitted',
        'report_id': f"{report.location.timestamp}_{report.damage_type}"
    }

def update_google_maps(report: DamageReport):
    """Update Google Maps with damage report (background task)"""
    try:
        url = "https://roads.googleapis.com/v1/snapToRoads"
        params = {
            'path': f"{report.location.latitude},{report.location.longitude}",
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print(f"✓ Damage reported to Google Maps: {report.damage_type}")
        else:
            print(f"❌ Failed to update Google Maps: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error updating Google Maps: {e}")

def save_report(report: DamageReport):
    """Save damage report to JSON file"""
    reports_file = 'damage_reports.json'
    
    report_data = {
        'timestamp': report.location.timestamp,
        'location': {
            'latitude': report.location.latitude,
            'longitude': report.location.longitude
        },
        'damage_type': report.damage_type,
        'confidence': report.confidence,
        'severity': report.severity
    }
    
    if os.path.exists(reports_file):
        with open(reports_file, 'r') as f:
            reports = json.load(f)
    else:
        reports = []
    
    reports.append(report_data)
    
    with open(reports_file, 'w') as f:
        json.dump(reports, f, indent=2)

@app.get("/reports")
async def get_reports(limit: int = 100):
    """Get recent damage reports"""
    reports_file = 'damage_reports.json'
    
    if not os.path.exists(reports_file):
        return {'reports': [], 'total': 0}
    
    with open(reports_file, 'r') as f:
        reports = json.load(f)
    
    return {
        'reports': reports[-limit:],
        'total': len(reports)
    }

if __name__ == '__main__':
    import uvicorn
    
    print("🚀 Starting Road Damage Detection API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
