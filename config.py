"""
Road Condition Detection System - Configuration
"""

import os

# Google Maps API Configuration
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY_HERE"  # Get from: https://console.cloud.google.com/

# Camera Configuration
CAMERA_INDEX = 0  # Default camera (usually 0 for built-in webcam)
CAMERA_RESOLUTION = (1280, 720)  # Width x Height
CAPTURE_INTERVAL = 5  # Seconds between captures in continuous mode

# Model Configuration
MODEL_PATH = "models/road_condition_model.pth"
INPUT_SIZE = (224, 224)  # Model input size
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to report condition

# Road Condition Classes
ROAD_CONDITIONS = {
    0: "Good",
    1: "Minor_Damage",
    2: "Pothole",
    3: "Crack",
    4: "Severe_Damage"
}

# GPS Configuration
USE_MOCK_GPS = True  # Set to False when using real GPS module
MOCK_GPS_LOCATION = {
    "latitude": 28.7041,  # Delhi coordinates (change to your location)
    "longitude": 77.1025
}

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging Configuration
LOG_FILE = os.path.join(LOGS_DIR, "road_detection.log")
REPORTS_FILE = os.path.join(LOGS_DIR, "road_reports.json")

# Display Configuration
SHOW_CONFIDENCE = True
SHOW_TIMESTAMP = True
DISPLAY_WINDOW_NAME = "Road Condition Detection"
