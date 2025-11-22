"""
Real-time road damage detection from camera/video
With GPS integration and Google Maps API updates
"""

import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
import requests
import time
from collections import defaultdict
import os

class RoadDamageDetector:
    def __init__(self, model_path='runs/detect/road_damage/weights/best.pt', 
                 confidence=0.5, google_maps_api_key=None):
        """
        Initialize real-time road damage detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence: Detection confidence threshold
            google_maps_api_key: Google Maps API key for updates
        """
        print("🚀 Initializing Road Damage Detector...")
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.google_maps_api_key = google_maps_api_key
        
        # Detection tracking
        self.detections_history = []
        self.location_damages = defaultdict(list)
        
        # GPS placeholder (replace with actual GPS module)
        self.current_location = {'lat': 0.0, 'lon': 0.0}
        
        # Colors for each class (BGR format)
        self.colors = {
            0: (0, 255, 255),    # Crack - Yellow
            1: (0, 165, 255),    # Pothole - Orange
            2: (0, 0, 255)       # Severe_Damage - Red
        }
        
        print("✓ Model loaded successfully!")
        print(f"✓ Confidence threshold: {confidence}")
        print(f"✓ Classes: {self.model.names}")
    
    def get_gps_location(self):
        """
        Get current GPS location
        Replace with actual GPS module (e.g., gpsd, serial GPS)
        """
        # Placeholder - integrate with actual GPS
        # For testing, you can use a mock location or GPS library
        try:
            # Example: Use gpsd library
            # from gps import gps, WATCH_ENABLE
            # session = gps(mode=WATCH_ENABLE)
            # report = session.next()
            # return {'lat': report.lat, 'lon': report.lon}
            
            # For now, return mock location (update in production)
            return {'lat': self.current_location['lat'], 
                    'lon': self.current_location['lon']}
        except:
            return {'lat': 0.0, 'lon': 0.0}
    
    def update_google_maps(self, damage_data):
        """
        Send damage data to Google Maps API
        Creates road condition reports
        """
        if not self.google_maps_api_key:
            print("⚠️ Google Maps API key not configured")
            return False
        
        try:
            # Google Maps Roads API endpoint
            url = "https://roads.googleapis.com/v1/snapToRoads"
            
            params = {
                'path': f"{damage_data['lat']},{damage_data['lon']}",
                'key': self.google_maps_api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                road_info = response.json()
                
                # Log to local database or cloud
                self.log_damage_report(damage_data, road_info)
                
                print(f"✓ Damage reported to Google Maps: {damage_data['class_name']}")
                return True
            else:
                print(f"❌ Failed to update Google Maps: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating Google Maps: {e}")
            return False
    
    def log_damage_report(self, damage_data, road_info=None):
        """Log damage report to file/database"""
        report = {
            'timestamp': damage_data['timestamp'],
            'location': damage_data['location'],
            'damage_type': damage_data['class_name'],
            'confidence': damage_data['confidence'],
            'road_info': road_info
        }
        
        # Save to JSON file (replace with database in production)
        reports_file = 'damage_reports.json'
        
        if os.path.exists(reports_file):
            with open(reports_file, 'r') as f:
                reports = json.load(f)
        else:
            reports = []
        
        reports.append(report)
        
        with open(reports_file, 'w') as f:
            json.dump(reports, f, indent=2)
    
    def analyze_road_condition(self, detections):
        """
        Analyze overall road condition based on detections
        Returns severity score and recommendation
        """
        if not detections:
            return {'score': 100, 'condition': 'Excellent', 'recommendation': 'No issues detected'}
        
        # Calculate severity score (0-100, higher is better)
        damage_weights = {
            'Crack': -5,
            'Pothole': -15,
            'Severe_Damage': -30
        }
        
        score = 100
        damage_counts = defaultdict(int)
        
        for det in detections:
            class_name = det['class_name']
            damage_counts[class_name] += 1
            score += damage_weights.get(class_name, 0)
        
        score = max(0, min(100, score))  # Clamp to 0-100
        
        # Determine condition
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
            recommendation = 'Urgent repair required - drive with caution'
        
        return {
            'score': score,
            'condition': condition,
            'recommendation': recommendation,
            'damage_counts': dict(damage_counts)
        }
    
    def process_frame(self, frame):
        """
        Process single frame for damage detection
        Returns annotated frame and detections
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        # Get current location
        location = self.get_gps_location()
        
        # Parse detections
        detections = []
        annotated_frame = frame.copy()
        
        for box in results.boxes:
            # Extract box data
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Store detection
            detection_data = {
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'location': location,
                'timestamp': datetime.now().isoformat()
            }
            detections.append(detection_data)
            
            # Draw bounding box
            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name} {confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Analyze road condition
        road_analysis = self.analyze_road_condition(detections)
        
        # Draw road condition overlay
        self.draw_overlay(annotated_frame, road_analysis, len(detections))
        
        return annotated_frame, detections, road_analysis
    
    def draw_overlay(self, frame, road_analysis, detection_count):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Road condition
        color_map = {
            'Excellent': (0, 255, 0),
            'Good': (0, 255, 255),
            'Fair': (0, 165, 255),
            'Poor': (0, 100, 255),
            'Critical': (0, 0, 255)
        }
        condition_color = color_map.get(road_analysis['condition'], (255, 255, 255))
        
        cv2.putText(frame, f"Road Condition: {road_analysis['condition']}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, condition_color, 2)
        cv2.putText(frame, f"Score: {road_analysis['score']}/100", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(self.model.speed['inference'])}ms", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Damage counts
        if road_analysis['damage_counts']:
            y_offset = height - 100
            cv2.rectangle(frame, (10, y_offset - 30), (250, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, "Damage Counts:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            for damage_type, count in road_analysis['damage_counts'].items():
                cv2.putText(frame, f"  {damage_type}: {count}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
    
    def run_video(self, video_source=0, save_output=False, output_path='output.mp4'):
        """
        Run detection on video stream
        
        Args:
            video_source: 0 for webcam, or path to video file
            save_output: Save annotated video
            output_path: Path to save output video
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n🎥 Video: {width}x{height} @ {fps}fps")
        print("Press 'q' to quit, 's' to save report\n")
        
        # Video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections, road_analysis = self.process_frame(frame)
                
                # Update Google Maps for significant detections
                if detections and frame_count % 30 == 0:  # Every 30 frames
                    for det in detections:
                        if det['confidence'] > 0.7:  # High confidence only
                            self.update_google_maps(det)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                cv2.imshow('Road Damage Detection', annotated_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'snapshot_{frame_count}.jpg', annotated_frame)
                    print(f"✓ Snapshot saved: snapshot_{frame_count}.jpg")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Cleanup
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n📊 Session Statistics:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Total detections: {len(self.detections_history)}")
            
            cap.release()
            if writer:
                writer.release()
                print(f"✓ Output saved: {output_path}")
            cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time road damage detection')
    parser.add_argument('--model', default='runs/detect/road_damage/weights/best.pt',
                        help='Path to YOLOv8 model')
    parser.add_argument('--source', default='0', help='Video source (0 for webcam, or video path)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output', default='output.mp4', help='Output video path')
    parser.add_argument('--api-key', default=None, help='Google Maps API key')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Initialize detector
    detector = RoadDamageDetector(
        model_path=args.model,
        confidence=args.conf,
        google_maps_api_key=args.api_key
    )
    
    # Run detection
    detector.run_video(
        video_source=source,
        save_output=args.save,
        output_path=args.output
    )
