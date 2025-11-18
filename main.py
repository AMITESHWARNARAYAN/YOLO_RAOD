"""
Road Condition Detection System - Main Application
"""

import cv2
import argparse
from datetime import datetime
from camera_capture import CameraCapture
from road_condition_model import RoadConditionModel
from google_maps_integration import GoogleMapsIntegration
from config import (
    CONFIDENCE_THRESHOLD,
    CAPTURE_INTERVAL,
    SHOW_CONFIDENCE,
    SHOW_TIMESTAMP,
    DISPLAY_WINDOW_NAME
)


class RoadConditionSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("Road Condition Detection System")
        print("="*60)
        
        self.camera = CameraCapture()
        self.model = RoadConditionModel()
        self.maps = GoogleMapsIntegration()
        
        print("\n=== Initializing Components ===")
        self.model.load_model()
        print("✓ All components initialized")
    
    def process_frame(self, frame, report_to_maps=True):
        """Process single frame and detect road condition"""
        # Predict condition
        result = self.model.predict_with_threshold(frame, CONFIDENCE_THRESHOLD)
        
        # Get location
        location = self.maps.get_current_location()
        
        # Display results on frame
        self._draw_results(frame, result)
        
        # Report if significant issue detected
        if report_to_maps and result['confidence'] >= CONFIDENCE_THRESHOLD:
            if result['class_name'] not in ['Good', 'Uncertain']:
                self.maps.report_road_condition(
                    condition=result['class_name'],
                    confidence=result['confidence'],
                    latitude=location['latitude'],
                    longitude=location['longitude']
                )
        
        return frame, result
    
    def _draw_results(self, frame, result):
        """Draw detection results on frame"""
        # Prepare text
        condition = result['class_name']
        confidence = result['confidence']
        
        # Choose color based on condition
        color_map = {
            'Good': (0, 255, 0),           # Green
            'Minor_Damage': (0, 255, 255), # Yellow
            'Crack': (0, 165, 255),        # Orange
            'Pothole': (0, 0, 255),        # Red
            'Severe_Damage': (0, 0, 139),  # Dark Red
            'Uncertain': (128, 128, 128)   # Gray
        }
        color = color_map.get(condition, (255, 255, 255))
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), color, 2)
        
        # Draw text
        y_offset = 40
        cv2.putText(frame, f"Condition: {condition}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if SHOW_CONFIDENCE:
            y_offset += 30
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if SHOW_TIMESTAMP:
            y_offset += 30
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_continuous_monitoring(self):
        """Run continuous road monitoring with camera"""
        print("\n=== Starting Continuous Monitoring ===")
        print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Force report current condition")
        print()
        
        self.camera.start_camera()
        
        try:
            while True:
                frame = self.camera.capture_frame()
                processed_frame, result = self.process_frame(frame)
                
                # Display
                cv2.imshow(DISPLAY_WINDOW_NAME, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    print("\n✓ Stopping monitoring...")
                    break
                elif key == ord('s'):
                    self.camera.save_frame(frame)
                elif key == ord('r'):
                    location = self.maps.get_current_location()
                    self.maps.report_road_condition(
                        condition=result['class_name'],
                        confidence=result['confidence'],
                        latitude=location['latitude'],
                        longitude=location['longitude']
                    )
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            self.camera.stop_camera()
    
    def run_single_detection(self, image_path):
        """Run detection on single image"""
        print(f"\n=== Single Image Detection ===")
        print(f"Image: {image_path}")
        
        # Load and process image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"✗ Failed to load image: {image_path}")
            return
        
        processed_frame, result = self.process_frame(frame, report_to_maps=False)
        
        # Display result
        print(f"\nDetection Result:")
        print(f"  Condition: {result['class_name']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"\nAll probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob*100:.1f}%")
        
        # Show image
        cv2.imshow("Detection Result", processed_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Road Condition Detection System")
    parser.add_argument('--image', type=str, help='Path to single image for detection')
    parser.add_argument('--no-report', action='store_true', help='Disable Google Maps reporting')
    
    args = parser.parse_args()
    
    # Create system
    system = RoadConditionSystem()
    
    # Run mode
    if args.image:
        system.run_single_detection(args.image)
    else:
        system.run_continuous_monitoring()
    
    print("\n" + "="*60)
    print("✓ System shutdown complete")
    print("="*60)


if __name__ == "__main__":
    main()
