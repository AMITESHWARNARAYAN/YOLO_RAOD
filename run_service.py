"""
Road Monitoring Service - Background monitoring without GUI
"""

import time
import signal
import sys
from datetime import datetime
from camera_capture import CameraCapture
from road_condition_model import RoadConditionModel
from google_maps_integration import GoogleMapsIntegration
from config import CONFIDENCE_THRESHOLD, CAPTURE_INTERVAL


class RoadMonitoringService:
    """Background service for continuous road monitoring"""
    
    def __init__(self):
        self.running = False
        self.camera = CameraCapture()
        self.model = RoadConditionModel()
        self.maps = GoogleMapsIntegration()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n✓ Received shutdown signal")
        self.stop()
    
    def start(self):
        """Start monitoring service"""
        print("\n" + "="*60)
        print("Road Monitoring Service")
        print("="*60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print("="*60)
        
        # Initialize components
        print("\n=== Initializing ===")
        self.model.load_model()
        self.camera.start_camera()
        
        self.running = True
        detection_count = 0
        report_count = 0
        start_time = time.time()
        
        print("\n✓ Service running (Press Ctrl+C to stop)\n")
        
        try:
            while self.running:
                # Capture frame
                frame = self.camera.capture_frame()
                
                # Detect condition
                result = self.model.predict_with_threshold(frame, CONFIDENCE_THRESHOLD)
                detection_count += 1
                
                # Log detection
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Detected: {result['class_name']} "
                      f"({result['confidence']*100:.1f}%)")
                
                # Report if significant issue
                if result['confidence'] >= CONFIDENCE_THRESHOLD:
                    if result['class_name'] not in ['Good', 'Uncertain']:
                        location = self.maps.get_current_location()
                        self.maps.report_road_condition(
                            condition=result['class_name'],
                            confidence=result['confidence'],
                            latitude=location['latitude'],
                            longitude=location['longitude']
                        )
                        report_count += 1
                
                # Wait for next capture
                time.sleep(CAPTURE_INTERVAL)
        
        except Exception as e:
            print(f"\n✗ Service error: {e}")
        
        finally:
            # Cleanup
            self.camera.stop_camera()
            
            # Print summary
            uptime = time.time() - start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            
            print("\n" + "="*60)
            print("Service Summary")
            print("="*60)
            print(f"Uptime: {hours}h {minutes}m {seconds}s")
            print(f"Total detections: {detection_count}")
            print(f"Issues reported: {report_count}")
            print("="*60)
    
    def stop(self):
        """Stop monitoring service"""
        self.running = False
        print("\n✓ Stopping service...")


def main():
    service = RoadMonitoringService()
    service.start()


if __name__ == "__main__":
    main()
