"""
Camera Capture Module for Road Condition Detection
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
from config import CAMERA_INDEX, CAMERA_RESOLUTION, DATA_DIR


class CameraCapture:
    """Handle camera operations for road condition detection"""
    
    def __init__(self, camera_index=CAMERA_INDEX, resolution=CAMERA_RESOLUTION):
        self.camera_index = camera_index
        self.resolution = resolution
        self.camera = None
        self.is_running = False
    
    def start_camera(self):
        """Initialize and start camera"""
        print(f"\n=== Starting Camera ===")
        print(f"Camera index: {self.camera_index}")
        print(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Verify actual resolution
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera started successfully!")
        print(f"  Actual resolution: {actual_width}x{actual_height}")
        
        self.is_running = True
        return True
    
    def capture_frame(self):
        """Capture single frame from camera"""
        if not self.is_running or self.camera is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")
        
        ret, frame = self.camera.read()
        
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        return frame
    
    def save_frame(self, frame, filename=None):
        """Save frame to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"road_image_{timestamp}.jpg"
        
        filepath = os.path.join(DATA_DIR, filename)
        cv2.imwrite(filepath, frame)
        
        print(f"✓ Frame saved: {filepath}")
        return filepath
    
    def capture_continuous(self, interval=5, callback=None):
        """Capture frames continuously at specified interval"""
        print(f"\n=== Continuous Capture Mode ===")
        print(f"Capture interval: {interval} seconds")
        print("Press 'q' to quit, 's' to save frame")
        
        last_capture_time = 0
        
        while self.is_running:
            frame = self.capture_frame()
            current_time = time.time()
            
            # Add timestamp overlay
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Camera Feed", frame)
            
            # Capture at interval
            if current_time - last_capture_time >= interval:
                if callback:
                    callback(frame.copy())
                last_capture_time = current_time
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Stopping capture...")
                break
            elif key == ord('s'):
                self.save_frame(frame)
        
        self.stop_camera()
    
    def stop_camera(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            cv2.destroyAllWindows()
            self.is_running = False
            print("✓ Camera stopped")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_camera()


if __name__ == "__main__":
    # Test camera
    camera = CameraCapture()
    camera.start_camera()
    
    print("\nCapturing test frame...")
    frame = camera.capture_frame()
    print(f"Frame shape: {frame.shape}")
    
    camera.stop_camera()
    print("\n✓ Camera test successful!")
