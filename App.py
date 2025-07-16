#!/usr/bin/env python3
"""
Low Power Motion Detection Photo Capture Script for Raspberry Pi
Optimized for minimal resource consumption with USB camera support
Takes photos only when motion is detected, saves to script directory
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
import sys
import signal

class LowPowerMotionDetector:
    def __init__(self, camera_id=0, sensitivity=1000, photo_interval=2.0, 
                 motion_timeout=5.0, resolution=(320, 240), fps=5):
        """
        Initialize low-power motion detector
        
        Args:
            camera_id: USB camera device ID (usually 0 for first USB camera)
            sensitivity: Motion detection sensitivity (lower = more sensitive)
            photo_interval: Minimum seconds between photos
            motion_timeout: Seconds to wait after last motion before stopping
            resolution: Camera resolution tuple (width, height) - lower = less resources
            fps: Camera FPS - lower = less CPU usage
        """
        self.camera_id = camera_id
        self.sensitivity = sensitivity
        self.photo_interval = photo_interval
        self.motion_timeout = motion_timeout
        self.frame_width, self.frame_height = resolution
        self.fps = fps
        
        # Use script directory for photos
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.photo_dir = os.path.join(self.script_dir, "motion_photos")
        os.makedirs(self.photo_dir, exist_ok=True)
        
        # Motion detection - simplified for low resource usage
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,  # Reduced history for less memory
            varThreshold=30,  # Lower threshold for better detection
            detectShadows=False  # Disable shadow detection to save CPU
        )
        
        # State variables
        self.cap = None
        self.running = False
        self.last_motion_time = 0
        self.last_photo_time = 0
        self.motion_detected = False
        self.photo_count = 0
        self.motion_sequence = 0
        
        # Performance counters
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"Low Power Motion Detector initialized")
        print(f"Camera ID: {camera_id}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {fps}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Photo directory: {self.photo_dir}")
    
    def find_usb_camera(self):
        """Find available USB camera - simplified"""
        for camera_id in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"Found USB camera at index {camera_id}")
                    return camera_id
        return self.camera_id
    
    def initialize_camera(self):
        """Initialize USB camera with minimal resource settings"""
        # Find camera if needed
        if self.camera_id == 0:
            self.camera_id = self.find_usb_camera()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open USB camera at index {self.camera_id}")
            return False
        
        # Set minimal resource properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        
        # Use MJPEG for USB cameras (more efficient)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        return True
    
    def detect_motion(self, frame):
        """Lightweight motion detection"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Simple noise reduction - minimal processing
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Count white pixels (motion pixels)
        motion_pixels = cv2.countNonZero(fg_mask)
        
        return motion_pixels > self.sensitivity
    
    def save_photo(self, frame):
        """Save photo with timestamp"""
        current_time = time.time()
        
        # Check if enough time has passed since last photo
        if current_time - self.last_photo_time < self.photo_interval:
            return False
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{self.motion_sequence:03d}_{timestamp}_{self.photo_count:03d}.jpg"
        filepath = os.path.join(self.photo_dir, filename)
        
        # Save photo with high compression to save space
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        self.last_photo_time = current_time
        self.photo_count += 1
        
        print(f"Photo saved: {filename}")
        return True
    
    def run(self):
        """Main detection loop"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        print("Starting motion detection...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                current_time = time.time()
                self.frame_count += 1
                
                # Detect motion
                motion_detected = self.detect_motion(frame)
                
                if motion_detected:
                    if not self.motion_detected:
                        # New motion sequence started
                        self.motion_sequence += 1
                        self.photo_count = 0
                        print(f"Motion detected! Starting sequence {self.motion_sequence}")
                    
                    self.motion_detected = True
                    self.last_motion_time = current_time
                    
                    # Save photo
                    self.save_photo(frame)
                    
                else:
                    # Check if motion has stopped
                    if (self.motion_detected and 
                        current_time - self.last_motion_time > self.motion_timeout):
                        print(f"Motion stopped. Sequence {self.motion_sequence} complete ({self.photo_count} photos)")
                        self.motion_detected = False
                
                # Print status periodically
                if self.frame_count % (self.fps * 10) == 0:  # Every 10 seconds
                    elapsed = current_time - self.start_time
                    fps_actual = self.frame_count / elapsed
                    status = "MOTION" if self.motion_detected else "MONITORING"
                    print(f"Status: {status} | FPS: {fps_actual:.1f} | Photos: {self.photo_count}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping motion detection...")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\nSummary:")
        print(f"Total runtime: {elapsed:.1f} seconds")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Motion sequences: {self.motion_sequence}")
        print(f"Photos saved to: {self.photo_dir}")
        print("Cleanup complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Low Power Motion Detection Photo Capture")
    parser.add_argument("--camera", type=int, default=0, help="USB Camera ID (default: 0)")
    parser.add_argument("--sensitivity", type=int, default=1000, 
                       help="Motion sensitivity - lower = more sensitive (default: 1000)")
    parser.add_argument("--photo-interval", type=float, default=2.0, 
                       help="Minimum seconds between photos (default: 2.0)")
    parser.add_argument("--motion-timeout", type=float, default=5.0, 
                       help="Seconds to wait after motion stops (default: 5.0)")
    parser.add_argument("--resolution", type=str, default="320x240", 
                       help="Camera resolution WxH (default: 320x240)")
    parser.add_argument("--fps", type=int, default=5, 
                       help="Camera FPS - lower = less CPU usage (default: 5)")
    parser.add_argument("--test-camera", action="store_true", 
                       help="Test camera and exit")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 320x240)")
        return 1
    
    # Test camera mode
    if args.test_camera:
        print("Testing camera...")
        detector = LowPowerMotionDetector(camera_id=args.camera, resolution=resolution)
        if detector.initialize_camera():
            print("Camera test successful!")
            detector.cleanup()
            return 0
        else:
            print("Camera test failed!")
            return 1
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run detector
    detector = LowPowerMotionDetector(
        camera_id=args.camera,
        sensitivity=args.sensitivity,
        photo_interval=args.photo_interval,
        motion_timeout=args.motion_timeout,
        resolution=resolution,
        fps=args.fps
    )
    
    success = detector.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
