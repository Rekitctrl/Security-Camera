#!/usr/bin/env python3
"""
Motion Detection and Recording Script for Raspberry Pi
Optimized for 512MB RAM systems
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
import sys

class MotionDetector:
    def __init__(self, camera_id=0, sensitivity=1000, min_record_time=5, 
                 max_record_time=60, output_dir="recordings"):
        """
        Initialize motion detector
        
        Args:
            camera_id: USB camera device ID (usually 0 for first USB camera)
            sensitivity: Motion detection sensitivity (lower = more sensitive)
            min_record_time: Minimum recording time in seconds
            max_record_time: Maximum recording time in seconds
            output_dir: Directory to save recordings
        """
        self.camera_id = self.find_usb_camera(camera_id)
        self.sensitivity = sensitivity
        self.min_record_time = min_record_time
        self.max_record_time = max_record_time
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Motion detection variables
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
        # Recording variables
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.last_motion_time = None
        
        # Camera setup
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 15  # Lower FPS to save memory
        
        print(f"USB Motion detector initialized")
        print(f"USB Camera ID: {self.camera_id}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Recording time: {min_record_time}-{max_record_time} seconds")
        print(f"Output directory: {output_dir}")
    
    def find_usb_camera(self, preferred_id=0):
        """Find available USB camera"""
        print("Scanning for USB cameras...")
        
        # Check multiple camera indices
        for camera_id in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test if it's actually working
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"Found USB camera at index {camera_id}")
                    if camera_id == preferred_id:
                        return camera_id
                    elif preferred_id == 0:  # If no preference, use first found
                        return camera_id
        
        print(f"Warning: No USB camera found, using default index {preferred_id}")
        return preferred_id
    
    def initialize_camera(self):
        """Initialize USB camera with optimal settings for low memory"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open USB camera at index {self.camera_id}")
            print("Make sure USB camera is connected and not in use by another program")
            return False
        
        # Set camera properties for memory optimization
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        # USB camera specific settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for USB cameras
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"USB Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Update our internal values with actual camera settings
        self.frame_width = actual_width
        self.frame_height = actual_height
        
        return True
    
    def detect_motion(self, frame):
        """Detect motion in frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion area
        motion_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small movements
                motion_area += area
        
        return motion_area > self.sensitivity
    
    def start_recording(self):
        """Start video recording"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use efficient codec for Pi
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        
        self.is_recording = True
        self.recording_start_time = time.time()
        print(f"Started recording: {filename}")
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        print(f"Stopped recording after {duration:.1f} seconds")
    
    def should_stop_recording(self):
        """Check if recording should stop"""
        if not self.is_recording:
            return False
        
        current_time = time.time()
        recording_duration = current_time - self.recording_start_time
        time_since_motion = current_time - self.last_motion_time if self.last_motion_time else 0
        
        # Stop if max time reached or no motion for a while (after min time)
        if recording_duration >= self.max_record_time:
            return True
        
        if (recording_duration >= self.min_record_time and 
            time_since_motion > 3):  # 3 seconds after last motion
            return True
        
        return False
    
    def process_frame(self, frame):
        """Process a single frame"""
        motion_detected = self.detect_motion(frame)
        
        if motion_detected:
            self.last_motion_time = time.time()
            if not self.is_recording:
                self.start_recording()
        
        # Record frame if recording
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        # Check if should stop recording
        if self.should_stop_recording():
            self.stop_recording()
        
        return motion_detected
    
    def run(self):
        """Main detection loop"""
        if not self.initialize_camera():
            return
        
        print("Starting motion detection... Press Ctrl+C to stop")
        
        try:
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                motion_detected = self.process_frame(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Status every 2 seconds at 15fps
                    status = "RECORDING" if self.is_recording else "MONITORING"
                    print(f"Status: {status} | Motion: {'YES' if motion_detected else 'NO'}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping motion detection...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="USB Motion Detection for Raspberry Pi")
    parser.add_argument("--camera", type=int, default=0, help="USB Camera ID (default: 0)")
    parser.add_argument("--sensitivity", type=int, default=1000, 
                       help="Motion sensitivity (lower = more sensitive)")
    parser.add_argument("--min-time", type=int, default=5, 
                       help="Minimum recording time in seconds")
    parser.add_argument("--max-time", type=int, default=60, 
                       help="Maximum recording time in seconds")
    parser.add_argument("--output", type=str, default="recordings", 
                       help="Output directory for recordings")
    
    args = parser.parse_args()
    
    # Check for USB camera
    print("Initializing USB camera motion detection system...")
    print("Make sure your USB camera is connected before starting")
    
    detector = MotionDetector(
        camera_id=args.camera,
        sensitivity=args.sensitivity,
        min_record_time=args.min_time,
        max_record_time=args.max_time,
        output_dir=args.output
    )
    
    detector.run()

if __name__ == "__main__":
    main()
