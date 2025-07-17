#!/usr/bin/env python3
"""
All-Lighting Motion Detection Photo Capture Script for Raspberry Pi
Optimized for accurate detection in all lighting conditions
Includes adaptive thresholding, light change compensation, and robust filtering
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
from collections import deque
import math

class AdaptiveMotionDetector:
    def __init__(self, camera_id=0, sensitivity=1000, photo_interval=2.0,
                 motion_timeout=5.0, resolution=(320, 240), fps=5):
        """
        Initialize adaptive motion detector for all lighting conditions

        Args:
            camera_id: USB camera device ID (usually 0 for first USB camera)
            sensitivity: Motion detection sensitivity (lower = more sensitive)
            photo_interval: Minimum seconds between photos
            motion_timeout: Seconds to wait after last motion before stopping
            resolution: Camera resolution tuple (width, height)
            fps: Camera FPS
        """
        self.camera_id = camera_id
        self.sensitivity = sensitivity
        self.photo_interval = photo_interval
        self.motion_timeout = motion_timeout
        self.frame_width, self.frame_height = resolution
        self.fps = fps
        
        # Directory setup
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_photo_dir = os.path.join(self.script_dir, "motion_photos")
        os.makedirs(self.base_photo_dir, exist_ok=True)
        self.current_session_dir = None
        
        # Adaptive background subtraction for varying lighting
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,  # Longer history for lighting adaptation
            varThreshold=16,  # Lower threshold for better sensitivity
            detectShadows=True,
            shadowValue=0,
            shadowThreshold=0.5
        )
        
        # Multi-method motion detection
        self.frame_buffer = deque(maxlen=5)  # Store frames for comparison
        self.gray_buffer = deque(maxlen=3)   # Grayscale frames for analysis
        self.background_ready = False
        self.frames_processed = 0
        
        # Lighting adaptation parameters
        self.light_history = deque(maxlen=30)  # Track lighting changes
        self.adaptive_threshold = 25
        self.min_threshold = 10
        self.max_threshold = 60
        self.light_change_sensitivity = 0.15  # Threshold for significant light change
        
        # Enhanced noise reduction for different lighting
        self.gaussian_kernels = {
            'low_light': (7, 7),
            'normal': (5, 5),
            'bright': (3, 3)
        }
        self.morphology_kernels = {
            'low_light': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'normal': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'bright': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        }
        
        # Adaptive contour filtering
        self.lighting_mode = 'normal'  # 'low_light', 'normal', 'bright'
        self.min_contour_areas = {
            'low_light': 200,   # Larger minimum for noise reduction
            'normal': 100,
            'bright': 80
        }
        
        # Motion validation with lighting consideration
        self.motion_confirmation_frames = {
            'low_light': 3,  # More frames needed in low light
            'normal': 2,
            'bright': 2
        }
        self.motion_history = deque(maxlen=7)
        self.motion_tracks = {}
        self.track_id_counter = 0
        
        # Lighting analysis
        self.exposure_compensation = True
        self.auto_exposure_target = 128  # Target brightness
        self.histogram_analysis = True
        
        # State variables
        self.cap = None
        self.running = False
        self.last_motion_time = 0
        self.last_photo_time = 0
        self.motion_detected = False
        self.photo_count = 0
        self.motion_sequence = 0
        self.motion_boxes = []
        self.confirmed_motion = False
        self.current_brightness = 0
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.lighting_changes = 0
        
        print(f"Adaptive Motion Detector initialized for all lighting conditions")
        print(f"Camera ID: {camera_id}, Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {fps}, Sensitivity: {sensitivity}")

    def create_session_directory(self):
        """Create a new directory for the current motion detection session"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir_name = f"detection_{timestamp}"
        self.current_session_dir = os.path.join(self.base_photo_dir, session_dir_name)
        
        os.makedirs(self.current_session_dir, exist_ok=True)
        print(f"Created session directory: {session_dir_name}")
        return self.current_session_dir

    def find_usb_camera(self):
        """Find available USB camera"""
        for camera_id in range(5):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"Found USB camera at index {camera_id}")
                    return camera_id
        return self.camera_id

    def initialize_camera(self):
        """Initialize USB camera with optimal settings for all lighting"""
        if self.camera_id == 0:
            self.camera_id = self.find_usb_camera()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open USB camera at index {self.camera_id}")
            return False
        
        # Set camera properties for better lighting adaptation
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable auto-exposure and auto-white balance for lighting adaptation
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
        
        # Set optimal encoding
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Try to set gain control for low light
        try:
            self.cap.set(cv2.CAP_PROP_GAIN, 0)  # Auto gain
        except:
            pass
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        return True

    def analyze_lighting(self, frame):
        """Analyze current lighting conditions and adapt parameters"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness metrics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Analyze histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Determine lighting mode
        if mean_brightness < 70:
            new_mode = 'low_light'
        elif mean_brightness > 180:
            new_mode = 'bright'
        else:
            new_mode = 'normal'
        
        # Update lighting mode if changed
        if new_mode != self.lighting_mode:
            self.lighting_mode = new_mode
            self.lighting_changes += 1
            print(f"Lighting mode changed to: {new_mode} (brightness: {mean_brightness:.1f})")
        
        # Store lighting history
        self.light_history.append(mean_brightness)
        self.current_brightness = mean_brightness
        
        # Adapt threshold based on lighting conditions and stability
        if len(self.light_history) >= 5:
            light_stability = np.std(list(self.light_history)[-5:])
            if light_stability > 20:  # Unstable lighting
                self.adaptive_threshold = min(self.max_threshold, self.adaptive_threshold + 5)
            else:  # Stable lighting
                self.adaptive_threshold = max(self.min_threshold, self.adaptive_threshold - 1)
        
        # Update background subtractor parameters
        self.background_subtractor.setVarThreshold(self.adaptive_threshold)
        
        return mean_brightness, brightness_std

    def enhance_frame_for_lighting(self, frame):
        """Enhance frame based on current lighting conditions"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.lighting_mode == 'low_light':
            # Low light enhancement
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Gamma correction for low light
            gamma = 1.5
            enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
            enhanced = np.uint8(enhanced)
            
        elif self.lighting_mode == 'bright':
            # Bright light handling
            # Reduce contrast to prevent over-saturation
            enhanced = cv2.convertScaleAbs(gray, alpha=0.8, beta=10)
            
        else:
            # Normal lighting
            # Light histogram equalization
            enhanced = cv2.equalizeHist(gray)
        
        return enhanced

    def advanced_motion_detection(self, frame):
        """Advanced motion detection with lighting adaptation"""
        # Enhance frame for current lighting
        enhanced_frame = self.enhance_frame_for_lighting(frame)
        
        # Apply appropriate gaussian blur for lighting conditions
        blur_kernel = self.gaussian_kernels[self.lighting_mode]
        blurred = cv2.GaussianBlur(enhanced_frame, blur_kernel, 0)
        
        # Store in buffer for multi-frame analysis
        self.gray_buffer.append(blurred)
        
        # Background subtraction with enhanced frame
        fg_mask = self.background_subtractor.apply(blurred)
        
        # Multi-frame differencing for validation
        motion_mask = fg_mask.copy()
        if len(self.gray_buffer) >= 3:
            # Frame differencing between current and previous frames
            diff1 = cv2.absdiff(self.gray_buffer[-1], self.gray_buffer[-2])
            diff2 = cv2.absdiff(self.gray_buffer[-2], self.gray_buffer[-3])
            
            # Threshold differences
            thresh1 = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
            thresh2 = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Combine with background subtraction
            combined_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_or(thresh1, thresh2))
            motion_mask = combined_mask
        
        # Noise reduction based on lighting
        kernel = self.morphology_kernels[self.lighting_mode]
        
        # Remove noise
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional filtering for low light
        if self.lighting_mode == 'low_light':
            # More aggressive noise reduction
            motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on lighting conditions
        min_area = self.min_contour_areas[self.lighting_mode]
        max_area = self.frame_width * self.frame_height * 0.3
        
        motion_boxes = []
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Size filtering based on lighting
                min_size = 20 if self.lighting_mode == 'low_light' else 15
                if w > min_size and h > min_size:
                    # Additional aspect ratio filtering
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratios
                        motion_boxes.append((x, y, w, h))
                        total_motion_area += area
        
        # Store motion boxes
        self.motion_boxes = motion_boxes
        
        # Motion confirmation based on lighting
        motion_detected = total_motion_area > (self.sensitivity * 0.5 if self.lighting_mode == 'low_light' else self.sensitivity)
        
        # Add to motion history
        self.motion_history.append(motion_detected)
        
        # Confirm motion based on lighting requirements
        confirmation_needed = self.motion_confirmation_frames[self.lighting_mode]
        if len(self.motion_history) >= confirmation_needed:
            recent_motion = sum(self.motion_history[-confirmation_needed:])
            self.confirmed_motion = recent_motion >= confirmation_needed
        
        return self.confirmed_motion

    def save_photo(self, frame):
        """Save photo with enhanced information overlay"""
        current_time = time.time()
        
        if current_time - self.last_photo_time < self.photo_interval:
            return False
        
        if self.current_session_dir is None:
            self.create_session_directory()
        
        # Create enhanced photo with information overlay
        photo_frame = frame.copy()
        
        # Draw motion detection boxes
        for i, (x, y, w, h) in enumerate(self.motion_boxes):
            # Color based on lighting mode
            if self.lighting_mode == 'low_light':
                color = (0, 255, 255)  # Yellow for low light
            elif self.lighting_mode == 'bright':
                color = (255, 0, 255)  # Magenta for bright
            else:
                color = (0, 255, 0)    # Green for normal
            
            cv2.rectangle(photo_frame, (x, y), (x + w, y + h), color, 2)
            
            # Object label
            label = f"Motion {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.rectangle(photo_frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(photo_frame, label, (x + 2, label_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Enhanced information overlay
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            f"Objects: {len(self.motion_boxes)} | {timestamp_str}",
            f"Mode: {self.lighting_mode.upper()} | Brightness: {self.current_brightness:.0f}",
            f"Threshold: {self.adaptive_threshold} | Sequence: {self.motion_sequence}"
        ]
        
        # Add info background
        info_height = len(info_lines) * 20 + 10
        cv2.rectangle(photo_frame, (10, 10), (400, 10 + info_height), (0, 0, 0), -1)
        
        # Add info text
        for i, line in enumerate(info_lines):
            y_pos = 28 + i * 20
            cv2.putText(photo_frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Generate filename with lighting info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{self.motion_sequence:03d}_{timestamp}_{self.lighting_mode}_{self.photo_count:03d}.jpg"
        filepath = os.path.join(self.current_session_dir, filename)
        
        # Save with appropriate quality
        quality = 90 if self.lighting_mode == 'low_light' else 85
        cv2.imwrite(filepath, photo_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        self.last_photo_time = current_time
        self.photo_count += 1
        
        relative_path = os.path.relpath(filepath, self.script_dir)
        print(f"Photo saved: {relative_path} ({len(self.motion_boxes)} objects, {self.lighting_mode} lighting)")
        return True

    def run(self):
        """Main detection loop with lighting adaptation"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        print("Starting adaptive motion detection...")
        print("Adapting to lighting conditions automatically...")
        print("Press Ctrl+C to stop")
        
        # Allow camera to stabilize
        print("Warming up camera and analyzing lighting...")
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret:
                self.analyze_lighting(frame)
                time.sleep(0.1)
        
        print(f"Initial lighting mode: {self.lighting_mode}")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                current_time = time.time()
                self.frame_count += 1
                
                # Analyze and adapt to lighting
                self.analyze_lighting(frame)
                
                # Detect motion with lighting adaptation
                motion_detected = self.advanced_motion_detection(frame)
                
                if motion_detected:
                    if not self.motion_detected:
                        self.motion_sequence += 1
                        self.photo_count = 0
                        self.create_session_directory()
                        print(f"Motion detected! Sequence {self.motion_sequence} ({self.lighting_mode} lighting)")
                    
                    self.motion_detected = True
                    self.last_motion_time = current_time
                    self.save_photo(frame)
                    
                else:
                    if (self.motion_detected and 
                        current_time - self.last_motion_time > self.motion_timeout):
                        print(f"Motion stopped. Sequence {self.motion_sequence} complete ({self.photo_count} photos)")
                        self.motion_detected = False
                        self.current_session_dir = None
                
                # Status updates
                if self.frame_count % (self.fps * 15) == 0:
                    elapsed = current_time - self.start_time
                    fps_actual = self.frame_count / elapsed
                    status = "MOTION" if self.motion_detected else "MONITORING"
                    objects_str = f" ({len(self.motion_boxes)} objects)" if self.motion_detected else ""
                    print(f"Status: {status}{objects_str} | Mode: {self.lighting_mode.upper()} | "
                          f"FPS: {fps_actual:.1f} | Brightness: {self.current_brightness:.0f} | "
                          f"Threshold: {self.adaptive_threshold}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping motion detection...")
        finally:
            self.cleanup()
        
        return True

    def cleanup(self):
        """Clean up resources and print summary"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        elapsed = time.time() - self.start_time
        print(f"\nSummary:")
        print(f"Runtime: {elapsed:.1f}s | Frames: {self.frame_count}")
        print(f"Motion sequences: {self.motion_sequence}")
        print(f"Lighting changes: {self.lighting_changes}")
        print(f"Final lighting mode: {self.lighting_mode}")
        print(f"Photos saved in: {self.base_photo_dir}")
        
        if os.path.exists(self.base_photo_dir):
            directories = [d for d in os.listdir(self.base_photo_dir) 
                         if os.path.isdir(os.path.join(self.base_photo_dir, d)) and d.startswith('detection_')]
            if directories:
                print(f"Detection sessions:")
                for dir_name in sorted(directories):
                    dir_path = os.path.join(self.base_photo_dir, dir_name)
                    photo_count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
                    print(f"  {dir_name} ({photo_count} photos)")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Motion Detection for All Lighting Conditions")
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
                        help="Camera FPS (default: 5)")
    parser.add_argument("--test-camera", action="store_true",
                        help="Test camera and lighting detection")
    
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
        print("Testing camera and lighting detection...")
        detector = AdaptiveMotionDetector(camera_id=args.camera, resolution=resolution)
        if detector.initialize_camera():
            print("Running 10-second lighting test...")
            for i in range(50):
                ret, frame = detector.cap.read()
                if ret:
                    brightness, _ = detector.analyze_lighting(frame)
                    print(f"Frame {i+1}: Brightness={brightness:.1f}, Mode={detector.lighting_mode}")
                    time.sleep(0.2)
            detector.cleanup()
            print("Camera and lighting test successful!")
            return 0
        else:
            print("Camera test failed!")
            return 1
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run detector
    detector = AdaptiveMotionDetector(
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
