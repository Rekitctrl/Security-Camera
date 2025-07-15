#!/usr/bin/env python3
"""
Enhanced Motion Detection and Recording Script for Raspberry Pi
Optimized for 512MB RAM systems with USB camera support
Added features: Email alerts, zone detection, improved error handling
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import queue
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, camera_id=0, sensitivity=1000, min_record_time=5, 
                 max_record_time=60, output_dir="recordings", config_file="motion_config.json"):
        """
        Initialize motion detector with enhanced features
        """
        self.camera_id = self.find_usb_camera(camera_id)
        self.sensitivity = sensitivity
        self.min_record_time = min_record_time
        self.max_record_time = max_record_time
        self.output_dir = output_dir
        self.config_file = config_file
        
        # Load configuration
        self.config = self.load_config()
        
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
        self.current_filename = None
        
        # Camera setup
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 15
        
        # Control variables
        self.running = False
        self.paused = False
        
        # Statistics
        self.motion_count = 0
        self.total_recordings = 0
        self.session_start_time = time.time()
        
        # Movement tracking
        self.motion_trail = []
        self.max_trail_length = 20
        self.motion_boxes = []
        
        # Detection zones (optional)
        self.detection_zones = self.config.get('detection_zones', [])
        
        # Email alerts
        self.email_enabled = self.config.get('email_enabled', False)
        self.email_config = self.config.get('email_config', {})
        self.last_email_time = 0
        self.email_cooldown = 300  # 5 minutes
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        
        logger.info(f"Motion detector initialized with camera ID: {self.camera_id}")
        logger.info(f"Sensitivity: {sensitivity}, Recording: {min_record_time}-{max_record_time}s")
    
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            'email_enabled': False,
            'email_config': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipient_email': ''
            },
            'detection_zones': [],
            'save_motion_images': True,
            'motion_threshold_percent': 0.5
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                self.save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def find_usb_camera(self, preferred_id=0):
        """Enhanced USB camera detection with better error handling"""
        logger.info("Scanning for USB cameras...")
        
        available_cameras = []
        for camera_id in range(10):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(camera_id)
                        logger.info(f"Found working USB camera at index {camera_id}")
                        if camera_id == preferred_id:
                            cap.release()
                            return camera_id
                cap.release()
            except Exception as e:
                logger.warning(f"Error checking camera {camera_id}: {e}")
        
        if available_cameras:
            logger.info(f"Using first available camera: {available_cameras[0]}")
            return available_cameras[0]
        else:
            logger.warning(f"No working USB cameras found, using default: {preferred_id}")
            return preferred_id
    
    def initialize_camera(self):
        """Initialize USB camera with enhanced error handling"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera at index {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Verify camera is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Camera not providing frames")
                return False
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.frame_width = actual_width
            self.frame_height = actual_height
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def point_in_zones(self, point):
        """Check if point is in any detection zone"""
        if not self.detection_zones:
            return True  # No zones defined = entire frame is active
        
        x, y = point
        for zone in self.detection_zones:
            if (zone['x'] <= x <= zone['x'] + zone['width'] and 
                zone['y'] <= y <= zone['y'] + zone['height']):
                return True
        return False
    
    def detect_motion(self, frame):
        """Enhanced motion detection with zone support"""
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_area = 0
            motion_centers = []
            motion_boxes = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small movements
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2
                    
                    # Check if motion is in detection zone
                    if self.point_in_zones((center_x, center_y)):
                        motion_area += area
                        motion_boxes.append((x, y, w, h))
                        motion_centers.append((center_x, center_y))
            
            # Update motion trail
            if motion_centers:
                if len(motion_centers) == 1:
                    center = motion_centers[0]
                else:
                    # Average all centers
                    avg_x = sum(c[0] for c in motion_centers) // len(motion_centers)
                    avg_y = sum(c[1] for c in motion_centers) // len(motion_centers)
                    center = (avg_x, avg_y)
                
                self.motion_trail.append(center)
                if len(self.motion_trail) > self.max_trail_length:
                    self.motion_trail.pop(0)
            
            self.motion_boxes = motion_boxes
            
            # Calculate motion as percentage of frame
            frame_area = self.frame_width * self.frame_height
            motion_percentage = (motion_area / frame_area) * 100
            
            motion_detected = motion_area > self.sensitivity
            
            return motion_detected, motion_area, motion_percentage
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0, 0
    
    def send_email_alert(self, image_path=None):
        """Send email alert with optional image attachment"""
        if not self.email_enabled or not self.email_config.get('sender_email'):
            return
        
        current_time = time.time()
        if current_time - self.last_email_time < self.email_cooldown:
            return  # Cooldown period
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = f"Motion Detected - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            body = f"""
            Motion detected by security camera.
            
            Detection Details:
            - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Camera: USB Camera {self.camera_id}
            - Motion Events: {self.motion_count}
            - Total Recordings: {self.total_recordings}
            - Currently Recording: {'Yes' if self.is_recording else 'No'}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='motion_snapshot.jpg')
                    msg.attach(image)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['sender_email'], self.email_config['recipient_email'], text)
            server.quit()
            
            self.last_email_time = current_time
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    def save_motion_snapshot(self, frame):
        """Save snapshot when motion is detected"""
        if not self.config.get('save_motion_images', True):
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_snapshot_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            logger.info(f"Motion snapshot saved: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return None
    
    def start_recording(self):
        """Start video recording with enhanced error handling"""
        if self.is_recording:
            return self.current_filename
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filepath, fourcc, self.fps, (self.frame_width, self.frame_height)
            )
            
            if not self.video_writer.isOpened():
                logger.error("Failed to open video writer")
                return None
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.current_filename = filename
            self.total_recordings += 1
            
            logger.info(f"Started recording: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return None
    
    def stop_recording(self):
        """Stop video recording with enhanced cleanup"""
        if not self.is_recording:
            return 0
        
        try:
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            logger.info(f"Stopped recording {self.current_filename} after {duration:.1f}s")
            self.current_filename = None
            return duration
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return 0
    
    def update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.actual_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self, frame):
        """Enhanced frame processing with error handling"""
        try:
            if self.paused:
                return False, 0, frame
            
            # Update FPS counter
            self.update_fps_counter()
            
            motion_detected, motion_area, motion_percentage = self.detect_motion(frame)
            
            if motion_detected:
                self.last_motion_time = time.time()
                self.motion_count += 1
                
                # Start recording if not already recording
                if not self.is_recording:
                    self.start_recording()
                
                # Save snapshot and send email (with cooldown)
                if self.motion_count % 30 == 1:  # Every ~2 seconds at 15fps
                    snapshot_path = self.save_motion_snapshot(frame)
                    if self.email_enabled:
                        threading.Thread(target=self.send_email_alert, 
                                       args=(snapshot_path,), daemon=True).start()
            
            # Record frame if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            
            # Check if should stop recording
            if self.should_stop_recording():
                self.stop_recording()
            
            # Create display frame with enhanced visualizations
            display_frame = self.create_display_frame(frame, motion_detected, motion_percentage)
            
            return motion_detected, motion_area, display_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return False, 0, frame
    
    def create_display_frame(self, frame, motion_detected, motion_percentage):
        """Create enhanced display frame with overlays"""
        display_frame = frame.copy()
        
        # Draw detection zones
        for zone in self.detection_zones:
            cv2.rectangle(display_frame, 
                         (zone['x'], zone['y']), 
                         (zone['x'] + zone['width'], zone['y'] + zone['height']),
                         (255, 255, 0), 2)
            cv2.putText(display_frame, zone.get('name', 'Zone'), 
                       (zone['x'], zone['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw motion bounding boxes
        for x, y, w, h in self.motion_boxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Motion: {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw motion trail
        if len(self.motion_trail) > 1:
            for i in range(1, len(self.motion_trail)):
                alpha = i / len(self.motion_trail)
                color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
                cv2.line(display_frame, self.motion_trail[i-1], self.motion_trail[i], color, 2)
                cv2.circle(display_frame, self.motion_trail[i], 3, color, -1)
        
        # Current motion center
        if self.motion_trail:
            current_center = self.motion_trail[-1]
            cv2.circle(display_frame, current_center, 8, (0, 0, 255), -1)
            cv2.circle(display_frame, current_center, 12, (0, 0, 255), 2)
        
        # Status overlays
        y_offset = 20
        if self.is_recording:
            cv2.rectangle(display_frame, (10, 10), (100, 40), (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if motion_detected:
            cv2.rectangle(display_frame, (10, 50), (120, 80), (0, 255, 0), -1)
            cv2.putText(display_frame, "MOTION", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Information overlay
        info_lines = [
            f"FPS: {self.actual_fps:.1f}",
            f"Motion: {motion_percentage:.1f}%",
            f"Trail: {len(self.motion_trail)}",
            f"Objects: {len(self.motion_boxes)}",
            f"Events: {self.motion_count}",
            f"Recordings: {self.total_recordings}"
        ]
        
        for i, line in enumerate(info_lines):
            y = display_frame.shape[0] - 120 + (i * 20)
            cv2.putText(display_frame, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def should_stop_recording(self):
        """Enhanced recording stop logic"""
        if not self.is_recording:
            return False
        
        current_time = time.time()
        recording_duration = current_time - self.recording_start_time
        time_since_motion = current_time - self.last_motion_time if self.last_motion_time else 0
        
        # Stop if max time reached
        if recording_duration >= self.max_record_time:
            return True
        
        # Stop if no motion for a while (after minimum time)
        if (recording_duration >= self.min_record_time and 
            time_since_motion > 3):
            return True
        
        return False
    
    def cleanup(self):
        """Enhanced cleanup with proper error handling"""
        try:
            logger.info("Starting cleanup...")
            self.running = False
            
            if self.is_recording:
                self.stop_recording()
            
            if self.cap:
                self.cap.release()
            
            cv2.destroyAllWindows()
            
            # Save final statistics
            session_duration = time.time() - self.session_start_time
            logger.info(f"Session completed: {session_duration:.1f}s, {self.motion_count} motion events, {self.total_recordings} recordings")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# The MotionDetectorGUI class would follow similar enhancement patterns
# with better error handling, configuration management, and additional features

def main():
    parser = argparse.ArgumentParser(description="Enhanced USB Motion Detection for Raspberry Pi")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    parser.add_argument("--camera", type=int, default=0, help="USB Camera ID")
    parser.add_argument("--sensitivity", type=int, default=1000, help="Motion sensitivity")
    parser.add_argument("--min-time", type=int, default=5, help="Minimum recording time")
    parser.add_argument("--max-time", type=int, default=60, help="Maximum recording time")
    parser.add_argument("--output", type=str, default="recordings", help="Output directory")
    parser.add_argument("--config", type=str, default="motion_config.json", help="Config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.nogui:
        detector = MotionDetector(
            camera_id=args.camera,
            sensitivity=args.sensitivity,
            min_record_time=args.min_time,
            max_record_time=args.max_time,
            output_dir=args.output,
            config_file=args.config
        )
        
        if detector.initialize_camera():
            detector.running = True
            try:
                logger.info("Starting motion detection loop...")
                while detector.running:
                    ret, frame = detector.cap.read()
                    if not ret:
                        logger.error("Failed to read frame")
                        break
                    
                    motion_detected, motion_area, display_frame = detector.process_frame(frame)
                    
                    # Periodic status updates
                    if detector.motion_count % 60 == 0:  # Every 4 seconds at 15fps
                        status = "RECORDING" if detector.is_recording else "MONITORING"
                        logger.info(f"Status: {status} | Motion: {'YES' if motion_detected else 'NO'} | FPS: {detector.actual_fps:.1f}")
                    
                    time.sleep(0.033)
                    
            except KeyboardInterrupt:
                logger.info("Stopping motion detection...")
            finally:
                detector.cleanup()
        else:
            logger.error("Failed to initialize camera")
            sys.exit(1)
    else:
        # GUI mode would be implemented here
        logger.info("GUI mode not implemented in this enhanced version")
        logger.info("Use --nogui for command line operation")

if __name__ == "__main__":
    main()
