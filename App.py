#!/usr/bin/env python3
"""
Ultra-Optimized Motion Detection Script for Extremely Low-Powered Systems
Designed for systems with <256MB RAM and limited processing power
Modified to capture photos when motion is detected
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
import sys
import gc
from collections import deque

def get_photo_directory():
    """Ask user for photo save directory"""
    while True:
        photo_dir = input("Enter the directory path to save motion photos (or press Enter for default 'photos'): ").strip()
        
        if not photo_dir:
            photo_dir = "photos"
        
        # Expand user path (~) if present
        photo_dir = os.path.expanduser(photo_dir)
        
        try:
            # Try to create directory if it doesn't exist
            os.makedirs(photo_dir, exist_ok=True)
            print(f"Photos will be saved to: {os.path.abspath(photo_dir)}")
            return photo_dir
        except Exception as e:
            print(f"Error creating directory '{photo_dir}': {e}")
            print("Please try a different path.")

class UltraLowPowerMotionDetector:
    def __init__(self, camera_id=0, sensitivity=2000, min_record_time=3, 
                 max_record_time=30, output_dir="recordings", photo_dir=None):
        """
        Ultra-optimized motion detector for very low-powered systems
        
        Args:
            camera_id: USB camera device ID
            sensitivity: Motion detection sensitivity (higher = less sensitive)
            min_record_time: Minimum recording time in seconds
            max_record_time: Maximum recording time in seconds
            output_dir: Directory to save video recordings
            photo_dir: Directory to save motion photos
        """
        self.camera_id = camera_id
        self.sensitivity = sensitivity
        self.min_record_time = min_record_time
        self.max_record_time = max_record_time
        self.output_dir = output_dir
        
        # Ask for photo directory if not provided
        if photo_dir is None:
            self.photo_dir = get_photo_directory()
        else:
            self.photo_dir = photo_dir
            os.makedirs(photo_dir, exist_ok=True)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Ultra-low memory settings
        self.frame_width = 320   # Very small resolution
        self.frame_height = 240
        self.fps = 10            # Lower FPS
        self.process_every_n_frames = 2  # Skip frames for processing
        
        # Lightweight motion detection
        self.background_frame = None
        self.frame_count = 0
        self.background_update_rate = 0.01  # Very slow background update
        
        # Recording variables
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.last_motion_time = None
        
        # Photo capture variables
        self.last_photo_time = 0
        self.photo_cooldown = 2.0  # Minimum seconds between photos
        self.photos_taken = 0
        
        # Camera setup
        self.cap = None
        
        # Control variables
        self.running = False
        self.paused = False
        
        # Statistics
        self.motion_count = 0
        self.total_recordings = 0
        
        # Memory-efficient motion tracking
        self.motion_history = deque(maxlen=5)  # Only keep last 5 motion events
        
        # Frame buffers (pre-allocate to avoid repeated allocation)
        self.gray_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.diff_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.thresh_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        print(f"Ultra-low power motion detector initialized")
        print(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"Memory usage optimized for systems with <256MB RAM")
        print(f"Photos will be saved to: {os.path.abspath(self.photo_dir)}")
    
    def initialize_camera(self):
        """Initialize camera with ultra-low memory settings"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set ultra-low memory properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        
        # Try to use most efficient codec
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        
        # Disable auto-exposure and gain for consistent performance
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        
        print(f"Camera initialized at {self.frame_width}x{self.frame_height}")
        return True
    
    def detect_motion_lightweight(self, frame):
        """Ultra-lightweight motion detection using frame differencing"""
        # Convert to grayscale in-place
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.gray_frame)
        
        # Blur to reduce noise (small kernel for speed)
        cv2.GaussianBlur(self.gray_frame, (5, 5), 0, dst=self.gray_frame)
        
        # Initialize background if needed
        if self.background_frame is None:
            self.background_frame = self.gray_frame.copy()
            return False, 0
        
        # Frame differencing
        cv2.absdiff(self.background_frame, self.gray_frame, dst=self.diff_frame)
        
        # Threshold
        cv2.threshold(self.diff_frame, 25, 255, cv2.THRESH_BINARY, dst=self.thresh_frame)
        
        # Count non-zero pixels (motion area)
        motion_pixels = cv2.countNonZero(self.thresh_frame)
        
        # Update background very slowly
        cv2.addWeighted(self.background_frame, 1 - self.background_update_rate, 
                       self.gray_frame, self.background_update_rate, 0, 
                       dst=self.background_frame)
        
        return motion_pixels > self.sensitivity, motion_pixels
    
    def capture_photo(self, frame):
        """Capture and save a photo when motion is detected"""
        current_time = time.time()
        
        # Check cooldown period to avoid too many photos
        if current_time - self.last_photo_time < self.photo_cooldown:
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.jpg"
        filepath = os.path.join(self.photo_dir, filename)
        
        # Save the photo
        success = cv2.imwrite(filepath, frame)
        
        if success:
            self.last_photo_time = current_time
            self.photos_taken += 1
            print(f"Photo captured: {filename}")
            return filename
        else:
            print(f"Error saving photo: {filename}")
            return None
    
    def start_recording(self):
        """Start ultra-lightweight video recording"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use most efficient codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.total_recordings += 1
        print(f"Recording: {filename}")
        return filename
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        print(f"Recording stopped: {duration:.1f}s")
        
        # Force garbage collection after recording
        gc.collect()
        return duration
    
    def should_stop_recording(self):
        """Check if recording should stop"""
        if not self.is_recording:
            return False
        
        current_time = time.time()
        recording_duration = current_time - self.recording_start_time
        time_since_motion = current_time - self.last_motion_time if self.last_motion_time else 0
        
        # Stop conditions
        if recording_duration >= self.max_record_time:
            return True
        
        if (recording_duration >= self.min_record_time and 
            time_since_motion > 2):  # 2 seconds after last motion
            return True
        
        return False
    
    def process_frame(self, frame):
        """Process frame with minimal overhead"""
        self.frame_count += 1
        
        # Skip frames to reduce processing load
        if self.frame_count % self.process_every_n_frames != 0:
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            return False, 0
        
        if self.paused:
            return False, 0
        
        motion_detected, motion_area = self.detect_motion_lightweight(frame)
        
        if motion_detected:
            self.last_motion_time = time.time()
            self.motion_count += 1
            self.motion_history.append(time.time())
            
            # Capture photo when motion is detected
            self.capture_photo(frame)
            
            if not self.is_recording:
                self.start_recording()
        
        # Record frame
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        # Check if should stop recording
        if self.should_stop_recording():
            self.stop_recording()
        
        return motion_detected, motion_area
    
    def run_detection(self):
        """Main detection loop optimized for low-powered systems"""
        print("Starting ultra-low power motion detection...")
        print("Press Ctrl+C to stop")
        
        if not self.initialize_camera():
            return
        
        self.running = True
        last_status_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                motion_detected, motion_area = self.process_frame(frame)
                
                # Status update every 10 seconds
                current_time = time.time()
                if current_time - last_status_time > 10:
                    status = "REC" if self.is_recording else "MON"
                    motion_rate = len(self.motion_history) / 5.0  # motions per second over last 5 events
                    print(f"[{status}] Motion: {motion_detected} | Rate: {motion_rate:.1f}/s | Total: {self.motion_count} | Photos: {self.photos_taken}")
                    last_status_time = current_time
                    
                    # Force garbage collection periodically
                    if self.frame_count % 300 == 0:  # Every 30 seconds at 10fps
                        gc.collect()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        gc.collect()
        print(f"Detection stopped. Total photos taken: {self.photos_taken}")
        print("Cleanup complete")

class MinimalGUI:
    """Extremely minimal GUI for ultra-low power systems"""
    def __init__(self):
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            self.tk = tk
            self.ttk = ttk
            self.filedialog = filedialog
            self.messagebox = messagebox
            self.gui_available = True
        except ImportError:
            self.gui_available = False
            print("GUI not available - running in command line mode")
            return
        
        self.root = tk.Tk()
        self.root.title("Ultra-Low Power Motion Detector")
        self.root.geometry("450x350")
        
        self.detector = None
        self.detection_thread = None
        self.photo_dir = None
        
        # Simple variables
        self.status_var = tk.StringVar(value="Ready")
        self.motion_count_var = tk.IntVar(value=0)
        self.recording_var = tk.StringVar(value="No")
        self.photos_taken_var = tk.IntVar(value=0)
        self.photo_dir_var = tk.StringVar(value="No directory selected")
        
        self.setup_minimal_gui()
    
    def setup_minimal_gui(self):
        """Setup minimal GUI"""
        if not self.gui_available:
            return
        
        # Main frame
        main_frame = self.ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Photo directory selection
        dir_frame = self.ttk.Frame(main_frame)
        dir_frame.pack(fill='x', pady=5)
        
        self.ttk.Label(dir_frame, text="Photo Directory:", font=('Arial', 12, 'bold')).pack(anchor='w')
        self.ttk.Label(dir_frame, textvariable=self.photo_dir_var, font=('Arial', 9), wraplength=400).pack(anchor='w')
        
        self.ttk.Button(dir_frame, text="Select Directory", command=self.select_photo_directory).pack(anchor='w', pady=5)
        
        # Status display
        self.ttk.Label(main_frame, text="Status:", font=('Arial', 12, 'bold')).pack(pady=5)
        self.ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 10)).pack(pady=5)
        
        self.ttk.Label(main_frame, text="Motion Count:", font=('Arial', 12, 'bold')).pack(pady=5)
        self.ttk.Label(main_frame, textvariable=self.motion_count_var, font=('Arial', 10)).pack(pady=5)
        
        self.ttk.Label(main_frame, text="Recording:", font=('Arial', 12, 'bold')).pack(pady=5)
        self.ttk.Label(main_frame, textvariable=self.recording_var, font=('Arial', 10)).pack(pady=5)
        
        self.ttk.Label(main_frame, text="Photos Taken:", font=('Arial', 12, 'bold')).pack(pady=5)
        self.ttk.Label(main_frame, textvariable=self.photos_taken_var, font=('Arial', 10)).pack(pady=5)
        
        # Control buttons
        control_frame = self.ttk.Frame(main_frame)
        control_frame.pack(pady=20)
        
        self.start_btn = self.ttk.Button(control_frame, text="Start", command=self.start_detection)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = self.ttk.Button(control_frame, text="Stop", command=self.stop_detection, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Update status periodically
        self.update_status()
    
    def select_photo_directory(self):
        """Select directory for saving photos"""
        directory = self.filedialog.askdirectory(title="Select directory to save motion photos")
        if directory:
            self.photo_dir = directory
            self.photo_dir_var.set(directory)
            print(f"Photo directory set to: {directory}")
    
    def start_detection(self):
        """Start detection in thread"""
        if not self.gui_available:
            return
        
        if not self.photo_dir:
            self.messagebox.showerror("Error", "Please select a directory to save photos first!")
            return
        
        self.detector = UltraLowPowerMotionDetector(photo_dir=self.photo_dir)
        
        def detection_thread():
            self.detector.run_detection()
        
        self.detection_thread = threading.Thread(target=detection_thread, daemon=True)
        self.detection_thread.start()
        
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("Running")
    
    def stop_detection(self):
        """Stop detection"""
        if self.detector:
            self.detector.running = False
            self.detector.cleanup()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Stopped")
    
    def update_status(self):
        """Update status display"""
        if not self.gui_available:
            return
        
        if self.detector:
            self.motion_count_var.set(self.detector.motion_count)
            self.recording_var.set("Yes" if self.detector.is_recording else "No")
            self.photos_taken_var.set(self.detector.photos_taken)
        
        self.root.after(1000, self.update_status)  # Update every second
    
    def run(self):
        """Run GUI"""
        if not self.gui_available:
            # Fallback to command line
            detector = UltraLowPowerMotionDetector()
            detector.run_detection()
            return
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Ultra-Low Power Motion Detection with Photo Capture")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--sensitivity", type=int, default=2000, help="Motion sensitivity")
    parser.add_argument("--min-time", type=int, default=3, help="Min recording time")
    parser.add_argument("--max-time", type=int, default=30, help="Max recording time")
    parser.add_argument("--output", type=str, default="recordings", help="Output directory for videos")
    parser.add_argument("--photos", type=str, help="Photo directory (if not specified, will ask user)")
    
    args = parser.parse_args()
    
    if args.nogui:
        # Command line mode
        detector = UltraLowPowerMotionDetector(
            camera_id=args.camera,
            sensitivity=args.sensitivity,
            min_record_time=args.min_time,
            max_record_time=args.max_time,
            output_dir=args.output,
            photo_dir=args.photos
        )
        detector.run_detection()
    else:
        # Try minimal GUI
        gui = MinimalGUI()
        gui.run()

if __name__ == "__main__":
    main()
