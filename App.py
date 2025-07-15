#!/usr/bin/env python3
"""
Motion Detection and Recording Script for Raspberry Pi with GUI
Optimized for 512MB RAM systems with USB camera support
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
        
        # Control variables
        self.running = False
        self.paused = False
        
        # Statistics
        self.motion_count = 0
        self.total_recordings = 0
        
        # Movement tracking
        self.motion_trail = []  # Store recent motion centers
        self.max_trail_length = 20  # Number of trail points to keep
        self.motion_boxes = []  # Store bounding boxes of motion areas
        
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
        """Detect motion in frame and track movement"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion area and track movement
        motion_area = 0
        motion_centers = []
        motion_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small movements
                motion_area += area
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))
                
                # Calculate center of motion
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    motion_centers.append((cx, cy))
        
        # Update motion trail if motion detected
        if motion_centers:
            # Use the largest motion center or average if multiple
            if len(motion_centers) == 1:
                center = motion_centers[0]
            else:
                # Average all centers
                avg_x = sum(c[0] for c in motion_centers) // len(motion_centers)
                avg_y = sum(c[1] for c in motion_centers) // len(motion_centers)
                center = (avg_x, avg_y)
            
            # Add to trail
            self.motion_trail.append(center)
            if len(self.motion_trail) > self.max_trail_length:
                self.motion_trail.pop(0)
        
        # Update motion boxes
        self.motion_boxes = motion_boxes
        
        return motion_area > self.sensitivity, motion_area
    
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
        self.total_recordings += 1
        print(f"Started recording: {filename}")
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
        print(f"Stopped recording after {duration:.1f} seconds")
        return duration
    
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
        if self.paused:
            return False, 0, frame
        
        motion_detected, motion_area = self.detect_motion(frame)
        
        if motion_detected:
            self.last_motion_time = time.time()
            self.motion_count += 1
            if not self.is_recording:
                self.start_recording()
        
        # Record frame if recording
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        # Check if should stop recording
        if self.should_stop_recording():
            self.stop_recording()
        
        # Add visual indicators to frame
        display_frame = frame.copy()
        
        # Draw motion bounding boxes
        for x, y, w, h in self.motion_boxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add motion area label
            cv2.putText(display_frame, f"Motion: {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw motion trail
        if len(self.motion_trail) > 1:
            # Draw trail lines
            for i in range(1, len(self.motion_trail)):
                # Calculate color fade (newer points are brighter)
                alpha = i / len(self.motion_trail)
                color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))  # Purple to bright purple
                
                # Draw line between consecutive points
                cv2.line(display_frame, self.motion_trail[i-1], self.motion_trail[i], color, 2)
                
                # Draw circle at each trail point
                cv2.circle(display_frame, self.motion_trail[i], 3, color, -1)
        
        # Draw current motion center
        if self.motion_trail:
            current_center = self.motion_trail[-1]
            cv2.circle(display_frame, current_center, 8, (0, 0, 255), -1)  # Red dot for current position
            cv2.circle(display_frame, current_center, 12, (0, 0, 255), 2)   # Red circle outline
        
        # Add recording indicator
        if self.is_recording:
            cv2.rectangle(display_frame, (10, 10), (100, 40), (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add motion indicator
        if motion_detected:
            cv2.rectangle(display_frame, (10, 50), (120, 80), (0, 255, 0), -1)
            cv2.putText(display_frame, "MOTION", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add trail info
        if self.motion_trail:
            trail_info = f"Trail: {len(self.motion_trail)} points"
            cv2.putText(display_frame, trail_info, (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add motion boxes count
        if self.motion_boxes:
            boxes_info = f"Objects: {len(self.motion_boxes)}"
            cv2.putText(display_frame, boxes_info, (10, display_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return motion_detected, motion_area, display_frame
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

class MotionDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("USB Motion Detector - Raspberry Pi")
        self.root.geometry("800x600")
        
        # Initialize detector
        self.detector = None
        self.detection_thread = None
        
        # GUI variables
        self.camera_id = tk.IntVar(value=0)
        self.sensitivity = tk.IntVar(value=1000)
        self.min_time = tk.IntVar(value=5)
        self.max_time = tk.IntVar(value=60)
        self.output_dir = tk.StringVar(value="recordings")
        
        # Status variables
        self.status_text = tk.StringVar(value="Ready")
        self.motion_count = tk.IntVar(value=0)
        self.recording_count = tk.IntVar(value=0)
        self.current_recording = tk.StringVar(value="None")
        self.trail_length = tk.IntVar(value=0)
        self.object_count = tk.IntVar(value=0)
        
        # Frame queue for video display
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.setup_gui()
        
        # Start frame update loop
        self.update_frame()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Camera settings
        ttk.Label(settings_frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(settings_frame, from_=0, to=9, textvariable=self.camera_id, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Sensitivity:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Scale(settings_frame, from_=100, to=5000, variable=self.sensitivity, orient=tk.HORIZONTAL).grid(row=0, column=3, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Label(settings_frame, textvariable=self.sensitivity).grid(row=0, column=4, sticky=tk.W)
        
        # Recording settings
        ttk.Label(settings_frame, text="Min Time (s):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(settings_frame, from_=1, to=60, textvariable=self.min_time, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Max Time (s):").grid(row=1, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Spinbox(settings_frame, from_=10, to=300, textvariable=self.max_time, width=10).grid(row=1, column=3, sticky=tk.W)
        
        # Output directory
        ttk.Label(settings_frame, text="Output Dir:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(settings_frame, textvariable=self.output_dir, width=30).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(settings_frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=3, sticky=tk.W)
        
        # Configure settings frame grid weights
        settings_frame.columnconfigure(3, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_detection, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Video display frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here")
        self.video_label.pack(expand=True)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(status_frame, text="Status:").pack(anchor=tk.W)
        ttk.Label(status_frame, textvariable=self.status_text, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Motion Events:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.motion_count, font=("Arial", 12)).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Total Recordings:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.recording_count, font=("Arial", 12)).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Current Recording:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.current_recording, font=("Arial", 10), wraplength=200).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Motion Trail Points:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.trail_length, font=("Arial", 12)).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Moving Objects:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(status_frame, textvariable=self.object_count, font=("Arial", 12)).pack(anchor=tk.W)
        
        # Clear trail button
        ttk.Button(status_frame, text="Clear Trail", command=self.clear_trail).pack(anchor=tk.W, pady=(10, 0))
        
        # Configure main frame grid weights
        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
    
    def clear_trail(self):
        """Clear motion trail"""
        if self.detector:
            self.detector.motion_trail.clear()
            self.detector.motion_boxes.clear()
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
    
    def start_detection(self):
        """Start motion detection"""
        try:
            # Create detector with current settings
            self.detector = MotionDetector(
                camera_id=self.camera_id.get(),
                sensitivity=self.sensitivity.get(),
                min_record_time=self.min_time.get(),
                max_record_time=self.max_time.get(),
                output_dir=self.output_dir.get()
            )
            
            # Initialize camera
            if not self.detector.initialize_camera():
                messagebox.showerror("Error", "Failed to initialize USB camera")
                return
            
            # Start detection thread
            self.detector.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.status_text.set("Running")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def pause_detection(self):
        """Pause/resume detection"""
        if self.detector:
            self.detector.paused = not self.detector.paused
            if self.detector.paused:
                self.pause_button.config(text="Resume")
                self.status_text.set("Paused")
            else:
                self.pause_button.config(text="Pause")
                self.status_text.set("Running")
    
    def stop_detection(self):
        """Stop motion detection"""
        if self.detector:
            self.detector.running = False
            self.detector.cleanup()
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.stop_button.config(state=tk.DISABLED)
        self.status_text.set("Stopped")
        self.current_recording.set("None")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.detector.running:
            try:
                ret, frame = self.detector.cap.read()
                if not ret:
                    break
                
                motion_detected, motion_area, display_frame = self.detector.process_frame(frame)
                
                # Update statistics
                self.motion_count.set(self.detector.motion_count)
                self.recording_count.set(self.detector.total_recordings)
                self.trail_length.set(len(self.detector.motion_trail))
                self.object_count.set(len(self.detector.motion_boxes))
                
                if self.detector.is_recording:
                    duration = time.time() - self.detector.recording_start_time
                    self.current_recording.set(f"Recording... ({duration:.1f}s)")
                else:
                    self.current_recording.set("None")
                
                # Put frame in queue for display
                if not self.frame_queue.full():
                    # Resize frame for display
                    display_frame = cv2.resize(display_frame, (320, 240))
                    try:
                        self.frame_queue.put_nowait(display_frame)
                    except queue.Full:
                        pass
                
                time.sleep(0.033)  # ~30 FPS for display
                
            except Exception as e:
                print(f"Detection error: {e}")
                break
    
    def update_frame(self):
        """Update video frame display"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update video label
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep reference
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        # Schedule next update
        self.root.after(50, self.update_frame)  # 20 FPS display update
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="USB Motion Detection GUI for Raspberry Pi")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI (command line mode)")
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
    
    if args.nogui:
        # Run in command line mode
        print("Initializing USB camera motion detection system...")
        print("Make sure your USB camera is connected before starting")
        
        detector = MotionDetector(
            camera_id=args.camera,
            sensitivity=args.sensitivity,
            min_record_time=args.min_time,
            max_record_time=args.max_time,
            output_dir=args.output
        )
        
        if detector.initialize_camera():
            detector.running = True
            try:
                while detector.running:
                    ret, frame = detector.cap.read()
                    if not ret:
                        break
                    
                    motion_detected, motion_area, display_frame = detector.process_frame(frame)
                    
                    if detector.motion_count % 30 == 0:  # Status every 2 seconds at 15fps
                        status = "RECORDING" if detector.is_recording else "MONITORING"
                        print(f"Status: {status} | Motion: {'YES' if motion_detected else 'NO'}")
                    
                    time.sleep(0.033)
                    
            except KeyboardInterrupt:
                print("\nStopping motion detection...")
            finally:
                detector.cleanup()
    else:
        # Run GUI mode
        root = tk.Tk()
        app = MotionDetectorGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()

if __name__ == "__main__":
    main()
