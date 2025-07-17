#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import json
import pickle
import threading
from datetime import datetime, timedelta
import argparse
import sys
import signal
import gc
from collections import deque, defaultdict
import statistics

class LearningMotionDetector:
    def __init__(self, camera_id=0, sensitivity=25, photo_interval=1.5,
                 motion_timeout=3.0, resolution=(160, 120), fps=3):
        self.camera_id = camera_id
        self.base_sensitivity = sensitivity
        self.current_sensitivity = sensitivity
        self.photo_interval = photo_interval
        self.motion_timeout = motion_timeout
        self.frame_width, self.frame_height = resolution
        self.fps = fps

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_photo_dir = os.path.join(self.script_dir, "motion_photos")
        self.learning_dir = os.path.join(self.script_dir, "learning_data")
        os.makedirs(self.base_photo_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)

        self.current_date = None
        self.photo_dir = None
        self.current_detection_dir = None
        
        self.learning_file = os.path.join(self.learning_dir, "learning_progress.json")
        self.patterns_file = os.path.join(self.learning_dir, "motion_patterns.pkl")
        self.stats_file = os.path.join(self.learning_dir, "detection_stats.json")
        
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        self.motion_threshold = 0.3
        
        self.learning_data = self.load_learning_data()
        self.motion_patterns = self.load_motion_patterns()
        self.detection_stats = self.load_detection_stats()
        
        self.hourly_stats = defaultdict(lambda: {'detections': 0, 'false_positives': 0, 'sensitivity': sensitivity})
        self.daily_patterns = defaultdict(list)
        self.lighting_conditions = deque(maxlen=50)
        self.motion_confidence_history = deque(maxlen=20)
        
        self.ambient_light_baseline = None
        self.noise_baseline = None
        self.typical_motion_areas = []
        
        self.cap = None
        self.running = False
        self.last_motion_time = 0
        self.last_photo_time = 0
        self.motion_detected = False
        self.photo_count = 0
        self.motion_sequence = 0
        self.motion_boxes = []
        self.last_save_time = time.time()
        
        self.frame_count = 0
        self.start_time = time.time()
        self.kernel = np.ones((3, 3), np.uint8)
        self.frame_area = self.frame_width * self.frame_height
        
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.adaptation_interval = 300
        self.last_adaptation = time.time()
        
        print(f"🧠 Self-Learning Motion Detector initialized")
        print(f"📊 Learning sessions: {self.learning_data.get('sessions', 0)}")
        print(f"🎯 Detection accuracy: {self.learning_data.get('accuracy', 0):.1f}%")
        print(f"⚙️  Current sensitivity: {self.current_sensitivity}")

    def get_date_photo_dir(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if self.current_date != current_date:
            self.current_date = current_date
            self.photo_dir = os.path.join(self.base_photo_dir, current_date)
            os.makedirs(self.photo_dir, exist_ok=True)
            print(f"📁 Created/Using date folder: {self.photo_dir}")
        
        return self.photo_dir

    def get_detection_folder(self):
        detection_time = datetime.now().strftime("%H%M%S")
        date_dir = self.get_date_photo_dir()
        detection_dir = os.path.join(date_dir, f"detection_{detection_time}")
        os.makedirs(detection_dir, exist_ok=True)
        return detection_dir

    def get_folder_stats(self):
        if not os.path.exists(self.base_photo_dir):
            return {}
        
        stats = {}
        for date_folder in os.listdir(self.base_photo_dir):
            date_path = os.path.join(self.base_photo_dir, date_folder)
            if os.path.isdir(date_path):
                detection_count = 0
                photo_count = 0
                
                for detection_folder in os.listdir(date_path):
                    detection_path = os.path.join(date_path, detection_folder)
                    if os.path.isdir(detection_path) and detection_folder.startswith('detection_'):
                        detection_count += 1
                        photos = [f for f in os.listdir(detection_path) if f.endswith('.jpg')]
                        photo_count += len(photos)
                
                stats[date_folder] = {
                    'detections': detection_count,
                    'photos': photo_count
                }
        
        return stats

    def load_learning_data(self):
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    data = json.load(f)
                    if 'version' not in data:
                        data['version'] = '1.0'
                    return data
        except Exception as e:
            print(f"⚠️ Could not load learning data: {e}")
        
        return {
            'version': '1.0',
            'sessions': 0,
            'total_detections': 0,
            'confirmed_detections': 0,
            'false_positives': 0,
            'accuracy': 0.0,
            'avg_sensitivity': self.base_sensitivity,
            'hourly_patterns': {},
            'environmental_baselines': {},
            'last_updated': datetime.now().isoformat()
        }

    def load_motion_patterns(self):
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Could not load motion patterns: {e}")
        
        return {
            'motion_shapes': [],
            'motion_velocities': [],
            'motion_durations': [],
            'false_positive_patterns': [],
            'confirmed_patterns': []
        }

    def load_detection_stats(self):
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load detection stats: {e}")
        
        return {
            'hourly_activity': {},
            'daily_patterns': {},
            'lighting_correlations': {},
            'motion_zones': {}
        }

    def save_learning_progress(self):
        try:
            self.learning_data['sessions'] += 1
            self.learning_data['last_updated'] = datetime.now().isoformat()
            
            total_detections = self.learning_data['total_detections']
            if total_detections > 0:
                self.learning_data['accuracy'] = (
                    self.learning_data['confirmed_detections'] / total_detections * 100
                )
            
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
            
            with open(self.patterns_file, 'wb') as f:
                pickle.dump(self.motion_patterns, f)
            
            with open(self.stats_file, 'w') as f:
                json.dump(self.detection_stats, f, indent=2)
            
            print(f"💾 Learning progress saved - Accuracy: {self.learning_data['accuracy']:.1f}%")
            
        except Exception as e:
            print(f"⚠️ Could not save learning progress: {e}")

    def analyze_lighting_conditions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        lighting_info = {
            'brightness': avg_brightness,
            'contrast': brightness_std,
            'timestamp': time.time()
        }
        self.lighting_conditions.append(lighting_info)
        
        if self.ambient_light_baseline is None and len(self.lighting_conditions) > 20:
            brightnesses = [lc['brightness'] for lc in self.lighting_conditions]
            self.ambient_light_baseline = statistics.median(brightnesses)
            print(f"🌞 Ambient light baseline established: {self.ambient_light_baseline:.1f}")
        
        return avg_brightness, brightness_std

    def calculate_motion_confidence(self, motion_boxes, motion_percent, brightness):
        confidence = 0.5
        
        if motion_boxes:
            total_area = sum(w * h for _, _, w, h in motion_boxes)
            area_ratio = total_area / self.frame_area
            
            if 0.05 <= area_ratio <= 0.3:
                confidence += 0.2
            elif area_ratio < 0.05:
                confidence -= 0.1
            elif area_ratio > 0.5:
                confidence -= 0.2
        
        if len(self.motion_history) >= 3:
            recent_motion = list(self.motion_history)[-3:]
            motion_variance = statistics.variance(recent_motion) if len(recent_motion) > 1 else 0
            if motion_variance < 0.1:
                confidence += 0.15
        
        if self.ambient_light_baseline and brightness:
            light_diff = abs(brightness - self.ambient_light_baseline)
            if light_diff < 20:
                confidence += 0.1
            elif light_diff > 50:
                confidence -= 0.15
        
        current_hour = datetime.now().hour
        if str(current_hour) in self.learning_data.get('hourly_patterns', {}):
            hour_data = self.learning_data['hourly_patterns'][str(current_hour)]
            if hour_data.get('typical_activity', 0) > 0.5:
                confidence += 0.1
        
        if motion_boxes:
            for x, y, w, h in motion_boxes:
                aspect_ratio = w / h if h > 0 else 1
                if 0.5 <= aspect_ratio <= 2.0:
                    confidence += 0.05
        
        return max(0.0, min(1.0, confidence))

    def adapt_sensitivity(self):
        if len(self.motion_confidence_history) < 5:
            return
        
        avg_confidence = statistics.mean(self.motion_confidence_history)
        
        if avg_confidence < 0.4:
            self.current_sensitivity = max(10, self.current_sensitivity * 0.9)
            print(f"📈 Sensitivity increased to {self.current_sensitivity:.1f}")
        elif avg_confidence > 0.8:
            self.current_sensitivity = min(100, self.current_sensitivity * 1.1)
            print(f"📉 Sensitivity decreased to {self.current_sensitivity:.1f}")
        
        if self.ambient_light_baseline:
            recent_brightness = [lc['brightness'] for lc in list(self.lighting_conditions)[-10:]]
            if recent_brightness:
                current_brightness = statistics.mean(recent_brightness)
                light_ratio = current_brightness / self.ambient_light_baseline
                
                if light_ratio < 0.5:
                    self.motion_threshold = 0.2
                elif light_ratio > 1.5:
                    self.motion_threshold = 0.4
                else:
                    self.motion_threshold = 0.3

    def learn_from_detection(self, motion_boxes, motion_percent, confidence):
        current_time = time.time()
        current_hour = datetime.now().hour
        
        hour_key = str(current_hour)
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {'detections': 0, 'false_positives': 0, 'sensitivity': self.current_sensitivity}
        
        self.hourly_stats[hour_key]['detections'] += 1
        
        if motion_boxes:
            pattern = {
                'boxes': motion_boxes,
                'confidence': confidence,
                'timestamp': current_time,
                'hour': current_hour,
                'motion_percent': motion_percent
            }
            
            if confidence > self.confidence_threshold:
                self.motion_patterns['confirmed_patterns'].append(pattern)
                self.learning_data['confirmed_detections'] += 1
            else:
                self.motion_patterns['false_positive_patterns'].append(pattern)
                self.learning_data['false_positives'] += 1
        
        self.learning_data['total_detections'] += 1
        
        self.motion_confidence_history.append(confidence)
        
        if current_time - self.last_adaptation > self.adaptation_interval:
            self.adapt_sensitivity()
            self.last_adaptation = current_time

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {self.camera_id}")
            return False
        
        learned_settings = self.learning_data.get('camera_settings', {})
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if 'exposure' in learned_settings:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, learned_settings['exposure'])
        
        print(f"📷 Camera initialized with learned optimizations")
        return True

    def detect_motion_with_learning(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        brightness, contrast = self.analyze_lighting_conditions(frame)
        
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return False, [], 0.0
        
        diff = cv2.absdiff(self.prev_frame, gray)
        
        _, thresh = cv2.threshold(diff, self.current_sensitivity, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        
        motion_pixels = cv2.countNonZero(thresh)
        motion_percent = motion_pixels / self.frame_area
        
        self.motion_history.append(motion_percent)
        
        motion_detected = motion_percent > self.motion_threshold and motion_pixels > 50
        
        motion_boxes = []
        if motion_detected:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 15 and h > 15:
                        motion_boxes.append((x, y, w, h))
        
        confidence = self.calculate_motion_confidence(motion_boxes, motion_percent, brightness)
        
        if motion_detected:
            self.learn_from_detection(motion_boxes, motion_percent, confidence)
        
        if self.frame_count % 3 == 0:
            self.prev_frame = gray.copy()
        
        final_motion_detected = motion_detected and confidence > self.confidence_threshold
        
        return final_motion_detected, motion_boxes, confidence

    def save_photo_with_learning(self, frame, confidence):
        current_time = time.time()
        
        if current_time - self.last_photo_time < self.photo_interval:
            return False
        
        if self.frame_width <= 160:
            save_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            scale_factor = 4
        else:
            save_frame = frame.copy()
            scale_factor = 1
        
        for i, (x, y, w, h) in enumerate(self.motion_boxes):
            sx, sy, sw, sh = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
            
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
            cv2.rectangle(save_frame, (sx, sy), (sx + sw, sy + sh), color, 2)
            
            label = f"Obj{i+1} ({confidence:.2f})"
            cv2.putText(save_frame, label, (sx, sy - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            f"Seq:{self.motion_sequence} | Conf:{confidence:.2f} | Sens:{self.current_sensitivity:.0f}",
            f"Accuracy:{self.learning_data['accuracy']:.1f}% | {timestamp_str}"
        ]
        
        y_offset = 20
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(save_frame, (10, y_offset - 15), (20 + text_size[0], y_offset + 5), (0, 0, 0), -1)
            cv2.putText(save_frame, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{self.motion_sequence:03d}_{timestamp}_c{confidence:.2f}.jpg"
        filepath = os.path.join(self.current_detection_dir, filename)
        
        cv2.imwrite(filepath, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        self.last_photo_time = current_time
        self.photo_count += 1
        
        print(f"📸 {filename} | Confidence: {confidence:.2f}")
        return True

    def run(self):
        if not self.initialize_camera():
            return False
        
        self.running = True
        print("🧠 Learning motion detection active - Press Ctrl+C to stop")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                self.frame_count += 1
                
                motion_detected, motion_boxes, confidence = self.detect_motion_with_learning(frame)
                self.motion_boxes = motion_boxes
                
                if motion_detected:
                    if not self.motion_detected:
                        self.motion_sequence += 1
                        self.photo_count = 0
                        self.current_detection_dir = self.get_detection_folder()
                        print(f"🚨 Motion detected! Seq {self.motion_sequence} | Conf: {confidence:.2f}")
                        print(f"📁 Saving to: {self.current_detection_dir}")
                    
                    self.motion_detected = True
                    self.last_motion_time = current_time
                    self.save_photo_with_learning(frame, confidence)
                    
                else:
                    if (self.motion_detected and 
                        current_time - self.last_motion_time > self.motion_timeout):
                        print(f"✅ Sequence {self.motion_sequence} complete")
                        self.motion_detected = False
                
                if current_time - self.last_save_time > 60:
                    self.save_learning_progress()
                    self.last_save_time = current_time
                
                if self.frame_count % (self.fps * 30) == 0:
                    elapsed = current_time - self.start_time
                    fps_actual = self.frame_count / elapsed
                    status = "🔴 RECORDING" if self.motion_detected else "🟢 MONITORING"
                    print(f"{status} | FPS: {fps_actual:.1f} | Sens: {self.current_sensitivity:.0f} | Acc: {self.learning_data['accuracy']:.1f}%")
                    gc.collect()
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping and saving learning progress...")
        finally:
            self.cleanup()
        
        return True

    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.save_learning_progress()
        
        self.prev_frame = None
        self.motion_history.clear()
        self.lighting_conditions.clear()
        self.motion_confidence_history.clear()
        self.current_detection_dir = None
        gc.collect()
        
        elapsed = time.time() - self.start_time
        print(f"\n📊 Final Learning Summary:")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   Total detections: {self.learning_data['total_detections']}")
        print(f"   Confirmed detections: {self.learning_data['confirmed_detections']}")
        print(f"   False positives: {self.learning_data['false_positives']}")
        print(f"   Final accuracy: {self.learning_data['accuracy']:.1f}%")
        print(f"   Final sensitivity: {self.current_sensitivity:.1f}")
        print(f"   Learning sessions: {self.learning_data['sessions']}")
        print(f"   Learning data saved to: {self.learning_dir}")
        
        folder_stats = self.get_folder_stats()
        if folder_stats:
            print(f"\n📁 Folder Statistics:")
            for date, stats in folder_stats.items():
                print(f"   {date}: {stats['detections']} detections, {stats['photos']} photos")


def signal_handler(sig, frame):
    print("\n⚠️ Interrupt received...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Self-Learning Motion Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--sensitivity", type=int, default=25, help="Initial sensitivity")
    parser.add_argument("--photo-interval", type=float, default=1.5, help="Photo interval")
    parser.add_argument("--motion-timeout", type=float, default=3.0, help="Motion timeout")
    parser.add_argument("--resolution", type=str, default="160x120", help="Resolution")
    parser.add_argument("--fps", type=int, default=3, help="FPS")
    parser.add_argument("--reset-learning", action="store_true", help="Reset learning data")
    parser.add_argument("--show-stats", action="store_true", help="Show learning stats")
    parser.add_argument("--show-folders", action="store_true", help="Show folder statistics")
    parser.add_argument("--test", action="store_true", help="Test camera")

    args = parser.parse_args()

    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print("❌ Invalid resolution format")
        return 1

    if args.reset_learning:
        learning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning_data")
        for file in ['learning_progress.json', 'motion_patterns.pkl', 'detection_stats.json']:
            filepath = os.path.join(learning_dir, file)
            if os.path.exists(filepath):
                os.remove(filepath)
        print("🧹 Learning data reset")
        return 0

    if args.show_stats:
        detector = LearningMotionDetector(resolution=resolution)
        print(f"\n📊 Learning Statistics:")
        print(f"   Sessions: {detector.learning_data['sessions']}")
        print(f"   Total detections: {detector.learning_data['total_detections']}")
        print(f"   Accuracy: {detector.learning_data['accuracy']:.1f}%")
        print(f"   Confirmed patterns: {len(detector.motion_patterns['confirmed_patterns'])}")
        return 0

    if args.show_folders:
        detector = LearningMotionDetector(resolution=resolution)
        folder_stats = detector.get_folder_stats()
        if folder_stats:
            print(f"\n📁 Folder Statistics:")
            total_detections = 0
            total_photos = 0
            for date, stats in sorted(folder_stats.items()):
                print(f"   {date}: {stats['detections']} detections, {stats['photos']} photos")
                total_detections += stats['detections']
                total_photos += stats['photos']
            print(f"\n   Total: {total_detections} detections, {total_photos} photos across {len(folder_stats)} days")
        else:
            print("📁 No detection folders found")
        return 0

    if args.test:
        detector = LearningMotionDetector(camera_id=args.camera, resolution=resolution)
        if detector.initialize_camera():
            print("✅ Camera test passed!")
            detector.cleanup()
            return 0
        else:
            print("❌ Camera test failed!")
            return 1

    signal.signal(signal.SIGINT, signal_handler)

    detector = LearningMotionDetector(
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
