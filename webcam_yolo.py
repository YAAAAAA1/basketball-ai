"""
Real-time webcam basketball detection using YOLOv8
"""

import cv2
from ultralytics import YOLO
import json
import time

class WebcamDribbleDetector:
    def __init__(self, model_path='Basketball_Dribbles_Count_Using_YOLOv8/yolov8l.pt'):
        """Initialize the detector with YOLOv8 model"""
        self.model = YOLO(model_path)
        
        # Optimize for speed - use half precision if GPU available
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("Using GPU acceleration!")
        
        # Warm up the model
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy_frame, verbose=False, imgsz=416)
        
        # Tracking state
        self.dribble_count = 0
        self.ball_positions = []
        self.ball_state = "UP"
        self.ball_detections = 0
        self.person_detections = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame + stats"""
        
        # Run YOLO detection with optimizations
        results = self.model(frame, verbose=False, imgsz=416, conf=0.25, iou=0.45)
        
        # Process detections
        ball_detected = False
        person_detected = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Class 32 is sports ball, 0 is person in COCO
                if cls == 32 and conf > 0.3:  # Sports ball with lower threshold
                    ball_detected = True
                    self.ball_detections += 1
                    
                    ball_y = (y1 + y2) / 2
                    ball_x = (x1 + x2) / 2
                    
                    self.ball_positions.append({'y': ball_y, 'x': ball_x})
                    
                    # Keep only recent positions
                    if len(self.ball_positions) > 10:
                        self.ball_positions.pop(0)
                    
                    # Draw ball
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame, f'Ball {conf:.2f}', (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                elif cls == 0 and conf > 0.5:  # Person
                    person_detected = True
                    self.person_detections += 1
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Dribble detection
        if len(self.ball_positions) >= 3:
            current_y = self.ball_positions[-1]['y']
            prev_y = self.ball_positions[-2]['y']
            
            # Detect downward motion
            if current_y > prev_y + 8 and self.ball_state == "UP":
                self.ball_state = "DOWN"
            
            # Detect upward motion (bounce)
            elif current_y < prev_y - 8 and self.ball_state == "DOWN":
                self.ball_state = "UP"
                self.dribble_count += 1
        
        # Draw dribble count
        cv2.putText(frame, f'Dribbles: {self.dribble_count}', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = int(self.frame_count / elapsed)
            self.frame_count = 0
            self.start_time = time.time()
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {self.fps}', (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame, {
            'dribbles': self.dribble_count,
            'ball_detections': self.ball_detections,
            'person_detections': self.person_detections,
            'fps': self.fps
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.dribble_count = 0
        self.ball_positions = []
        self.ball_state = "UP"
        self.ball_detections = 0
        self.person_detections = 0
        self.frame_count = 0
        self.start_time = time.time()


def generate_frames(camera_id=0):
    """Generator function for video streaming"""
    detector = WebcamDribbleDetector()
    cap = cv2.VideoCapture(camera_id)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    # JPEG encoding optimization
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Balanced quality/speed
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            annotated_frame, stats = detector.process_frame(frame)
            
            # Encode frame as JPEG with optimization
            ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        cap.release()


if __name__ == '__main__':
    # Test the detector
    detector = WebcamDribbleDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, stats = detector.process_frame(frame)
        
        cv2.imshow('YOLOv8 Basketball Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()