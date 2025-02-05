import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Object Detection")
        self.setGeometry(150, 150, 900, 650)
        
        # Load trained YOLO model
        self.detector = YOLO("model.pt")  
        self.processing_device = "cpu"  
        
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout()
    
        button_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Select Video")
        self.upload_btn.setFixedSize(100, 30)  # Set button size
        self.upload_btn.setStyleSheet("background-color: lightgreen; font-size: 12px;")  # Set button color and font
    
        self.upload_btn.clicked.connect(self.select_video)
        button_layout.addWidget(self.upload_btn)
    
        layout.addLayout(button_layout)
        self.setLayout(layout)

    
    def select_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Choose Video File", "", "Videos (*.mp4 *.avi)")
        if video_path:
            self.process_video(video_path)
    
    def process_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            
            height, width = frame.shape[:2]
            resized_frame = cv2.resize(frame, (640, 640))
            
            detection_results = self.detector(resized_frame, device=self.processing_device)
            
            for detection in detection_results:
                for bounding_box in detection.boxes:
                    x1, y1, x2, y2 = bounding_box.xyxy[0].tolist()
                    confidence = bounding_box.conf[0].item()
                    category_id = int(bounding_box.cls[0].item())
                    object_label = detection.names[category_id]
                    
                    # Rescale coordinates
                    x1 = int(x1 * width / 640)
                    y1 = int(y1 * height / 640)
                    x2 = int(x2 * width / 640)
                    y2 = int(y2 * height / 640)
                    
                    # Draw detection results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{object_label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    application = QApplication(sys.argv)
    main_window = ObjectDetectionApp()
    main_window.show()
    sys.exit(application.exec())
