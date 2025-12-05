import cv2
import numpy as np
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

class PersonDetector:
    def __init__(self):
        """Initialize the person detector with YOLO model."""
        # Load YOLO model for person detection
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        
        # Load class names
        with open('coco.names', 'r') as f:
            self.classes = f.read().strip().split('\n')
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Detection parameters
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4   # Non-maximum suppression threshold
        
    def detect_people(self, frame):
        """Detect people in a frame and return bounding boxes."""
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class_id = 0 in COCO dataset)
                if class_id == 0 and confidence > self.conf_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
        
        return final_boxes, confidences[:len(final_boxes)]
    
    def draw_detections(self, frame, boxes, confidences):
        """Draw bounding boxes on the frame."""
        for i, box in enumerate(boxes):
            x, y, w, h = box
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f'Person {confidences[i]:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - 20), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display person count
        count_text = f'People detected: {len(boxes)}'
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

class GoogleSheetsLogger:
    def __init__(self, credentials_path, sheet_name):
        """Initialize Google Sheets connection."""
        try:
            # Define the scope
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Authenticate using service account
            creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
            self.client = gspread.authorize(creds)
            
            # Open the spreadsheet
            self.sheet = self.client.open(sheet_name).sheet1
            
            # Initialize headers if sheet is empty
            if self.sheet.row_count == 0 or self.sheet.cell(1, 1).value is None:
                self.sheet.append_row(['Timestamp', 'Date', 'Time', 'Person Count', 'Change'])
                print("Initialized Google Sheet with headers")
            
            print("Google Sheets connected successfully")
            
        except Exception as e:
            print(f"Error initializing Google Sheets: {e}")
            raise
    
    def log_person_count(self, person_count, previous_count):
        """Log person count to Google Sheets."""
        try:
            # Get current timestamp
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")
            
            # Calculate change
            change = person_count - previous_count
            change_str = f"+{change}" if change > 0 else str(change)
            
            # Append row to sheet
            row = [timestamp, date, time, person_count, change_str]
            self.sheet.append_row(row)
            
            print(f"✓ Logged to Google Sheets: {person_count} people ({change_str})")
            return True
            
        except Exception as e:
            print(f"Error logging to Google Sheets: {e}")
            return False

def main():
    """Main function to run person detection from camera."""
    print("Person Detection Application with Google Sheets Logging")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit application")
    print("  's' - Save screenshot locally")
    print("=" * 60)
    
    # Google Sheets configuration
    CREDENTIALS_PATH = 'google-credentials.json'  # Path to your Google service account credentials
    SHEET_NAME = 'Person Detection Log'           # Name of your Google Sheet
    
    # Initialize detector
    try:
        detector = PersonDetector()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease download the YOLO files:")
        print("1. yolov3.weights from https://pjreddie.com/media/files/yolov3.weights")
        print("2. yolov3.cfg from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg")
        print("3. coco.names from https://github.com/pjreddie/darknet/blob/master/data/coco.names")
        return
    
    # Initialize Google Sheets logger
    try:
        logger = GoogleSheetsLogger(CREDENTIALS_PATH, SHEET_NAME)
    except Exception as e:
        print(f"Google Sheets initialization failed: {e}")
        print("\nPlease ensure:")
        print("1. google-credentials.json is in the same directory")
        print("2. Your Google Sheet name is correct")
        print("3. The service account has edit access to the sheet")
        print("\nContinuing without Google Sheets logging...")
        logger = None
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera, or specify camera index
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera initialized. Starting detection...")
    
    frame_count = 0
    previous_person_count = 0
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Perform detection every frame (can be adjusted for performance)
        if frame_count % 1 == 0:  # Process every frame
            boxes, confidences = detector.detect_people(frame)
            current_person_count = len(boxes)
            frame = detector.draw_detections(frame, boxes, confidences)
            
            # Check if person count changed
            if current_person_count != previous_person_count:
                print(f"Person count changed: {previous_person_count} → {current_person_count}")
                
                # Log to Google Sheets if available
                if logger:
                    logger.log_person_count(current_person_count, previous_person_count)
                
                previous_person_count = current_person_count
        
        # Display frame
        cv2.imshow('Person Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved locally: {filename}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()