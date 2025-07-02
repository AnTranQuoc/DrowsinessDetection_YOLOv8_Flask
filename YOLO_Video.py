from ultralytics import YOLO
import cv2
import math
import pygame

# Initialize alarm sound
pygame.mixer.init()
pygame.mixer.music.load("static/files/alarm.mp3")

def video_detection(path_x):
    
    # Create a Webcam Object
    cap = cv2.VideoCapture(path_x)
    
    # Load YOLO model to GPU
    model = YOLO("YOLO-Weights/best.pt").to('cuda')
    classNames = ["Buon_Ngu", "Tinh_Tao"]

    drowsy_count = 0
    DROWSY_THRESHOLD = 15  # Number of consecutive frames indicating drowsiness

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        detected_drowsy = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Get class information
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # Set color based on class
                color = (0, 255, 0)
                if class_name == "Buon_Ngu":
                    color = (0, 0, 255)
                    detected_drowsy = True

                # Draw bounding box and label
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], 2, cv2.LINE_AA)

        # Check for consecutive drowsy frames
        if detected_drowsy:
            drowsy_count += 1
        else:
            drowsy_count = 0

        if drowsy_count >= DROWSY_THRESHOLD:
            cv2.putText(img, "WARNING: Buon ngu!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

        yield img
