from ultralytics import YOLO
import cv2
import math

# Initialize the camera and set the image resolution
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)

# Load our model
model = YOLO('yolo-Weights/yolov8n.pt')

# Define the classes we are going to detect
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Capture Loop
while True:
    success, img = cap.read()  # Read the image from the camera
    results = model(img, stream=True)  # Send the image to YOLO for detection

    # Loop over detected objects
    for r in results:
        boxes = r.boxes  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to int
            
            # Detect the class name of the object
            cls = int(box.cls[0])
            classname = classNames[cls]
            
            # Calculate the confidence of the detected object
            confidence = math.ceil(box.conf[0] * 100) 
            print(f'{classname}: {confidence}%')

            # Draw the bounding box from the image
            color = (0, 255, 0)  # Default color: green
            if classname == 'cat':
                color = (128, 0, 128)  # Purple color for cats

            # Draw the rectangle on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Write the class name and confidence on the image
            org = (x1, y1 - 10)  # Position the text above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{classname} {confidence:.2f}%", org, font, 0.5, color, 2)

    # Create a window to display the image
    cv2.imshow("Webcam", img)

    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
