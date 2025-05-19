# RetailShelfInventoryMonitoring

from ultralytics import YOLO

import cv2

\# Load the YOLO model

model = YOLO('yolov8n.pt')

\# Path to the image to process

image\_path = 'shelf\_or\_wall.jpg'

\# Load the image using OpenCV

image = cv2.imread(image\_path)

\# Perform object detection

results = model(image)\[0\]

\# Count the number of detected objects

object\_count = len(results.boxes)

\# Draw bounding boxes on the image for each detected object

for box in results.boxes:

x1, y1, x2, y2 = map(int, box.xyxy\[0\])

cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

\# Add text displaying the object count

cv2.putText(image, f'Objects: {object\_count}', (30, 40),

cv2.FONT\_HERSHEY\_SIMPLEX, 1, (255, 0, 0), 2)

\# Display the image with detections

cv2.imshow("Wall Detection Count", image)

\# Wait for a key press before closing the window

cv2.waitKey(0)

cv2.destroyAllWindows()

\# Print the total number of objects detected

print(f"Total objects detected: {object\_count}")
