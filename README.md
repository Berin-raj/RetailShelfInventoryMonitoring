# RetailShelfInventoryMonitoring
Python Script: YOLO Object Detection with OpenCV
------------------------------------------------

This Python script performs object detection on an image using a pre-trained YOLO (You Only Look Once) model, specifically yolov8n.pt, from the ultralytics library. It processes the image to detect objects, draw bounding boxes around them, and display the total count of detected objects.

### Step-by-step Description:

1.  **Importing Required Libraries:**
    
    *   from ultralytics import YOLO: This imports the YOLO model from the ultralytics library, allowing the use of the pre-trained YOLO object detection model.
        
    *   import cv2: OpenCV is used for image processing and visualization. It is responsible for loading the image, drawing bounding boxes, and displaying the image.
        
2.  pythonCopyEditmodel = YOLO('yolov8n.pt')The YOLO model is loaded with a pre-trained weights file (yolov8n.pt). This file contains the trained parameters of the model, allowing it to detect a wide variety of objects.
    
3.  pythonCopyEditimage\_path = 'shelf\_or\_wall.jpg'The path to the image file you want to process is specified. In this case, the image is called shelf\_or\_wall.jpg.
    
4.  pythonCopyEditimage = cv2.imread(image\_path)The image located at the image\_path is loaded into memory using OpenCV's imread function.
    
5.  pythonCopyEditresults = model(image)\[0\]The YOLO model is applied to the loaded image to perform object detection. The model(image) returns the detection results, and \[0\] is used to access the first element in the returned results, which typically contains the bounding boxes and the detected objects.
    
6.  pythonCopyEditobject\_count = len(results.boxes)The number of detected objects is counted by getting the length of the results.boxes array, which contains the coordinates of the bounding boxes around the detected objects.
    
7.  pythonCopyEditfor box in results.boxes: x1, y1, x2, y2 = map(int, box.xyxy\[0\]) cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)For each detected object, the script extracts the coordinates of the bounding box (top-left corner x1, y1 and bottom-right corner x2, y2). A green rectangle is then drawn on the image using OpenCV's rectangle function.
    
8.  pythonCopyEditcv2.putText(image, f'Objects: {object\_count}', (30, 40), cv2.FONT\_HERSHEY\_SIMPLEX, 1, (255, 0, 0), 2)The script adds text to the image, displaying the total count of detected objects. The text is placed at coordinates (30, 40) with the font FONT\_HERSHEY\_SIMPLEX, size 1, and color (255, 0, 0) (red).
    
9.  pythonCopyEditcv2.imshow("Wall Detection Count", image)The image, with bounding boxes and object count text, is displayed in a window titled "Wall Detection Count."
    
10.  pythonCopyEditcv2.waitKey(0)cv2.destroyAllWindows()The script waits indefinitely for a key press before closing the displayed window. After the key press, all OpenCV windows are closed using destroyAllWindows.
    
11.  pythonCopyEditprint(f"Total objects detected: {object\_count}")
