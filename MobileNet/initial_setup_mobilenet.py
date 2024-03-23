import numpy as np
import cv2

# Path to the prototxt file with text description of the network architecture
prototxt = "MobileNetSSD_deploy.prototxt"
# Path to the .caffemodel file with learned network
caffe_model = "MobileNetSSD_deploy.caffemodel"

# Read a network model (pre-trained) stored in Caffe framework's format
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# Dictionary with the object class id and names on which the model is trained
classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

cap = cv2.VideoCapture("class.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Size of image
    height, width = frame.shape[:2]
    
    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    
    # Blob object is passed as input to the object detection network
    net.setInput(blob)
    
    # Network prediction
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        # Confidence of prediction
        confidence = detections[0, 0, i, 2]
        
        # Set confidence level threshold to filter weak predictions
        if confidence > 0.5:
            # Get class id
            class_id = int(detections[0, 0, i, 1])
            
            # Scale to the frame
            x_top_left = int(detections[0, 0, i, 3] * width)
            y_top_left = int(detections[0, 0, i, 4] * height)
            x_bottom_right = int(detections[0, 0, i, 5] * width)
            y_bottom_right = int(detections[0, 0, i, 6] * height)
            
            # Draw bounding box around the detected object
            cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)
            
            if class_id in classNames:
                # Get class label
                label = classNames[class_id] + ": " + str(confidence)
                
                # Draw label text on the frame
                cv2.putText(frame, label, (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("frame", frame)
    
    # Break the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
