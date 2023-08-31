# Install required libraries if not already installed
pip install opencv-python-headless numpy

import cv2
import numpy as np

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load image
image = cv2.imread("input.jpg")

# Get the image dimensions
height, width, _ = image.shape

# Create a blob from the image and pass it through the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Initialize lists for detected class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Loop through each detection from each output layer
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Minimum confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove redundant overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[class_ids[i]]
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the output image with detected objects
cv2.imwrite("output.jpg", image)

# Display the output image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
