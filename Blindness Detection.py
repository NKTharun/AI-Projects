import cv2
import numpy as np

# Load the image
image = cv2.imread("eye_image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If there are no contours, the image might be considered as blindness
if len(contours) == 0:
    print("Blindness detected.")
else:
    # If there are contours, the eyes are likely present
    print("Eyes detected.")

# Display the original image with contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Blindness Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
