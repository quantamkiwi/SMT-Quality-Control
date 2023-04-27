import cv2
import numpy as np

# Load the image
image = cv2.imread('plain_pcb.jpg')
image = cv2.resize(image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)))

# Split the image into its color channels
b, g, r = cv2.split(image)

# Calculate the difference between the red and green channels
diff_rg = cv2.subtract(r, g)

# Normalize the difference image
norm_diff_rg = cv2.normalize(diff_rg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(norm_diff_rg,kernel,iterations = 1)
closing = cv2.morphologyEx(norm_diff_rg, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(norm_diff_rg, cv2.MORPH_OPEN, kernel)

# Display the resulting grayscale image
cv2.imshow('Grayscale Image', norm_diff_rg)
# cv2.imshow('Closing', closing)
# cv2.imshow('Erosion', erosion)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()