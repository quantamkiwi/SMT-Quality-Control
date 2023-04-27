import cv2
import numpy as np 

img = cv2.imread('plain_pcb.jpg')

img = cv2.resize(img, (int(img.shape[1]*0.7), int(img.shape[0]*0.7)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)

thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 2000:  # Set a lower bound on the elipse area.
        continue

    if len(contour) < 5:  # The fitEllipse function requires at least five points on the contour to function.
        continue

    # cv2.drawContours(img, contours, -1, (0,0,255), 1)
    ellipse = cv2.fitEllipse(contour)  # Fit an ellipse to the points in the contour.
    cv2.ellipse(img, ellipse, (0,255,0), 2)  # Draw the ellipse on the original image.

    # Use the moments of the contour to draw a dot at the centroid of each coin.
    moments = cv2.moments(contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)

cv2.imshow('Gray', gray)
cv2.imshow('After Adaptive Threshold', thresh)
cv2.imshow('After Morphological Operations', closing)
cv2.imshow('After Contour Detection', img)

while cv2.waitKey(1) < 0:
    pass

cv2.destroyAllWindows
