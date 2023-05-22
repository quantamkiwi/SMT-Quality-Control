""" 
roi_identification.py

Description: Uses grayscale, binary thresholding, dilation and
Suzuki and Abe's contour algorithm to draw and extract regions of interest
from an image. 

Author: Bailey Smith

Last modified: 17.05.2023
"""

import cv2
import numpy as np
import os.path


DILATION_ITERATIONS = 18


def read_image(filename):
    """
    Uses OpenCV to read an image.
    """
    return cv2.imread(filename)


def scale_for_screen(image):
    """ 
    Resize image to fit in screen.
    """
    return cv2.resize(image, (int(image.shape[1]*0.6), int(image.shape[0]*0.6)))


def image_processing(image, dilation_iterations):
    """ 
    Takes the image matrix and performs the neccessary preprocessing before 
    creating bounding boxes.
    Preprocessing includes: Greyscale from normalized red-green difference and 
    morphological closing.
    """

    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Calculate the difference between the red and green channels
    diff_rg = cv2.subtract(r, g)
    # Normalize the difference image
    norm_diff_rg = cv2.normalize(diff_rg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Morpholigically open the normalized diffence image.
    kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(norm_diff_rg, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilation = cv2.dilate(norm_diff_rg, kernel, iterations=dilation_iterations)
    
    return dilation, diff_rg, norm_diff_rg


def bounding_boxes(processed_image, image, write=False, predict=False):
    """ 
    Compute the Regions of Interest and surround then with bounding boxes.
    """
    cnts = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    roi_number = find_latest_roi_number()

    rois = []
    dimensions = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w > 12 and h > 12:
            x -= 3 
            y -= 3
            w += 6
            h += 6
            
            if write:
                roi = image[y:y+h, x:x+w]
                classification = ask_user_roi_classification(roi)
                if classification: 
                    write_region_of_interest(roi, roi_number)
                    write_label_to_csv(roi_number, classification)
                    roi_number += 1
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            elif predict:
                roi = image[y:y+h, x:x+w]
                rois.append(roi)
                dimensions.append([(x, y), (x + w, y + h)])
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    if predict:
        return image, rois, dimensions
    else:
        return image 



def ask_user_roi_classification(roi):
    """ 
    Asks the user whether the region of interest is solder defect, correct soldering
    or not applicable. Returns the result.
    """
    
    if roi.shape[0] > 0 and roi.shape[1] > 0: # check if roi has a valid size
        title = f'roi: {roi.shape}'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(title, f'Press Q then Enter classification in Python Shell {title}')
        cv2.resizeWindow(title, 800, 600)
        cv2.imshow(title, roi)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow(title)
            
        try:
            classification = int(input("Is this image N/A (0), correct soldering (1) or a Defect (2): "))
        except Exception as e:
            print(e)
            classification = 0
        return classification
    else:
        return 0
    


def write_label_to_csv(roi_number, classification):
    """ 
    Writes the classification of the image to the csv file.
    """
    with open("train_labels.csv", 'a') as file:
        file.write(f'{roi_number},{0 if classification == 2 else classification}\n')
        

def write_region_of_interest(roi, roi_number):
    """ 
    Writes the region of interest to the files.
    """
    cv2.imwrite('./Train Images/regions of interest/ROI_{}.png'.format(roi_number), roi)
    
def find_latest_roi_number():
    """ 
    Goes through the roi file and finds the latest roi number
    to set the roi to
    """
    roi_number = 0
    
    while os.path.isfile('./Train Images/regions of interest/ROI_{}.png'.format(roi_number)):
        roi_number += 1 
    return roi_number


def mainBB():
    """
    Main function for this file.
    """
    image = read_image('./Train Images/IMG_0123.jpg')
    # image = scale_for_screen(image)
    processed_image, diff_rg, norm_diff_rg = image_processing(image, DILATION_ITERATIONS)
    imageBB = bounding_boxes(processed_image, image, write=False)
    imageBB = scale_for_screen(imageBB)
    # cv2.imshow('image', image)
    # cv2.imshow('redgreen', diff_rg)
    # cv2.imshow('thresh', norm_diff_rg)
    
    # processed_image = scale_for_screen(processed_image)
    cv2.imshow('dilated', processed_image)
    cv2.imshow('imageBB', imageBB)
    cv2.waitKey() 


    
