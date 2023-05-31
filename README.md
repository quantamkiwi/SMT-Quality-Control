# SMT Soldering Quality Control README

Author: Bailey Smith

Last Modified: 22.05.2023

## Description

This is a two-stage program that takes a stereo image input and outputs the regions of interest classified as soldered or not.

### First Stage

**roi_identification.py**

This file identifies the regions of interest using various techniques:

1. Grayscaling by calculating the difference between the red and green RGB channels.
2. The result from step 1 will leave negative values. Binary thresholding will turn the image binary and normalize all values to 0 or 1.
3. Dilation of the image, iteration specified by the user.
4. Map rectangles to contours found using Suzuki and Abe's contour finding algorithm.
5. Output regions of interest.

**Key Tips**

- The global variable at the top `DILATION_ITERATIONS` will control how dilated the image will be before the contour following algorithm is applied. This value needs to be manually calibrated to your input image.
- Calling `mainBB()` and uncommenting/commenting lines at your own discretion can be used to collect more regions of interest for training/testing or for looking at the output after the dilation (to get the right amount of iterations).
- Changing `write=True` in `mainBB` will write the regions of interest to the files (`train_labels.csv` and `Train Images/regions of interest/ROI_X`). The program will prompt you to classify each region of interest in the terminal.

### Second Stage

**roi_classification.py**

This file classifies regions of interest on an image as soldered or not. The `main()` at the bottom of the file controls whether you test and train the neural network or you use it to predict the result of an image.

**Key Tips**

- To predict an image, `mainBB` must not be called in `roi_identification.py`.
- In `roi_classification.py`, the global variable at the top `TEST_TRAIN_RATIO` will determine the ratio between testing and training images when gathering the regions of interest.
- Neural Network does not currently save. Must train before predicting.
- CNN architecture used currently is shown in a diagram saved as `CNN.png`.
- Images are in the `Train Images` folder. However, the simplest example is at `report_images/IMG0109x.jpg`.

## Requirements 
- TensorFlow
- OpenCV
- NumPy
- Pandas