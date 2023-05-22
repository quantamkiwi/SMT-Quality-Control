""" 
roi_classification.py

Description: 
    - Initializes Convolutional Neural Network Model.
    - Grabs regions of interest and their classifications from files.
    - Can split the regions of interest into test and training data with a ratio.
    - Trains the model.
    - Tests the model.
    - Or can make a prediction from the trained model with a full image.

Author: Bailey Smith

Last Modified: 17/05/2023
"""

import numpy as np 
import cv2 
import pandas
import tensorflow as tf
from tensorflow import keras

# Change for training and testing the model at your own discretion.
TEST_TRAIN_RATIO = 3


def init_model(input_shape, summary=True):
    """ 
    Initialises the model to be trained and evaluated.
    """
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(
        16, (4, 4), 1, activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D())
    
    model.add(keras.layers.Conv2D(
        32, (3, 3), 1, activation="relu")) 
    model.add(keras.layers.MaxPooling2D())
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(128, activation="relu"))
    
    model.add(keras.layers.Dense(64, activation="relu"))
    
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    if summary:
        # Printing a summary of the model to the console.
        model.summary()
    
    return model


def allocate_log_directory():
    """ 
    Create a log directory so we can save training changes.
    """
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    return tensorboard_callback


def train_model(model, train_images, train_labels):
    """ 
    Trains the model with respect to the training images and labels.
    """
    tensorboard_callback = allocate_log_directory()

    # For saving the model
    # hist = model.fit(train_images, epochs=25, validation_data=train_labels, callbacks=[tensorboard_callback])
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    hist = model.fit(train_images, train_labels, epochs=25, verbose=1)
    
    return hist, model
    
    
def test_model(model, test_images, test_labels):
    """ 
    Evaluates the trained model with respect to the test images and labels.
    """

    # Format arrays.
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Evaluate model.
    loss, accuracy, precision, recall = model.evaluate(test_images, test_labels, verbose=1)

    # Calculate F1 score.
    f1 = 2 * ((precision * recall) / (precision + recall))

    # Print results.
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test f1-score: {f1}")

    return loss, accuracy, precision, recall, f1

def retrieve_training_images():
    """ 
    Goes through the file locations and retrieves the training images. 
    """
    roi_number = 0 
    train_images = []
    
    image = cv2.imread('./Train Images/regions of interest/ROI_{}.png'.format(roi_number))

    while image is not None: 
        train_images.append(image)
        roi_number += 1
        image = cv2.imread('./Train Images/regions of interest/ROI_{}.png'.format(roi_number))
    
    return train_images

        
def retrieve_training_labels():
    """ 
    Reads the train_labels.csv file to find the appropriate labels in the appropriate 
    order.
    """
    return pandas.read_csv('./train_labels.csv', delimiter=',', usecols=[1])


def delete_small_images(images, dimensions):
    """ 
    Goes through all of the regions of interest and eliminates the smallest ones 
    from the array. 
    """
    i = 0
    for image in images: 
        if image.shape[0] < 10 or image.shape[1] < 10:
            images.pop(i)
            dimensions.pop(i)
        i += 1
    return images, dimensions



def determine_smallest_dimensions(train_images):
    """ 
    Iterates through images and returns the smallest height and width dimensions found.
    """
    min_height = 1000
    min_width = 1000
    
    for image in train_images:
        height, width, rgb = image.shape 
        
        if height < min_height: 
            min_height = height
        if width < min_width:
            min_width = width
    
    return min_height, min_width 
        


def equalize_image_dimensions(train_images):
    """ 
    Resizes the images so that all of the images have the same dimensions for the model.
    """

    min_height, min_width = determine_smallest_dimensions(train_images)

    i = 0
    while i < len(train_images):
        train_images[i] = cv2.resize(train_images[i], (min_height, min_width)) / 255
        i += 1 
    
    
    return train_images, (min_width, min_height, 3)


    
def seperate(items, dilution):
    """ 
    For seperating testing and training data.
    """
    train = []
    test = []
    i = 0

    # Seperate data into test and train arrays.
    for item in items:
        if i % dilution == 0:
            test.append(item)
        else: 
            train.append(item)
        i += 1
    
    return train, test

def predict(filename, model, input_shape, display=False):
    """ 
    Takes an image as an input and returns a result of the different rois.
    """
    from roi_identification import bounding_boxes, image_processing, scale_for_screen

    # Get image.
    image = cv2.imread(filename)

    # Process Image.
    dilation, diff_rg, norm_diff_rg = image_processing(image, dilation_iterations=18)

    # Get regions of interest.
    imagebox, rois, dimensions = bounding_boxes(dilation, image, predict=True)

    # Delete tiny regions of interest.
    rois, dimensions = delete_small_images(rois, dimensions)

    # Resize all of the regions of interest to the training size.
    i = 0
    while i < (len(rois) - 1):
        rois[i] = cv2.resize(rois[i], (input_shape[1], input_shape[0])) / 255
        i += 1 

    # Pop the last roi (its inconsequential).
    rois.pop(-1)

    # For each roi make a prediction.
    predictions = []
    for roi in rois:
        roi = np.expand_dims(np.array(roi), axis=0)
        predictions.append(
            model.predict(roi)
            )
    
    if display:
        display_predictions(image, rois, predictions, dimensions)
    
    return predictions


def display_predictions(image, rois, predictions, dimensions):
    """ 
    Displays the predictions back to the operator.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)

    # Draw boxes around each roi detailing the prediction.
    i = 0
    for prediction in predictions:
        if prediction > 0.5:
            cv2.rectangle(image, 
                          dimensions[i][0], 
                          dimensions[i][1], 
                          green, 
                          2)
            cv2.putText(image, 
                        f'{prediction[0][0]:.2f}',
                        (dimensions[i][0][0] + 5, dimensions[i][0][1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        green, 
                        thickness=2)
        else:
            cv2.rectangle(image, 
                          dimensions[i][0], 
                          dimensions[i][1], 
                          red, 
                          2)
            cv2.putText(image, 
                        f'{prediction[0][0]:.2f}',
                        (dimensions[i][0][0] + 5, dimensions[i][0][1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        red, 
                        thickness=2)
        i += 1
    
    # Scale the image for the screen (may need to be changed if too small).
    image = cv2.resize(image, (int(image.shape[1]*0.2), int(image.shape[0]*0.2)))

    cv2.imshow('Labelled predictions for image', image)
    cv2.waitKey()
    


def main():
    """ 
    Main Function.
    """
    # Get regions of interest.
    images = retrieve_training_images()

    # Equalise roi dimensions.
    train_images, input_shape = equalize_image_dimensions(images)

    # Get roi labels.
    train_labels = retrieve_training_labels()

    # Split rois into testing and training arrays.
    train_images, test_images = seperate(train_images, TEST_TRAIN_RATIO)
    train_labels, test_labels = seperate(train_labels["Pass"], TEST_TRAIN_RATIO)
    
    # Initialize model. 
    model = init_model(input_shape)
    
    # Train model.
    hist, model = train_model(model, train_images, train_labels)

    # Uncomment if testing the model.
    # loss, accuracy, precision, recall, f1 = test_model(model, test_images, test_labels)
    # print(hist)

    # Predictions, comment if testing.
    predictions = predict('./Train Images/IMG_0109x.jpg', model, input_shape, display=True)

main()