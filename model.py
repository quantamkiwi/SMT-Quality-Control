import numpy as np 
import cv2 
import pandas
import tensorflow as tf
from tensorflow import keras

def init_model(input_shape, summary=True):
    """ 
    Initialises the model to be trained and evaluated.
    """
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv2D(
        16, (4, 4), 1, activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D())
    
    # model.add(keras.layers.Conv2D(
    #     32, (4,4), 1, activation="gelu"))
    # model.add(keras.layers.MaxPooling2D())
    
    # model.add(keras.layers.Conv2D(
    #     16, (3,3), 1, activation="relu"))

    # model.add(keras.layers.MaxPooling2D())
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256, activation="relu"))
    
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    if summary:
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
    # hist = model.fit(train_images, epochs=10, validation_data=train_labels, callbacks=[tensorboard_callback])
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    hist = model.fit(train_images, train_labels, epochs=20, verbose=1)
    
    return hist
    
    
def test_model(model, test_images, test_labels):
    """ 
    Evaluates the trained model with respect to the test images and labels.
    """
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    return loss, accuracy

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

    for item in items:
        if i % dilution == 0:
            test.append(item)
        else: 
            train.append(item)
        i += 1
    
    return train, test

def main():
    """ 
    Main Function.
    """

    images = retrieve_training_images()
    train_images, input_shape = equalize_image_dimensions(images)
    train_labels = retrieve_training_labels()
    train_images, test_images = seperate(train_images, 4)
    train_labels, test_labels = seperate(train_labels["Pass"], 4)
    
    
    
    model = init_model(input_shape)
    
    hist = train_model(model, train_images, train_labels)
    loss, accuracy = test_model(model, test_images, test_labels)
    
    print(hist)

main()