import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.mnist.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for i in range(16):
    plt.subplot(4,4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])

plt.show()
