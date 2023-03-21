#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:51:29 2022

@author: pcunha
"""

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

import numpy as np

tf.random.set_seed(1234)

# Set PATH for the train and validation directories
train_dir = "./train/"
validation_dir = "./validation/"

# View random image from the selected class
def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img

# Ouptut a image from the chosen class
img_g = view_random_image(target_dir=train_dir,target_class="CLASS_OF_INTEREST")

# Batch size
BS = 32

# Generates the training data. Here three steps are made: rescaling of the images,rotation (90 degrees in this case), horizontal flip
# 
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1.0/255,
                    rotation_range=90,
                    horizontal_flip=True,
                    )

validation_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(299,299),
                                                    batch_size=BS,
                                                    shuffle=False)

test_generator = validation_datagen.flow_from_directory(validation_dir,
                                                         target_size=(299,299),
                                                         batch_size=BS,
                                                         shuffle=False)

# Data size of the images. This should be the same you use as input_shape in your model!
image_size = (299, 299) #CHANGE THIS PLEASE

#Number of classes being used
nr_classes = 2 

# BASELINE MODEL

model_0 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(299, 299, 3)),
    tf.keras.layers.Dense(128, activation="relu"), # binary activation output
    tf.keras.layers.Dense(nr_classes, activation="sigmoid") # binary activation output
])

model_0.summary()

# If you want to have the plot model, run this
tf.keras.utils.plot_model(
    model_0,
    to_file="model_0.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=120,
)

# Compile the model
model_0.compile(loss="binary_crossentropy", # for binary classification
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

epochs=20 #Change the number of epochs as needed

# Fit the model
history_0 = model_0.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    verbose=1)

# Plot the loss and accuracy. This plots are going to be important for the analysis of the models
plt.plot(history_0.history["loss"], label="train_loss")
plt.plot(history_0.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history_0.history['accuracy'], label='accuracy')
plt.plot(history_0.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Evaluate on Validation data. Here you have two ways of getting the same evaluation score
scores = model_0.evaluate(test_generator)
print("%s%s: %.2f%%" % ("evaluate ",model_0.metrics_names[1], scores[1]*100))

scores = model_0.evaluate_generator(test_generator)
print("%s%s: %.2f%%" % ("evaluate_generator ",model_0.metrics_names[1], scores[1]*100))



# CNN model

model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10, 
                         kernel_size=3, # can also be (3, 3)
                         activation="relu", 
                         input_shape=(299, 299, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(nr_classes, activation="sigmoid") # binary activation output
])

model_1.summary()

# If you want to plot the model
tf.keras.utils.plot_model(
    model_1,
    to_file="model_1.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=120,
)

# Compile the model
model_1.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

epochs=20

# Fit the model
# Here you have a Reduce Learning Rate on Plateau (this helps the model not to overfit, i.e be very good for the training data and not able to generalise in the validation data)
# I set the metric to monitor, in this case the validation loss, the factor, the patience in epochs and the minimum learning rate
# You can also set early stop conditions. This preventes the model to run epochs that are not helping the model. 
# It can be set either in the validation loss or in the validation accuracy, which in this case you want the maximum
history_1 = model_1.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    verbose=1,
                    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001), 
                                               EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=1,restore_best_weights=True),
                                               #EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1,restore_best_weights=True                                               ])
                                               ])
                    
plt.plot(history_1.history["loss"], label="train_loss")
plt.plot(history_1.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history_1.history['accuracy'], label='accuracy')
plt.plot(history_1.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Evaluate on Validation data
scores = model_1.evaluate(test_generator)
print("%s%s: %.2f%%" % ("evaluate ",model_1.metrics_names[1], scores[1]*100))

scores = model_1.evaluate_generator(test_generator)
print("%s%s: %.2f%%" % ("evaluate_generator ",model_1.metrics_names[1], scores[1]*100))

# Save model using HDF5 
#model_1.save('cnn_gqs.h5')

# Recreate the exact same model, including its weights and the optimizer
#new_model = tf.keras.models.load_model('cnn_gqs.h5')

# Show the model architecture
#new_model.summary()

scores = model_1.evaluate(test_generator)
print("%s%s: %.2f%%" % ("evaluate saved model ",model_1.metrics_names[1], scores[1]*100))

# To get the model output probabilities

probability_model = tf.keras.Sequential([model_1, tf.keras.layers.Softmax()]) # convert model's linear outputs to probabilities

predictions = probability_model.predict(test_generator)

class_names = ['CLASS1', 'CLASS2']

def plot_image(i, predictions_array, data):
  img, label = data.next()
  true_label = data.classes
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[i])

  predicted_label = np.argmax(predictions_array)
  
  if predicted_label == true_label[i]:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label[i]]),
                                color=color)

def plot_value_array(i, predictions_array, data):
  true_label = data.classes[i]
  plt.grid(False)
  plt.xticks(range(3))
  plt.yticks([])
  thisplot = plt.bar(range(3), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_generator)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_generator)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_generator)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_generator)
plt.tight_layout()
plt.show()

# You can also create on top of other models! In this example, it will create on top of XCEPTION.
# This is a well known algorithms in computer vision, but is also very complex. Sometimes, complex models do not work well with images with less resolution.
# Take this as an example for future projects that you may want to do.
# Create on top of Xception

# Always admit the images as RGB so 3 channels
conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))
conv_base.trainable = False

# Add Xception to a new model
model = tf.keras.models.Sequential()
model.add(conv_base)

# Add new layers on top of the Xception one
#In this case, you have a flat layer, a dense layer with 32 nodes and a sigmoid outpout layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(nr_classes, activation='sigmoid'))

# ver o modelo e confirmar
model.summary()

#If you want to plot the model
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=120,
)


model.compile(loss='binary_crossentropy', # for binary classification
              optimizer='adam',
              metrics=['accuracy'])

# treinar o modelo
epochs = 20
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    verbose=1)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Evaluate on Validation data
scores = model.evaluate(test_generator)
print("%s%s: %.2f%%" % ("evaluate ",model.metrics_names[1], scores[1]*100))
