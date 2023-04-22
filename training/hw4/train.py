#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Define image dimensions
IMG_WIDTH = 640
IMG_HEIGHT = 450

# Define file paths for dataset
not_stop_path = './CustomDataset/not_stop/*.jpg'
stop_path = './CustomDataset/stop/*.jpg'

# Define function to load and preprocess image
def load_and_preprocess_image(filepath):
    # Load image as grayscale
    img = load_img(filepath, color_mode='rgb', target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Convert image to numpy array
    img = img_to_array(img)
    # Normalize pixel values
    img = img / 255.0
    return img

# Load images and labels into lists
not_stop_images = []
stop_images = []
not_stop_labels = []
stop_labels = []
for filepath in tqdm(tf.io.gfile.glob(not_stop_path)):
    not_stop_images.append(load_and_preprocess_image(filepath))
    not_stop_labels.append(0)
for filepath in tqdm(tf.io.gfile.glob(stop_path)):
    stop_images.append(load_and_preprocess_image(filepath))
    stop_labels.append(1)

# Concatenate images and labels
images = not_stop_images + stop_images
labels = not_stop_labels + stop_labels

# Display one image from each class with label
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(not_stop_images[0])
plt.title('Not Stop')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(stop_images[0])
plt.title('Stop')
plt.axis('off')
plt.show()


# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Define image dimensions
IMG_WIDTH = 96
IMG_HEIGHT = 96

# Define file paths for dataset
not_stop_path = './CustomDataset/not_stop/*.jpg'
stop_path = './CustomDataset/stop/*.jpg'

# Define function to load and preprocess image
def load_and_preprocess_image(filepath):
    # Load image as grayscale
    img = load_img(filepath, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Convert image to numpy array
    img = img_to_array(img)
    # Normalize pixel values
    img = img / 255.0
    return img

# Load images and labels into lists
not_stop_images = []
stop_images = []
not_stop_labels = []
stop_labels = []
for filepath in tqdm(tf.io.gfile.glob(not_stop_path)):
    not_stop_images.append(load_and_preprocess_image(filepath))
    not_stop_labels.append(0)
for filepath in tqdm(tf.io.gfile.glob(stop_path)):
    stop_images.append(load_and_preprocess_image(filepath))
    stop_labels.append(1)

# Concatenate images and labels
images = not_stop_images + stop_images
labels = not_stop_labels + stop_labels

# Display one image from each class with label
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(not_stop_images[0], cmap='gray')
plt.title('Not Stop')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(stop_images[0], cmap='gray')
plt.title('Stop')
plt.axis('off')
plt.show()


# In[3]:


import random
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Convert data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)


# In[4]:


print(X_train.shape)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.test.gpu_device_name())

# Define the model architecture
model = Sequential([
    SeparableConv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    SeparableConv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    SeparableConv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[6]:


model.summary()


# In[7]:


# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[8]:


model.save("my_model.h5")

