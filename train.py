#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image  # Added Image import

# Paths
img_size = 128
batch_size = 32
dataset_path = os.path.join(os.getcwd(), 'C:\\Users\\DELL\\Downloads\\archive\\brain_tumor_dataset')

# Data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='binary', subset='validation')

# Build Simple CNN Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=20)

# Save the model in the recommended .keras format
model.save("model.keras")


# Provide a download link for the model
FileLink("model.keras")


# In[ ]:




