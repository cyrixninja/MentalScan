#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:08:55 2020

@author: dristishah
"""


import os
#GREYSCALE CODE
from os import listdir,makedirs
from os.path import isfile,join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

TRAINING_DIR = 'add training directory here'
VALIDATION_DIR = 'add validation directory here'

img_width=256; img_height=256
batch_size=64


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=(img_height, img_width)
                                                    )

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              target_size=(img_height, img_width)
                                                             )

callbacks = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
best_model_file = 'imagenet'
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose = 1, save_best_only = True)

resnet50_base = ResNet50(include_top=False, weights='imagenet',
                         input_tensor=None, input_shape=(img_width, img_height,3))


print('Adding new layers...')
output = Sequential()
output = resnet50_base.get_layer(index = -1).output  
output = Flatten()(output)
output = Dense(2048,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(7, activation='softmax')(output)
print('New layers added!')



resnet50_model = Model(resnet50_base.input, output)
for layer in resnet50_model.layers[:-7]:
    layer.trainable = False

resnet50_model.summary()


resnet50_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics ='accuracy')

history = resnet50_model.fit(train_generator,
                              epochs=12,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks = [callbacks, best_model]
                              )






