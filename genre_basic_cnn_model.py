import numpy as np 
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import os

import librosa
from librosa.feature import mfcc
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
print("1")
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.preprocessing.image import image_dataset_from_directory

print("1")
print("1")
import tensorflow as tf
print("1")


from glob import glob 
import wave
import random




# settings for images and training
IMG_HEIGHT = IMG_WIDTH = 256
BATCH_SIZE = 8
N_CHANNELS = 1
N_CLASSES = 11
seed_train_validation = 1
shuffle_value = True
validation_split = 0.2
directory = './spectrograms'

print("here")
# split data into train, valid, test
train_ds = mage_dataset_from_directory(
directory = directory,
image_size = (IMG_HEIGHT, IMG_WIDTH),
validation_split = validation_split,
subset = "training",
seed = seed_train_validation,
color_mode = 'grayscale',
shuffle = shuffle_value)

val_ds = image_dataset_from_directory(
directory = directory,
image_size = (IMG_HEIGHT, IMG_WIDTH),
validation_split = validation_split,
subset = "validation",
seed = seed_train_validation,
color_mode = 'grayscale',
shuffle = shuffle_value)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take((2*val_batches) // 3)
val_ds = val_ds.skip((2*val_batches) // 3)

## Make Baseline CNN Model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)))
# 1st convolutional block
model.add(tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 2nd convolutional block
model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 3rd convolutional block
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#4th convolutional block

model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dense layer 1
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
# Dense layer 2 (softmax)
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)
print("here2")

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save =tf.keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
print("here3")
history = model.fit(train_ds, batch_size = 2, steps_per_epoch=300, epochs=6, validation_data=val_ds,callbacks=[earlyStopping, mcp_save, reduce_lr_loss],)
model.save('./classification_model')

history_dict = history.history
val_acc_values = history_dict['val_accuracy']
val_loss = history_dict['val_loss']
epochs = range(1, len(val_acc_values)+1)

plt.plot(epochs, val_acc_values, label='Validation accuracy')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

final_loss, final_acc = model.evaluate(test_ds, verbose=0)
print("Final test loss: {0:.6f}, final test accuracy: {1:.6f}".format(final_loss, final_acc))



