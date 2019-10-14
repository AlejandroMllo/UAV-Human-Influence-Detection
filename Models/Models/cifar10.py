from __future__ import print_function

from Functions.image_handling import find_files, load_images
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

from sklearn.preprocessing import normalize

import numpy as np

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


# ------------ LOAD DATA -----------------------

# The data, split between train and test sets:
(x_train_org, y_train_org), (x_test_org, y_test_org) = cifar10.load_data()


# ------------ PRE-PROCESS DATA -----------------------

lg_filter_32 = laguerre_gauss_filter(32, 0.9)
ft_lg_32 = np.fft.fft2(lg_filter_32)

x_pr_train, y_pr_train = ft_pipeline(ft_lg_32, x_train_org)
x_pr_test, y_pr_test = ft_pipeline(ft_lg_32, x_test_org)

x_train = np.abs(np.concatenate((x_pr_train, y_pr_train), axis=1))
x_test = np.abs(np.concatenate((x_pr_test, y_pr_test), axis=1))

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train_org, num_classes)
y_test = keras.utils.to_categorical(y_test_org, num_classes)


# ------------ MODEL -----------------------


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=64, name='my1'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu', name='my2'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', name='my3'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu', name='my4'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu', name='my5'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='my6'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='my100'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='my600'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='my50000'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', name='my60000'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', name='my7'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu', name='my8'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu', name='my9'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax', name='output_layer'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001)#lr=0.01, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

x_train = normalize(x_train)
x_test = normalize(x_test)

cut = 10000

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])