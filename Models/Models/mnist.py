from __future__ import print_function

from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout

import numpy as np


# ------------ LOAD DATA -----------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ------------ PRE-PROCESS DATA -----------------------

image_size = 28

lg_28 = laguerre_gauss_filter(image_size, 0.9)
ft_lg_28 = np.fft.fft2(lg_28)

x_pr_train, y_pr_train = ft_pipeline(ft_lg_28, x_train)
x_pr_test, y_pr_test = ft_pipeline(ft_lg_28, x_test)

x_train = np.concatenate((x_pr_train, y_pr_train), axis=1)
x_test = np.concatenate((x_pr_test, y_pr_test), axis=1)

num_classes = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ------------ MODEL -----------------------

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=56, name='my1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='my2'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name='my3'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu', name='my10'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu', name='my4'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

epochs = 32
batch_size = 128

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss (x_profiles):', score[0])
print('Test accuracy (x_profiles):', score[1])

save_model(model, 'mnist_lineProfile_v0.1')