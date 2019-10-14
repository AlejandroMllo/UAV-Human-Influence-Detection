from __future__ import print_function

from Functions.image_handling import find_files, load_images
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline_images

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 256
num_classes = 3
epochs = 16

# input image dimensions
img_rows, img_cols = 32, 32

# ------------ LOAD DATA -----------------------

shapes_dataset_path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/GEOMETRIC_SHAPES/data'
shapes_type = ['circle', 'square', 'triangle']
shapes_label = {'circle': 0, 'square': 1, 'triangle': 2}


def load_data(split='train'):
    data = dict()

    for shape in shapes_type:
        path = shapes_dataset_path + '/' + str(split) + '/' + str(shape) + '/'
        imgs = find_files(path)
        x = load_images(imgs, path)
        y = [shapes_label[shape]] * len(x)
        data[shape] = (x, y)

    return data


train_data = load_data(split='train')
val_data = load_data(split='validation')


# ------------ PRE-PROCESS DATA -----------------------

image_size = 32

lg_32 = laguerre_gauss_filter(image_size, 0.9)
ft_lg_32 = np.fft.fft2(lg_32)


def join_data(dataset):

    shapes, labels = [], []

    for shape in shapes_type:
        x, y = dataset[shape]
        shapes.extend(x)
        labels.extend(y)

    shapes = np.array(shapes)
    labels = np.array(labels)

    return shapes, labels


x_train, y_train = join_data(train_data)
x_val, y_val = join_data(val_data)

# -------- Fourier Pre-processing

x_train = ft_pipeline_images(ft_lg_32, x_train)
x_val = ft_pipeline_images(ft_lg_32, x_val)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape((x_train.shape[0], 1, img_rows, img_cols))
    x_val = x_val.reshape((x_val.shape[0], 1, img_rows, img_cols))
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1))
    x_val = x_val.reshape((x_val.shape[0], img_rows, img_cols, 1))
    input_shape = (img_rows, img_cols, 1)

# -------- Shuffle data
x_train, _, y_train, _ = \
  train_test_split(x_train, y_train, test_size=0.05, random_state=42)

x_val, _, y_val, _ = \
  train_test_split(x_val, y_val, test_size=0.05, random_state=42)


# ------------ MODEL -----------------------

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

print('output shapes', y_train.shape, y_val.shape)

model = Sequential(name='GeometricShapes_OriginalImages')
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_model(model, 'geometricShapes_fourierPreprocessedImages(MNIST_architecture)_v0.1')

print(model.summary())
