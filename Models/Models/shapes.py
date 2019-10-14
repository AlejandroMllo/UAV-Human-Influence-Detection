from __future__ import print_function

from Functions.image_handling import find_files, load_images
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
from sklearn.model_selection import train_test_split


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

def get_line_profiles(x):

    x_profile, y_profile = ft_pipeline(ft_lg_32, x)
    return np.abs(np.concatenate((x_profile, y_profile), axis=1))


x_train = get_line_profiles(x_train)
x_val = get_line_profiles(x_val)


# -------- Shuffle data
x_train, _, y_train, _ = \
  train_test_split(x_train, y_train, test_size=0.05, random_state=42)

x_val, _, y_val, _ = \
  train_test_split(x_val, y_val, test_size=0.05, random_state=42)


# ------------ MODEL -----------------------

num_classes = 3

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = Sequential(name='GeometricShapes_LineProfiles')
model.add(Dense(64, activation='relu', input_dim=64, name='my1'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.RMSprop(),
    metrics=['accuracy']
)

epochs = 16
batch_size = 16

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val)
)
score = model.evaluate(x_val, y_val, verbose=0)

print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

save_model(model, 'geometricShapes_lineProfiles_v0.1')
print(model.summary())