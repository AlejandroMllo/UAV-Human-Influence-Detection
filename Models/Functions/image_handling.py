import os

import matplotlib.image as mpimg
import numpy as np


def rgb2grayscale(rgb_img):

    rgb_img = np.array(rgb_img)

    if rgb_img.ndim != 3:
        return rgb_img

    img = np.zeros(rgb_img.shape)
    img[:, :, 0] = rgb_img[:, :, 0] * 0.2125  # RED
    img[:, :, 1] = rgb_img[:, :, 1] * 0.7154  # GREEN
    img[:, :, 2] = rgb_img[:, :, 2] * 0.0721  # BLUE

    return np.sum(img, axis=2)


def find_files(path):
    files = next(os.walk(path))[2]
    return np.array(sorted(
        files, key=lambda f: int("".join(list(filter(str.isdigit, f))))
    ))


def load_images(image_names, path):
    images = []

    for name in image_names:
        img_path = path + name
        img = mpimg.imread(img_path)
        img = rgb2grayscale(img)
        images.append(img)

    return images
