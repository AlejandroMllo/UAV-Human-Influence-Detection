import numpy as np
import cmath
from skimage import filters

from Functions.image_handling import rgb2grayscale


def kernel_transform(batch, kernel=np.fft.fft2):

    transformed_batch = []

    for instance in batch:

        instance = rgb2grayscale(instance)
        transformed = kernel(instance)
        transformed_batch.append(transformed)

    return np.array(transformed_batch)


def convolve(transformed_kernel, batch):

    convolved_batch = []

    for instance in batch:
        conv = np.multiply(transformed_kernel, instance)
        convolved_batch.append(conv)

    return np.array(convolved_batch)


def shift(batch):

    shifted_batch = []

    for instance in batch:
        shifted = np.fft.fftshift(instance)
        shifted_batch.append(shifted)

    return np.array(shifted_batch)


def line_profile(batch, axis=1):
    # Axis 0: y axis
    # Axis 1: x axis
    # Equivalent axes definition to numpy.

    line_profiles = []

    num_instances = batch.shape[0]

    for i in range(num_instances):

        instance = batch[i]
        axis_length = instance.shape[1 - axis] - 1

        if axis == 0:
            line_profiles.append(instance[: ,axis_length//2])
        else:
            line_profiles.append(instance[axis_length//2 ,:])

    return np.array(line_profiles)


def inv_transform(batch, kernel=np.fft.ifft2):
    transformed_batch = []

    for instance in batch:
        transformed = kernel(instance)
        transformed_batch.append(transformed)

    return np.array(transformed_batch)


def threshold_image(px):
    return 1.0 if np.abs(px) > 0.001 else 0.0


def ft_pipeline(transformed_filter, sample):
    transformed = kernel_transform(sample)
    convolved = convolve(transformed_filter, transformed)
    shifted = shift(convolved)
    x_profile = line_profile(shifted, axis=1)
    y_profile = line_profile(shifted, axis=0)

    return x_profile, y_profile


def binarize(images):

    binarized = []

    for img in images:
        val = filters.threshold_otsu(img)
        img = np.where(img > val, 0,0, 1.0)
        binarized.append(img)

    return np.array(binarized)


def ft_pipeline_images(transformed_filter, sample):

    vectorized_phase = np.vectorize(cmath.phase)

    transformed = kernel_transform(sample)
    convolved = convolve(transformed_filter, transformed)
    inv_conv = inv_transform(convolved)
    images = binarize(vectorized_phase(inv_conv))

    return images
