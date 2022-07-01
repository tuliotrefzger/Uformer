from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross, Downsample, Upsample
from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import torchvision


# Utils
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def get_dimensions(img):
    height, width, _ = img.size()
    if height >= width:
        return height, width
    else:
        return width, height


def expand2square(img):
    height, width, channels = img.size()
    if width == height:
        return img
    elif width > height:
        result = torch.zeros(width, width, channels)
        if (width - height) % 2:
            padding_top = int((width - height + 1) / 2)
        else:
            padding_top = int((width - height) / 2)
        result[padding_top : (height + padding_top), :, :] = img
        return result
    else:
        result = torch.zeros(height, height, channels)
        if (height - width) % 2:
            padding_left = int((height - width + 1) / 2)
        else:
            padding_left = int((height - width) / 2)
        result[:, padding_left : (width + padding_left), :] = img
        return result


def return2original_size(square_img, largest_dimension, smallest_dimension):
    height, width, channels = square_img.size()
    if largest_dimension == "none":
        return square_img
    elif largest_dimension == "height":
        height = largest_dimension
        width = smallest_dimension
        if height % 2:
            padding_left = int((height - width + 1) / 2)
        else:
            padding_left = int((height - width) / 2)
        return square_img[padding_left : (width + padding_left), :, :]
    else:
        width = largest_dimension
        height = smallest_dimension
        if width % 2:
            padding_top = int((width - height + 1) / 2)
        else:
            padding_top = int((width - height) / 2)
        return square_img[:, padding_top : (height + padding_top), :]


# Beginning of the code

# clean = torch.from_numpy(np.float32(load_img('GT_SRGB_010.PNG')))
# noisy = torch.from_numpy(np.float32(load_img('NOISY_SRGB_010.PNG')))
clean = torch.from_numpy(np.float32(load_img("GT_SRGB_010_ROT.PNG")))
noisy = torch.from_numpy(np.float32(load_img("NOISY_SRGB_010_ROT.PNG")))

largest_dimension, smallest_dimension = get_dimensions(clean)


clean_expanded2square = expand2square(clean)
noisy_expanded2square = expand2square(noisy)
original_clean = return2original_size(
    clean_expanded2square, largest_dimension, smallest_dimension
)
original_noisy = return2original_size(
    noisy_expanded2square, largest_dimension, smallest_dimension
)


plt.figure()
plt.imshow(clean)
plt.title("Original clean image")

plt.figure()
plt.imshow(noisy)
plt.title("Original noisy image")

plt.figure()
plt.imshow(clean_expanded2square)
plt.title("Original image expanded to square")

plt.figure()
plt.imshow(noisy_expanded2square)
plt.title("Noisy image expanded to square")

plt.figure()
plt.imshow(original_clean)
plt.title("Return to the original clean image")

plt.figure()
plt.imshow(original_noisy)
plt.title("Return to the original noisy image")
